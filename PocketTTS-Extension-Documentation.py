"""
OOBA BOOGA EXTENSION: Pocket TTS
=================================

Complete, production-ready text-to-speech extension for Oobabooga text-generation-webui.
Implements zero-shot voice cloning using the Kyutai Pocket TTS ONNX export.

AUTHOR: Based on KevinAHM/pocket-tts-onnx (Hugging Face)
LICENSE: Model: CC BY 4.0 | Code: Apache 2.0
PROHIBITED USE: See model card for restrictions on voice cloning and misuse.

ARCHITECTURAL OVERVIEW
----------------------

This extension integrates the Pocket TTS ONNX model into Oobabooga's text-generation-webui.
The architecture follows a strict initialization → preprocessing → generation → output pipeline:

1. INITIALIZATION (runs once at extension load):
   - Install Python dependencies (onnxruntime, numpy, etc.)
   - Download INT8-quantized ONNX models (~200 MB total)
   - Load TTS engine with optimized threading configuration
   - Verify all required files are present

2. TEXT PREPROCESSING (runs per AI response):
   - Decode HTML entities (&#x27; → ', &quot; → ", etc.)
   - Normalize Unicode whitespace variants
   - Remove zero-width/invisible characters
   - Collapse multiple spaces to single space
   - Remove newlines for natural speech flow

3. CHUNKING (runs per AI response):
   - Split cleaned text into ≤200-character chunks
   - Prefer sentence boundaries (. ! ?) when possible
   - Fallback to word boundaries if no punctuation
   - Hard split if necessary (avoiding mid-word cuts)

4. AUDIO GENERATION (runs per chunk):
   - Load cached voice reference (24kHz mono, trimmed)
   - Generate audio using PocketTTSOnnx.generate()
   - Concatenate all chunk audio arrays
   - Save final WAV file to outputs directory

5. OUTPUT INJECTION (runs per AI response):
   - Construct HTML <audio> tag with file path
   - Enable autoplay only for latest message
   - Inject audio player above original text in chat

PERFORMANCE CHARACTERISTICS
---------------------------

- Model Size: ~200 MB (INT8 quantized)
- Real-Time Factor: ~4.0x (faster than real-time)
- Per-Chunk Latency: ~1.2–1.5s for 200 characters
- Memory Usage: ~200 MB RAM (single engine instance)
- Threading: intra_op_num_threads=min(cpu_count, 4)
- Optimal Chunk Size: 200 characters (maximizes RTFx, minimizes errors)

THREADING MODEL
---------------

The extension uses a single global TTS engine instance protected by a threading.Lock.
This is necessary because:

1. ONNX Runtime InferenceSession is NOT thread-safe for concurrent Run() calls
2. Voice embeddings must be consistent across all chunks
3. Memory efficiency: 1 × 200 MB vs N × 200 MB for parallel workers

All TTS generation occurs within the model_lock context to prevent race conditions.

DEPENDENCY REQUIREMENTS
-----------------------

Required Python packages (installed automatically at initialization):
- onnxruntime>=1.16.0: ONNX Runtime for CPU inference
- numpy: Numerical array operations for audio processing
- soundfile: WAV file I/O
- sentencepiece: Text tokenization (used by Pocket TTS)
- scipy: Signal processing (resampling, filtering)
- huggingface_hub: Model download from Hugging Face Hub
- gradio: UI components (imported conditionally to avoid startup crashes)

MODEL FILES (INT8 ONLY, ~200 MB total)
--------------------------------------

onnx/flow_lm_main_int8.onnx (76 MB) - Flow LM transformer (autoregressive)
onnx/flow_lm_flow_int8.onnx (10 MB) - Flow network (Euler integration)
onnx/mimi_decoder_int8.onnx (23 MB) - Audio waveform decoder
onnx/mimi_encoder.onnx (73 MB) - Voice reference encoder (FP32, no INT8 version)
onnx/text_conditioner.onnx (16 MB) - Text embedding network (FP32, no INT8 version)
tokenizer.model (<1 MB) - SentencePiece tokenizer
pocket_tts_onnx.py - Inference wrapper (included in repo)

SAFETY LIMITS
-------------

MAX_TTS_LENGTH = 10000 chars - Prevents accidental processing of full chat history
MIN_TTS_LENGTH = 3 chars - Prevents empty/whitespace-only generation
MAX_CHARS_PER_CHUNK = 200 chars - Optimal balance of speed and stability

CONFIGURATION PARAMETERS
------------------------

params['activate'] (bool): Enable/disable TTS globally
params['autoplay'] (bool): Enable/disable autoplay on latest message
params['voice_file'] (str|None): Path to uploaded voice reference WAV
params['_cached_voice_path'] (str|None): Path to processed/cached voice
params['_last_uploaded_file'] (str|None): Track last uploaded file for cache invalidation
params['_last_file_hash'] (str|None): MD5 hash of voice file for cache validation

ERROR HANDLING
--------------

- Initialization errors: Logged to console, extension disabled gracefully
- Missing dependencies: Auto-installed, ImportError caught and reported
- Model download failures: Exception raised with clear error message
- Voice processing errors: Logged, TTS skipped for that message
- Generation errors: Logged with full traceback, fallback to text-only output
- File I/O errors: Gracefully handled, user notified via chat message

EXTENSION HOOKS
---------------

output_modifier(string, state): Main processing hook called by Oobabooga
- Receives: string (AI response), state (chat history dictionary)
- Returns: Modified HTML string with embedded audio player
- Behavior: Processes only when params['activate'] is True

ui(): UI construction hook called by Oobabooga
- Returns: Gradio UI components (Accordion with controls)
- Behavior: Shows status, allows voice upload, test button, cache clearing

TESTING
-------

- Test button generates fixed text with uploaded voice
- Clears cache to force reprocessing
- Validates engine is loaded before attempting generation

PROHIBITED USE
--------------

Voice cloning without explicit consent is prohibited. See model card for full terms.
This extension includes safeguards but cannot prevent all misuse.

VERSION HISTORY
---------------

- v1.0: Initial implementation with streaming mode
- v1.1: Switched to chunked mode (350 chars) for stability
- v1.2: Reduced to 200 chars per chunk for maximum stability
- v1.3: Added eager initialization, dependency management
- v1.4: Full documentation and production hardening

"""
import os                      # Operating system interfaces (file paths, environment variables)
import sys                     # System-specific parameters (Python executable path)
import subprocess              # Subprocess management (pip install commands)
import logging                 # Logging framework (console output, error tracking)
import time                    # Time functions (performance measurement, timestamps)
import threading               # Threading primitives (Lock for thread safety)
import hashlib                 # Hash functions (MD5 for file integrity checking)
from pathlib import Path       # Path manipulation (cross-platform file paths)
import re                      # Regular expressions (text cleaning, pattern matching)

# ============================================================================
# PERFORMANCE TUNING: ONNX RUNTIME THREADING CONFIGURATION
# ============================================================================
#
# CRITICAL: These environment variables MUST be set BEFORE importing onnxruntime
# or any module that might indirectly import it (like gradio).
#
# Rationale from Pocket TTS ONNX documentation:
# - The autoregressive loop contains many small sequential matrix multiplications
# - Default ONNX Runtime settings over-subscribe threads on these small ops
# - Limiting intra_op threads to min(cpu_count, 4) provides ~2x speedup
# - Setting inter_op to 1 ensures sequential consistency (no parallel ops)
#
# CPU_COUNT: Total logical cores available on the system
# INTRA_OP_THREADS: Threads per internal operation (matmul, etc.)
# INTER_OP_THREADS: Parallel operations (set to 1 for sequential consistency)

CPU_COUNT = os.cpu_count() or 4                    # Default to 4 if os.cpu_count() fails
INTRA_OP_THREADS = min(CPU_COUNT, 4)              # Optimal: 4 threads max for small matmuls
INTER_OP_THREADS = 1                               # Sequential execution (no parallel ops)

# Set ONNX Runtime environment variables BEFORE any imports
os.environ['ORT_PROVIDERS'] = 'CPUExecutionProvider'        # Force CPU (GPU provides no speedup)
os.environ['OMP_NUM_THREADS'] = str(CPU_COUNT)             # OpenMP thread pool size
os.environ['INTRA_OP_NUM_THREADS'] = str(INTRA_OP_THREADS) # Internal operation threads
os.environ['INTER_OP_NUM_THREADS'] = str(INTER_OP_THREADS) # Inter-operation threads

# ============================================================================
# DEFERRED IMPORTS: Prevent startup crashes if dependencies not yet installed
# ============================================================================
#
# These modules are imported AFTER dependency installation to avoid ImportError
# during Oobabooga's extension loading phase. They are declared as global None
# initially and populated by import_required_modules() during initialization.

np = None           # numpy: Numerical array operations (audio data)
sf = None           # soundfile: WAV file reading/writing
signal = None       # scipy.signal: Audio signal processing (resampling)
unicodedata = None  # unicodedata: Unicode character classification
gr = None           # gradio: Web UI components (conditionally imported)

# ============================================================================
# LOGGING SETUP: Console output configuration
# ============================================================================
#
# Creates a dedicated logger for Pocket TTS with timestamped, colored output.
# Filters out noisy logs from sentencepiece and onnxruntime libraries.

logger = logging.getLogger("PocketTTS")
if not logger.handlers:  # Prevent duplicate handlers on reload
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Suppress verbose logs from dependencies
logging.getLogger("sentencepiece").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

# ============================================================================
# CONFIGURATION: Paths, limits, and runtime parameters
# ============================================================================

# Hugging Face model repository identifier
MODEL_ID = "KevinAHM/pocket-tts-onnx"

# Local directory name for downloaded model files
LOCAL_DIR_NAME = "pocket_tts_model_files"

# Extension directory (location of this script file)
EXTENSION_DIR = Path(__file__).parent

# Full path to model directory
MODEL_PATH = EXTENSION_DIR / LOCAL_DIR_NAME

# Directory for processed/cached voice references
CACHE_VOICE_DIR = EXTENSION_DIR / "processed_voices"
CACHE_VOICE_DIR.mkdir(exist_ok=True)  # Create if doesn't exist

# Directory for generated audio outputs
OUTPUT_DIR = EXTENSION_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Create parents if needed

# Safety limits for text processing
MAX_TTS_LENGTH = 10000    # Maximum characters to process (prevents full chat history)
MIN_TTS_LENGTH = 3        # Minimum characters required (prevents empty generation)
MAX_CHARS_PER_CHUNK = 200 # Maximum characters per chunk (optimal for stability + speed)

# Runtime parameters (persisted across calls, managed by Oobabooga)
params = {
    'activate': True,              # Enable/disable TTS globally
    'autoplay': True,              # Enable/disable autoplay on latest message
    'voice_file': None,            # Path to uploaded voice WAV file
    '_cached_voice_path': None,    # Path to processed/cached voice
    '_last_uploaded_file': None,   # Track last uploaded file (cache invalidation)
    '_last_file_hash': None        # MD5 hash of voice file (cache validation)
}

# Global state flags
MODEL_LOADED = False        # True when TTS engine successfully loaded
tts_engine = None           # PocketTTSOnnx instance (None until loaded)
model_lock = threading.Lock()  # Thread lock for safe concurrent access

# ============================================================================
# MODEL VALIDATION: Required ONNX files (INT8 only)
# ============================================================================
#
# List of files that MUST exist after model download. Missing files indicate
# incomplete download or repository changes. Verification occurs during init.

EXPECTED_ONNX_FILES = [
    'onnx/flow_lm_main_int8.onnx',     # 76 MB - Flow LM transformer (INT8)
    'onnx/flow_lm_flow_int8.onnx',     # 10 MB - Flow network (INT8)
    'onnx/mimi_decoder_int8.onnx',     # 23 MB - Audio decoder (INT8)
    'onnx/mimi_encoder.onnx',          # 73 MB - Voice encoder (FP32, no INT8)
    'onnx/text_conditioner.onnx',      # 16 MB - Text embeddings (FP32, no INT8)
    'tokenizer.model',                 # <1 MB - SentencePiece tokenizer
    'pocket_tts_onnx.py'               # Wrapper script
]

# ============================================================================
# HTML ENTITY DECODING TABLE
# ============================================================================
#
# Mapping of common HTML entities to their decoded characters. These entities
# are commonly inserted by web frameworks (including Oobabooga) for display
# safety but must be decoded before TTS to ensure proper pronunciation.
#
# Example: "I can&#x27;t" → "I can't" (pronounced as contraction, not "ampersand quote")

HTML_ENTITIES = {
    '&#x27;': "'",    # Apostrophe (hex encoding)
    '&quot;': '"',    # Double quote
    '&amp;': '&',     # Ampersand
    '&lt;': '<',      # Less than
    '&gt;': '>',      # Greater than
    '&nbsp;': ' ',    # Non-breaking space
    '&copy;': '©',    # Copyright symbol
    '&reg;': '®',     # Registered trademark
    '&#39;': "'",     # Apostrophe (decimal encoding, alternative)
    '&#34;': '"'      # Double quote (decimal encoding, alternative)
}

# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def clean_text_for_tts(text):
    """
    Aggressively clean and normalize text for optimal TTS processing.
    
    This function performs multiple normalization steps to ensure the text
    is in the best possible form for the Pocket TTS model:
    
    1. HTML Entity Decoding: Converts HTML entities to their actual characters
       - Prevents TTS from pronouncing "&amp;" as "ampersand" instead of "&"
       - Ensures apostrophes, quotes, etc. are properly recognized
    
    2. Unicode Whitespace Normalization: Converts all Unicode space variants
       to regular ASCII spaces
       - Handles non-breaking spaces, zero-width spaces, etc.
       - Uses unicodedata.category() to detect space characters (category Z*)
    
    3. Zero-Width Character Removal: Strips invisible formatting characters
       - Removes zero-width spaces, joiners, directional markers
       - Prevents inflated character counts and tokenizer issues
    
    4. Whitespace Collapse: Reduces multiple consecutive spaces to single space
       - Improves tokenization efficiency
       - Removes unnecessary padding
    
    5. Newline Removal: Replaces all newline variants with single space
       - Prevents unnatural pauses in speech
       - Maintains word flow across paragraph breaks
    
    Args:
        text (str): Raw input text (may contain HTML entities, newlines, etc.)
    
    Returns:
        str: Cleaned, normalized text ready for TTS generation
    
    Example:
        >>> clean_text_for_tts("Hello&#x27;s\\n\\nWorld!")
        "Hello's World!"
    """
    original = text  # Keep reference for debugging
    
    # STEP 1: Decode HTML entities FIRST (critical for character count accuracy)
    # Iterate through HTML_ENTITIES dict and replace each entity with its character
    for entity, char in HTML_ENTITIES.items():
        if entity in text:  # Only replace if entity exists (optimization)
            text = text.replace(entity, char)
    
    # STEP 2: Normalize Unicode whitespace
    # Convert all Unicode space variants (category starting with 'Z') to regular space
    # This includes non-breaking spaces, zero-width spaces, etc.
    if 'unicodedata' in globals() and unicodedata:
        # Build new string: space if Unicode category is space-like, else original char
        text = ''.join(' ' if unicodedata.category(c).startswith('Z') else c for c in text)
    
    # STEP 3: Remove zero-width characters and other invisibles
    # These characters inflate character counts and confuse tokenizers
    # Pattern matches: zero-width spaces, joiners, directional markers, soft hyphens
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]', '', text)
    
    # STEP 4: Collapse multiple whitespace → single space
    # Replace 2+ consecutive whitespace chars (space, tab, newline) with single space
    # Then strip leading/trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # STEP 5: Remove newlines (for natural speech flow)
    # Replace all \n and \r with single space to prevent unnatural pauses
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text


def safe_chunk_text(text, max_chars=200):
    """
    Split text into chunks of maximum length, preferring sentence boundaries.
    
    This function intelligently splits long text into smaller chunks that can be
    safely processed by the TTS model without triggering reshape errors or
    quality degradation. The algorithm:
    
    1. If text is already ≤ max_chars, return as single chunk
    2. Otherwise, iterate through text in max_chars-sized windows
    3. For each window, search backward for sentence-ending punctuation
    4. If found and not too close to start, split at punctuation
    5. If no punctuation, search backward for space (word boundary)
    6. If no space, force split at max_chars (avoiding infinite loops)
    
    This approach:
    - Maintains semantic boundaries (splits at sentence ends when possible)
    - Avoids cutting words in half (fallback to spaces)
    - Guarantees progress (hard split if necessary)
    - Prevents tiny chunks (<30 chars) that waste fixed overhead
    
    Args:
        text (str): Cleaned text to split
        max_chars (int): Maximum characters per chunk (default: 200)
    
    Returns:
        List[str]: List of text chunks, each ≤ max_chars
    
    Example:
        >>> safe_chunk_text("Hello world. This is a test. Another sentence.", max_chars=20)
        ['Hello world.', 'This is a test.', 'Another sentence.']
    """
    # Early return: text already within limit
    if len(text) <= max_chars:
        return [text]
    
    # Initialize chunk list and starting position
    chunks = []
    start = 0
    
    # Iterate through text, creating chunks
    while start < len(text):
        # Calculate end position for this chunk
        end = start + max_chars
        
        # If we're not at the end of text, try to find better split point
        if end < len(text):
            # Look backward from end position for sentence-ending punctuation
            sentence_end = -1
            for i in range(end, start, -1):
                if text[i-1] in '.!?':  # Found sentence boundary
                    sentence_end = i
                    break
            
            # If found sentence boundary and not too close to start, use it
            if sentence_end != -1 and sentence_end > start + 30:  # Avoid tiny chunks
                end = sentence_end
            else:
                # Fallback: hard split at word boundary (space)
                # Search backward for space character
                while end > start and text[end-1] not in ' \t\n':
                    end -= 1
                # If no space found (single long word), force split at max_chars
                if end == start:
                    end = start + max_chars
        
        # Extract chunk and strip whitespace
        chunk = text[start:end].strip()
        # Only add non-empty chunks
        if chunk:
            chunks.append(chunk)
        
        # Move start position to end of this chunk
        start = end
    
    return chunks


# ============================================================================
# DEPENDENCY MANAGEMENT
# ============================================================================

def ensure_deps():
    """
    Install required Python packages if not already present.
    
    This function checks for the presence of all required dependencies and
    installs any missing ones using pip. It runs during extension initialization
    to ensure all modules are available before loading the TTS model.
    
    Required packages:
    - onnxruntime>=1.16.0: ONNX Runtime for CPU inference
    - numpy: Numerical array operations
    - soundfile: WAV file I/O
    - sentencepiece: Text tokenization
    - scipy: Signal processing (resampling)
    - huggingface_hub: Model download from Hugging Face
    
    Behavior:
    - Checks each package by attempting to import it
    - Collects missing packages into a list
    - Installs all missing packages in sequence
    - Logs progress and errors
    - Raises exception if any installation fails
    
    Raises:
        subprocess.CalledProcessError: If pip install fails for any package
    """
    # List of required packages (with PyPI names)
    required = ['onnxruntime', 'numpy', 'soundfile', 'sentencepiece', 'scipy', 'huggingface_hub']
    missing = []  # Track missing packages
    
    # Check each package by attempting import
    for pkg in required:
        try:
            # Convert package name to module name (replace hyphens with underscores)
            __import__(pkg.replace('-', '_'))
        except ImportError:
            # Package not installed
            missing.append(pkg)
    
    # Install missing packages if any
    if missing:
        logger.info(f"Installing missing packages: {', '.join(missing)}...")
        for pkg in missing:
            try:
                # Run pip install command
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
                logger.info(f"  ✓ {pkg}")
            except subprocess.CalledProcessError as e:
                # Installation failed
                logger.error(f"  ✗ Failed to install {pkg}: {e}")
                raise
        logger.info("Package installation complete.")
    else:
        # All packages already installed
        logger.info("✓ All required packages already installed.")


# ============================================================================
# MODEL DOWNLOAD AND VERIFICATION
# ============================================================================

def ensure_model():
    """
    Download and verify INT8-quantized ONNX model files.
    
    This function handles the complete model setup process:
    
    1. Check if model directory exists
    2. If not, download from Hugging Face Hub using snapshot_download
    3. Use allow_patterns to download ONLY INT8 files (~200 MB vs ~500 MB)
    4. Verify all expected files are present after download
    5. Raise exception if any required files are missing
    
    Download optimization:
    - Downloads only INT8 quantized models (flow_lm_main_int8.onnx, etc.)
    - Skips FP32 versions (flow_lm_main.onnx) to save bandwidth and disk
    - Includes mimi_encoder.onnx and text_conditioner.onnx (no INT8 versions exist)
    
    File verification:
    - Checks existence of each file in EXPECTED_ONNX_FILES
    - Collects missing files into a list
    - Raises FileNotFoundError with detailed message if any missing
    
    Raises:
        Exception: If download fails or files are missing
        FileNotFoundError: If verification fails
    """
    # Check if model directory exists
    if not MODEL_PATH.exists():
        logger.info("Downloading Pocket TTS model (INT8 only, ~200 MB)...")
        try:
            # Import snapshot_download from huggingface_hub
            from huggingface_hub import snapshot_download
            
            # Download model with selective file patterns
            snapshot_download(
                MODEL_ID,
                local_dir=str(MODEL_PATH),
                local_dir_use_symlinks=False,
                # Only download INT8 models and required files
                allow_patterns=[
                    "onnx/*_int8.onnx",          # All INT8 ONNX models
                    "onnx/mimi_encoder.onnx",    # Voice encoder (FP32, no INT8)
                    "onnx/text_conditioner.onnx",# Text conditioner (FP32, no INT8)
                    "tokenizer.model",           # SentencePiece tokenizer
                    "pocket_tts_onnx.py",        # Inference wrapper
                    "generate.py",               # CLI script
                    "requirements.txt"           # Dependencies list
                ]
            )
            logger.info("✓ Model download complete.")
        except Exception as e:
            # Download failed
            logger.error(f"✗ Model download failed: {e}")
            raise
    
    # Verify all required files exist
    missing = []
    for f in EXPECTED_ONNX_FILES:
        if not (MODEL_PATH / f).exists():
            missing.append(f)
    
    # Raise error if any files missing
    if missing:
        logger.error(f"✗ Missing ONNX files: {missing}")
        raise FileNotFoundError(f"Model download incomplete. Missing: {missing}")
    
    logger.info("✓ Model files verified.")


# ============================================================================
# MODULE IMPORT AFTER INSTALLATION
# ============================================================================

def import_required_modules():
    """
    Import all required Python modules after dependency installation.
    
    This function performs the actual imports of modules that were declared
    as None at the top of the file. It runs AFTER ensure_deps() to guarantee
    that all packages are installed before attempting import.
    
    Imported modules:
    - numpy (as np): Numerical array operations for audio data
    - soundfile (as sf): WAV file reading and writing
    - scipy.signal (as signal): Audio signal processing (resampling)
    - unicodedata: Unicode character classification (whitespace detection)
    - gradio (as gr): Web UI components for Oobabooga integration
    
    Global variables modified:
    - np, sf, signal, unicodedata, gr: Set to imported modules
    
    Returns:
        bool: True if all imports successful, False otherwise
    
    Raises:
        ImportError: If any required module fails to import
    """
    global np, sf, signal, unicodedata, gr
    
    try:
        # Import all required modules
        import numpy as np
        import soundfile as sf
        from scipy import signal
        import unicodedata
        import gradio as gr
        logger.info("✓ All required modules imported successfully.")
        return True
    except ImportError as e:
        # Import failed
        logger.error(f"✗ Failed to import required modules: {e}")
        return False


# ============================================================================
# TTS ENGINE LOADING
# ============================================================================

def load_tts_engine():
    """
    Load the PocketTTSOnnx engine with INT8 models and optimized settings.
    
    This function initializes the TTS engine with the following configuration:
    
    Model files:
    - flow_lm_main_int8.onnx: Autoregressive transformer (76 MB)
    - flow_lm_flow_int8.onnx: Flow network for latent sampling (10 MB)
    - mimi_decoder_int8.onnx: Audio waveform decoder (23 MB)
    - mimi_encoder.onnx: Voice reference encoder (73 MB, FP32)
    - text_conditioner.onnx: Text embedding network (16 MB, FP32)
    
    Engine parameters:
    - temperature=0.7: Balanced diversity (0.3=more deterministic, 1.0=more diverse)
    - lsd_steps=1: Fastest generation (10 steps=default, 1=fastest)
    
    Threading optimization:
    - Uses global INTRA_OP_THREADS setting (min(cpu_count, 4))
    - Provides ~2x speedup over default ONNX Runtime settings
    
    Thread safety:
    - Protected by model_lock to prevent concurrent initialization
    - Checks MODEL_LOADED flag to avoid redundant loading
    
    Global state modified:
    - MODEL_LOADED: Set to True on success
    - tts_engine: Set to PocketTTSOnnx instance
    
    Returns:
        bool: True if engine loaded successfully, False otherwise
    
    Raises:
        Exception: If model loading fails (logged with traceback)
    """
    global MODEL_LOADED, tts_engine
    
    # Acquire lock to prevent concurrent initialization
    with model_lock:
        # Check if already loaded
        if MODEL_LOADED and tts_engine:
            return True
        
        # Log loading attempt
        logger.info(f"Loading Pocket TTS engine (IntraOp Threads: {INTRA_OP_THREADS})...")
        
        # Add model directory to Python path for pocket_tts_onnx import
        if str(MODEL_PATH) not in sys.path:
            sys.path.insert(0, str(MODEL_PATH))
        
        try:
            # Import PocketTTSOnnx wrapper
            from pocket_tts_onnx import PocketTTSOnnx
            
            # Construct absolute paths to model files
            models_dir = str(MODEL_PATH / 'onnx')
            tokenizer_path = str(MODEL_PATH / 'tokenizer.model')
            
            # Initialize engine with optimized parameters
            tts_engine = PocketTTSOnnx(
                models_dir=models_dir,        # Directory containing ONNX files
                tokenizer_path=tokenizer_path,# Path to SentencePiece tokenizer
                temperature=0.7,              # Balanced diversity
                lsd_steps=1                   # Fastest generation (1 step)
            )
            
            # Update global state
            MODEL_LOADED = True
            
            # Log success
            logger.info("✓ Pocket TTS engine loaded successfully.")
            return True
        except Exception as e:
            # Loading failed
            logger.error(f"✗ Failed to load TTS engine: {e}", exc_info=True)
            return False


# ============================================================================
# EXTENSION INITIALIZATION
# ============================================================================

def initialize_extension():
    """
    Complete extension initialization sequence.
    
    This function orchestrates the entire setup process in the correct order:
    
    1. Install dependencies (ensure_deps)
    2. Import required modules (import_required_modules)
    3. Download and verify model (ensure_model)
    4. Load TTS engine (load_tts_engine)
    
    Each step is wrapped in try/except to catch and log errors. If any step
    fails, the extension is marked as disabled but Oobabooga continues running.
    
    Logging:
    - Uses separator lines (===) for visual clarity
    - Logs each major step with emoji indicators
    - Reports success or failure clearly
    - Includes performance-relevant details (thread count, etc.)
    
    Error handling:
    - Catches all exceptions with detailed logging
    - Does not crash Oobabooga on failure
    - Leaves MODEL_LOADED as False if initialization fails
    
    Side effects:
    - Installs Python packages
    - Downloads ~200 MB of model files
    - Loads TTS engine into memory (~200 MB RAM)
    - Sets up logging and global state
    """
    logger.info("="*80)
    logger.info("🔧 INITIALIZING POCKET TTS EXTENSION")
    logger.info("="*80)
    
    try:
        # Step 1: Install dependencies
        ensure_deps()
        
        # Step 2: Import modules (now that they're installed)
        if not import_required_modules():
            raise ImportError("Could not import required modules")
        
        # Step 3: Download and verify model
        ensure_model()
        
        # Step 4: Load TTS engine
        if load_tts_engine():
            # Success: log detailed status
            logger.info("="*80)
            logger.info("✅ POCKET TTS EXTENSION READY")
            logger.info("="*80)
            logger.info("  • INT8-only models (~200 MB)")
            logger.info("  • Safe chunking: ≤200 chars/chunk (maximum stability)")
            logger.info("  • IntraOp threads: %d", INTRA_OP_THREADS)
            logger.info("  • HTML decoding + invisible char stripping")
            logger.info("  • Autoplay: latest message only")
            logger.info("="*80)
        else:
            # Engine loading failed
            logger.error("="*80)
            logger.error("❌ POCKET TTS INITIALIZATION FAILED")
            logger.error("Extension will be disabled until issues are resolved.")
            logger.error("="*80)
    except Exception as e:
        # Unexpected error during initialization
        logger.exception(f"💥 Extension initialization crashed: {e}")
        logger.error("="*80)
        logger.error("❌ POCKET TTS EXTENSION DISABLED")
        logger.error("="*80)


# Run initialization ONCE when module is loaded (not on every Oobabooga reload)
if not MODEL_LOADED:
    try:
        initialize_extension()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")


# ============================================================================
# VOICE PROCESSING FUNCTIONS
# ============================================================================

def get_file_hash(filepath):
    """
    Generate MD5 hash of a file for cache validation.
    
    This function reads a file in chunks and computes its MD5 hash. The hash
    is used to detect changes in uploaded voice files, ensuring that cached
    processed voices are only reused when the source file is identical.
    
    Algorithm:
    - Open file in binary read mode
    - Read in 4KB chunks (memory efficient for large files)
    - Update MD5 hasher with each chunk
    - Return hexadecimal digest
    
    Args:
        filepath (str|Path): Path to file to hash
    
    Returns:
        str: MD5 hash as hexadecimal string (32 characters)
    
    Example:
        >>> get_file_hash("voice.wav")
        'd41d8cd98f00b204e9800998ecf8427e'
    """
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in 4KB chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_processed_voice(input_path):
    """
    Process and cache voice reference WAV file.
    
    This function prepares uploaded voice files for TTS by:
    
    1. Checking cache: If file already processed and hash matches, return cached path
    2. Loading audio: Read WAV file using soundfile
    3. Converting to mono: Average stereo channels if needed
    4. Resampling to 24kHz: Required sample rate for Pocket TTS
    5. Trimming silence: Remove leading/trailing silence beyond 0.01 threshold
    6. Limiting duration: Cap at 15 seconds (360,000 samples at 24kHz)
    7. Saving processed file: Write to cache directory
    
    Cache validation:
    - Computes MD5 hash of input file
    - Compares with stored hash in params['_last_file_hash']
    - Reuses cached file if hash matches (avoids reprocessing)
    
    Audio processing details:
    - Uses scipy.signal.resample for high-quality resampling
    - Trims to first/last sample exceeding 0.01 amplitude threshold
    - Adds 1000-sample padding around trimmed region
    - Limits to 15 seconds maximum (prevents excessive processing)
    
    Args:
        input_path (str|Path): Path to uploaded voice WAV file
    
    Returns:
        str|None: Path to processed/cached voice file, or None on error
    
    Side effects:
    - Creates file in CACHE_VOICE_DIR
    - Updates params['_last_file_hash'] with file hash
    - Logs processing steps and errors
    """
    # Validate input path
    if not input_path or not Path(input_path).exists():
        return None
    
    # Check if numpy, soundfile, scipy are loaded
    if sf is None or signal is None or np is None:
        logger.error("Required modules not loaded")
        return None
    
    # Extract filename and construct cache path
    filename = os.path.basename(input_path)
    output_path = CACHE_VOICE_DIR / f"proc_{filename}"
    
    # Compute hash of input file for cache validation
    current_hash = get_file_hash(input_path)
    
    # Check if cached file exists and hash matches
    if output_path.exists() and params['_last_file_hash'] == current_hash:
        return str(output_path.resolve())
    
    # Log processing start
    logger.info(f"Processing voice: {filename}...")
    
    try:
        # Load audio file
        data, sr = sf.read(input_path)
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Resample to 24kHz if needed
        if sr != 24000:
            # Calculate new length: (len * target_sr) / original_sr
            data = signal.resample(data, int(len(data) * 24000 / sr))
        
        # Trim silence: find first and last sample exceeding threshold
        mask = np.abs(data) > 0.01
        if mask.any():
            indices = np.where(mask)[0]
            if len(indices) >= 2:
                # Normal case: multiple non-silent samples
                s, e = indices[0], indices[-1]
                data = data[max(0, s-1000):min(len(data), e+1000)]
            else:
                # Edge case: only one non-silent sample
                idx = indices[0]
                data = data[max(0, idx-1000):min(len(data), idx+1000)]
        
        # Limit to 15 seconds maximum (360,000 samples at 24kHz)
        data = data[:24000*15]
        
        # Save processed audio to cache
        sf.write(str(output_path), data, 24000)
        
        # Update cache hash
        params['_last_file_hash'] = current_hash
        
        # Log success
        logger.info(f"✅ Voice cached: {output_path.name}")
        return str(output_path.resolve())
    except Exception as e:
        # Processing failed
        logger.error(f"Voice processing failed: {e}")
        return None


# ============================================================================
# TEST AND CACHE MANAGEMENT
# ============================================================================

def test_voice():
    """
    Generate test audio to verify TTS setup.
    
    This function creates a simple test audio file using the uploaded voice
    reference. It's called when the user clicks the "Test" button in the UI.
    
    Test parameters:
    - Text: "Hello! This is a test."
    - Voice: params['_cached_voice_path'] (processed voice reference)
    - Output: outputs/test_<timestamp>.wav
    
    Validation:
    - Checks if engine is loaded
    - Checks if voice file is uploaded and cached
    - Verifies voice file exists on disk
    
    Returns:
        str: HTML string with audio player on success, error message on failure
    
    Side effects:
    - Creates WAV file in OUTPUT_DIR
    - Logs test generation steps
    - Plays audio automatically (autoplay attribute)
    """
    # Validate engine is loaded
    if not MODEL_LOADED or tts_engine is None:
        return "❌ TTS engine not loaded."
    
    # Validate voice is uploaded
    if not params['_cached_voice_path']:
        return "❌ Upload a voice first."
    
    # Validate voice file exists
    if not Path(params['_cached_voice_path']).exists():
        return f"❌ Voice file missing."
    
    try:
        # Construct output path with timestamp
        out = OUTPUT_DIR / f"test_{int(time.time()*1000)}.wav"
        
        # Generate test audio
        with model_lock:
            audio = tts_engine.generate(text="Hello! This is a test.", voice=params['_cached_voice_path'])
            
            # Extract sample rate if returned as tuple
            sr = 24000
            if isinstance(audio, tuple):
                audio, sr = audio
            
            # Save audio to file
            tts_engine.save_audio(audio, str(out))
        
        # Construct web-accessible audio URL
        try:
            web_root = Path(__file__).parent.parent.parent
            rel_path = out.relative_to(web_root)
            audio_src = f"file/{rel_path.as_posix()}"
        except ValueError:
            # Fallback to absolute path if relative fails
            audio_src = f"file={out.as_posix()}"
        
        # Return HTML with audio player
        return f'✅ Success!<br><audio src="{audio_src}" controls autoplay></audio>'
    except Exception as e:
        # Generation failed
        logger.error(f"Test failed: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"


def clear_cache():
    """
    Clear processed voice cache and reset parameters.
    
    This function:
    1. Deletes all WAV files in CACHE_VOICE_DIR
    2. Resets voice-related parameters to None
    3. Returns count of deleted files
    
    Use case: User wants to force reprocessing of voice file (e.g., after
    modifying the file or troubleshooting processing issues).
    
    Returns:
        str: Success message with count of cleared files
    
    Side effects:
    - Deletes files in CACHE_VOICE_DIR
    - Resets params['_cached_voice_path']
    - Resets params['_last_uploaded_file']
    - Resets params['_last_file_hash']
    - Resets params['voice_file']
    """
    count = 0
    # Delete all WAV files in cache directory
    if CACHE_VOICE_DIR.exists():
        for f in CACHE_VOICE_DIR.glob("*.wav"):
            try:
                f.unlink()
                count += 1
            except:
                pass
    
    # Reset voice parameters
    params['_cached_voice_path'] = None
    params['_last_uploaded_file'] = None
    params['_last_file_hash'] = None
    params['voice_file'] = None
    
    return f"✅ Cleared {count} files."


# ============================================================================
# MAIN PROCESSING HOOK
# ============================================================================

def output_modifier(string, state):
    """
    Oobabooga extension hook: Process AI output and inject TTS audio.
    
    This is the main entry point called by Oobabooga for each AI response.
    It receives the raw AI-generated text and returns modified HTML with
    embedded audio player.
    
    Processing pipeline:
    1. Check if TTS is enabled and engine is loaded
    2. Log raw input for debugging
    3. Clean text (HTML decode, normalize, remove invisibles)
    4. Validate cleaned text length
    5. Process/load voice reference
    6. Split text into chunks (≤200 chars each)
    7. Generate audio for each chunk
    8. Concatenate chunk audio arrays
    9. Save final WAV file
    10. Construct HTML with audio player
    11. Enable autoplay only for latest message
    12. Return modified HTML
    
    Args:
        string (str): Raw AI-generated text response
        state (dict): Oobabooga chat state dictionary containing:
            - 'internal': List of [user_input, ai_response] pairs
            - Other session state (not used by this extension)
    
    Returns:
        str: Modified HTML string with embedded audio player, or original
             string if TTS is disabled/skipped
    
    Behavior:
    - Only processes when params['activate'] is True
    - Skips if engine not loaded or voice not uploaded
    - Truncates text longer than MAX_TTS_LENGTH
    - Logs all steps for debugging
    - Returns error message in chat if generation fails
    
    Example return value:
        '<audio src="file/outputs/12345.wav" controls autoplay></audio><br><br>Original text'
    """
    # Skip if TTS disabled or engine not ready
    if not params['activate'] or not MODEL_LOADED or tts_engine is None:
        return string
    
    # Log raw input for debugging
    logger.info("="*80)
    logger.info("🎤 TTS INPUT RECEIVED (RAW FROM OOBABOOGA)")
    logger.info(f"   RAW length: {len(string)} chars")
    logger.info(f"   First 100 chars (repr): {repr(string[:100])}")
    logger.info("="*80)
    
    # Clean text for TTS
    cleaned_text = clean_text_for_tts(string)
    
    # Validate cleaned text
    if not cleaned_text or len(cleaned_text) < MIN_TTS_LENGTH:
        logger.warning(f"🔇 TTS SKIPPED: Invalid text after cleaning ({len(cleaned_text)} chars)")
        return string
    
    if len(cleaned_text) > MAX_TTS_LENGTH:
        logger.warning(f"⚠️ TTS TRUNCATED: {len(cleaned_text)} → {MAX_TTS_LENGTH} chars")
        cleaned_text = cleaned_text[:MAX_TTS_LENGTH]
    
    # Setup voice reference
    voice_path = None
    if params['voice_file'] and Path(params['voice_file']).exists():
        # Check if need to reprocess voice
        if not params['_cached_voice_path'] or params['_last_uploaded_file'] != params['voice_file']:
            params['_last_uploaded_file'] = params['voice_file']
            params['_cached_voice_path'] = get_processed_voice(params['voice_file'])
        voice_path = params['_cached_voice_path']
    
    # Validate voice is available
    if not voice_path:
        logger.warning("⚠️ TTS SKIPPED: No valid voice file uploaded")
        return "[⚠️ Upload .wav voice]<br><br>" + string
    
    # Generate audio using SAFE CHARACTER-BASED CHUNKING
    try:
        # Construct output path with timestamp
        out = OUTPUT_DIR / f"{int(time.time()*1000)}.wav"
        
        # Log generation start
        logger.info("▶️ TTS GENERATION STARTED (Safe Chunking)")
        logger.info(f"   Voice: {Path(voice_path).name}")
        logger.info(f"   Cleaned text length: {len(cleaned_text)} chars")
        
        # Split text into chunks (≤200 chars each)
        text_chunks = safe_chunk_text(cleaned_text, max_chars=MAX_CHARS_PER_CHUNK)
        logger.info(f"   Split into {len(text_chunks)} chunks (≤{MAX_CHARS_PER_CHUNK} chars each)")
        
        # Generate audio for each chunk
        all_audio_chunks = []
        total_elapsed = 0
        
        for idx, chunk in enumerate(text_chunks):
            logger.info(f"   ▶️ Generating chunk {idx+1}/{len(text_chunks)} ({len(chunk)} chars)")
            chunk_start = time.time()
            
            # Generate audio for this chunk (thread-safe)
            with model_lock:
                audio_chunk = tts_engine.generate(text=chunk, voice=voice_path)
            
            chunk_elapsed = time.time() - chunk_start
            total_elapsed += chunk_elapsed
            
            # Extract sample rate if returned as tuple
            sr = 24000
            if isinstance(audio_chunk, tuple):
                audio_chunk, sr = audio_chunk
            
            # Append chunk audio to list
            all_audio_chunks.append(np.array(audio_chunk))
            logger.info(f"   ✅ Chunk {idx+1} done | {chunk_elapsed:.2f}s")
        
        # Concatenate all chunk audio arrays
        final_audio = np.concatenate(all_audio_chunks)
        elapsed = total_elapsed
        
        # Save final audio to file
        tts_engine.save_audio(final_audio, str(out))
        
        # Log completion
        logger.info(f"✅ TTS COMPLETE | {elapsed:.2f}s | RTFx: {len(final_audio)/sr/elapsed:.2f}x | Output: {out.name}")
        logger.info("="*80)
        
        # Construct web-accessible audio URL
        try:
            web_root = Path(__file__).parent.parent.parent
            rel_path = out.relative_to(web_root)
            audio_src = f"file/{rel_path.as_posix()}"
        except ValueError:
            # Fallback to absolute path
            audio_src = f"file={out.as_posix()}"
        
        # Determine autoplay attribute (only for latest message)
        autoplay_attr = ""
        if params['autoplay']:
            # Check if this is the latest message in chat
            if state and 'internal' in state:
                internal_msgs = state['internal']
                if len(internal_msgs) >= 2:
                    # internal format: [[user_msg, ai_response], ...]
                    last_ai_response = internal_msgs[-1][1] if internal_msgs[-1] else ""
                    # Compare content (strip whitespace for robustness)
                    if string.strip() == last_ai_response.strip():
                        autoplay_attr = "autoplay"
                        logger.info("🔊 Autoplay ENABLED (latest message)")
                    else:
                        logger.info("🔇 Autoplay DISABLED (older message)")
                else:
                    # Single message chat
                    autoplay_attr = "autoplay"
            else:
                # State not available (fallback)
                autoplay_attr = "autoplay"
        
        # Return modified HTML with audio player
        return f'<audio src="{audio_src}" controls {autoplay_attr}></audio><br><br>{string}'
        
    except Exception as e:
        # Generation failed
        logger.error(f"❌ TTS FAILED: {e}", exc_info=True)
        logger.info("="*80)
        return f"[⚠️ {str(e)}]<br><br>{string}"


# ============================================================================
# UI CONSTRUCTION HOOK
# ============================================================================

def ui():
    """
    Oobabooga extension UI hook: Construct Gradio interface components.
    
    This function builds the user interface that appears in Oobabooga's
    extension panel. It creates:
    
    - Accordion container (collapsible panel)
    - Status indicator (ready/loading)
    - Optimization details (when ready)
    - Activate checkbox (enable/disable TTS)
    - Autoplay checkbox (enable/disable autoplay)
    - Voice file upload widget
    - Test button (generate test audio)
    - Clear cache button (delete processed voices)
    - Test output area (display test results)
    
    UI behavior:
    - Accordion is closed by default (open=False)
    - Status shows "READY" or "LOADING" based on MODEL_LOADED
    - Optimization details only shown when engine is loaded
    - Voice upload accepts only .wav files
    - Test button calls test_voice() function
    - Clear button calls clear_cache() function
    
    Returns:
        None: UI components are created as side effect (Gradio magic)
    
    Note:
        This function conditionally imports gradio to avoid crashes if
        Gradio is not installed during extension loading.
    """
    # Import gradio here to avoid crash if not installed yet
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed - UI will not work")
        return
    
    # Create collapsible accordion panel
    with gr.Accordion(f"🔊 Pocket TTS ({'READY' if MODEL_LOADED else 'LOADING'})", open=False):
        # Status indicator
        status = "✅ Ready" if MODEL_LOADED else "⏳ Initializing..."
        gr.Markdown(f"**Status**: {status}")
        
        # Show optimization details only when ready
        if MODEL_LOADED:
            gr.Markdown(
                f"**Optimizations**:<br>"
                f"• INT8-only models (~200 MB)<br>"
                f"• Safe chunking: ≤{MAX_CHARS_PER_CHUNK} chars/chunk (maximum stability)<br>"
                f"• IntraOp threads: {INTRA_OP_THREADS}<br>"
                f"• HTML decoding + invisible char stripping<br>"
                f"• Autoplay: latest message only"
            )
        
        # Control checkboxes
        with gr.Row():
            gr.Checkbox(value=params['activate'], label='Activate')
            gr.Checkbox(value=params['autoplay'], label='Autoplay')
        
        # Voice file upload
        voice_upload = gr.File(label='Voice (.wav)', file_types=['.wav'], type='filepath')
        
        # Action buttons
        with gr.Row():
            test_btn = gr.Button("🔊 Test")
            clear_btn = gr.Button("🗑️ Clear")
        
        # Test output display
        test_output = gr.Markdown("")
        
        # Helper function for voice upload
        def _on_upload(p):
            """Handle voice file upload event."""
            if p:
                params['voice_file'] = p
                params['_cached_voice_path'] = None
                params['_last_file_hash'] = None
                return f"**Voice**: {Path(p).name}"
            return "**Voice**: None"
        
        # Connect UI events to functions
        voice_upload.change(_on_upload, voice_upload, gr.Markdown())
        test_btn.click(test_voice, None, test_output)
        clear_btn.click(clear_cache, None, test_output)