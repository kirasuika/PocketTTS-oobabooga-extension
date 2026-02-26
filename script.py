""" OOBA BOOGA EXTENSION 
Pocket TTS - Safe Chunked Mode (200 chars/chunk)
- Downloads ONLY INT8 models (~200 MB).
- CHUNKED GENERATION: ≤200 characters per chunk (maximum stability).
- EAGER INITIALIZATION: All setup at startup (no lazy loading).
- HTML ENTITY DECODING + INVISIBLE CHAR STRIPPING.
- AUTPLAY FIX: Only latest message auto-plays.
- ZERO AUDIO POST-PROCESSING.
"""
import os
import sys
import subprocess
import logging
import time
import threading
import hashlib
from pathlib import Path
import re

# --- CRITICAL PERFORMANCE TUNIaNG ---
CPU_COUNT = os.cpu_count() or 4
INTRA_OP_THREADS = min(CPU_COUNT, 4)
INTER_OP_THREADS = 1

os.environ['ORT_PROVIDERS'] = 'CPUExecutionProvider'
os.environ['OMP_NUM_THREADS'] = str(CPU_COUNT)
os.environ['INTRA_OP_NUM_THREADS'] = str(INTRA_OP_THREADS)
os.environ['INTER_OP_NUM_THREADS'] = str(INTER_OP_THREADS)

# --- DEFERRED IMPORTS ---
np = None
sf = None
signal = None
unicodedata = None
gr = None

logger = logging.getLogger("PocketTTS")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- CONFIG ---
MODEL_ID = "KevinAHM/pocket-tts-onnx"
LOCAL_DIR_NAME = "pocket_tts_model_files"
EXTENSION_DIR = Path(__file__).parent
MODEL_PATH = EXTENSION_DIR / LOCAL_DIR_NAME
CACHE_VOICE_DIR = EXTENSION_DIR / "processed_voices"
CACHE_VOICE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = EXTENSION_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SAFETY LIMITS
MAX_TTS_LENGTH = 10000
MIN_TTS_LENGTH = 3
MAX_CHARS_PER_CHUNK = 200  # Reduced from 350 → 200 for maximum stability

params = {
    'activate': True,
    'autoplay': True,
    'voice_file': None,
    '_cached_voice_path': None,
    '_last_uploaded_file': None,
    '_last_file_hash': None
}

MODEL_LOADED = False
tts_engine = None
model_lock = threading.Lock()

# --- VALIDATION (INT8 ONLY) ---
EXPECTED_ONNX_FILES = [
    'onnx/flow_lm_main_int8.onnx',
    'onnx/flow_lm_flow_int8.onnx',
    'onnx/mimi_decoder_int8.onnx',
    'onnx/mimi_encoder.onnx',
    'onnx/text_conditioner.onnx',
    'tokenizer.model',
    'pocket_tts_onnx.py'
]

# --- HTML ENTITY DECODING ---
HTML_ENTITIES = {
    '&#x27;': "'",
    '&quot;': '"',
    '&amp;': '&',
    '&lt;': '<',
    '&gt;': '>',
    '&nbsp;': ' ',
    '&copy;': '©',
    '&reg;': '®',
    '&#39;': "'",   # Alternative apostrophe encoding
    '&#34;': '"',   # Alternative quote encoding
}

def clean_text_for_tts(text):
    """Aggressively clean text for TTS: decode HTML → normalize Unicode → remove invisibles"""
    original = text
    
    # STEP 1: Decode HTML entities FIRST
    for entity, char in HTML_ENTITIES.items():
        if entity in text:
            text = text.replace(entity, char)
    
    # STEP 2: Normalize Unicode whitespace
    if 'unicodedata' in globals() and unicodedata:
        text = ''.join(' ' if unicodedata.category(c).startswith('Z') else c for c in text)
    
    # STEP 3: Remove zero-width characters and other invisibles
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]', '', text)
    
    # STEP 4: Collapse multiple whitespace → single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # STEP 5: Remove newlines
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text

def safe_chunk_text(text, max_chars=200):
    """Split text into chunks of ≤max_chars, preferring sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        # If we're not at the end, try to split at sentence boundary
        if end < len(text):
            # Look backward for sentence-ending punctuation
            sentence_end = -1
            for i in range(end, start, -1):
                if text[i-1] in '.!?':
                    sentence_end = i
                    break
            
            if sentence_end != -1 and sentence_end > start + 30:  # Avoid tiny chunks
                end = sentence_end
            else:
                # Fallback: hard split, but avoid cutting mid-word
                while end > start and text[end-1] not in ' \t\n':
                    end -= 1
                if end == start:  # All one word? Just split
                    end = start + max_chars
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    
    return chunks

# --- DEPENDENCY INSTALLATION ---
def ensure_deps():
    required = ['onnxruntime', 'numpy', 'soundfile', 'sentencepiece', 'scipy', 'huggingface_hub']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        logger.info(f"Installing missing packages: {', '.join(missing)}...")
        for pkg in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
                logger.info(f"  ✓ {pkg}")
            except subprocess.CalledProcessError as e:
                logger.error(f"  ✗ Failed to install {pkg}: {e}")
                raise
        logger.info("Package installation complete.")
    else:
        logger.info("✓ All required packages already installed.")

# --- MODEL DOWNLOAD ---
def ensure_model():
    if not MODEL_PATH.exists():
        logger.info("Downloading Pocket TTS model (INT8 only, ~200 MB)...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                MODEL_ID,
                local_dir=str(MODEL_PATH),
                local_dir_use_symlinks=False,
                allow_patterns=[
                    "onnx/*_int8.onnx",
                    "onnx/mimi_encoder.onnx",
                    "onnx/text_conditioner.onnx",
                    "tokenizer.model",
                    "pocket_tts_onnx.py",
                    "generate.py",
                    "requirements.txt"
                ]
            )
            logger.info("✓ Model download complete.")
        except Exception as e:
            logger.error(f"✗ Model download failed: {e}")
            raise
    
    missing = []
    for f in EXPECTED_ONNX_FILES:
        if not (MODEL_PATH / f).exists():
            missing.append(f)
    
    if missing:
        logger.error(f"✗ Missing ONNX files: {missing}")
        raise FileNotFoundError(f"Model download incomplete. Missing: {missing}")
    
    logger.info("✓ Model files verified.")

# --- IMPORT AFTER INSTALLATION ---
def import_required_modules():
    global np, sf, signal, unicodedata, gr
    
    try:
        import numpy as np
        import soundfile as sf
        from scipy import signal
        import unicodedata
        import gradio as gr
        logger.info("✓ All required modules imported successfully.")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import required modules: {e}")
        return False

# --- ENGINE LOADING ---
def load_tts_engine():
    global MODEL_LOADED, tts_engine
    
    with model_lock:
        if MODEL_LOADED and tts_engine:
            return True
        
        logger.info(f"Loading Pocket TTS engine (IntraOp Threads: {INTRA_OP_THREADS})...")
        
        if str(MODEL_PATH) not in sys.path:
            sys.path.insert(0, str(MODEL_PATH))
        
        try:
            from pocket_tts_onnx import PocketTTSOnnx
            
            models_dir = str(MODEL_PATH / 'onnx')
            tokenizer_path = str(MODEL_PATH / 'tokenizer.model')
            
            tts_engine = PocketTTSOnnx(
                models_dir=models_dir,
                tokenizer_path=tokenizer_path,
                temperature=0.9,
                lsd_steps=3
            )
            
            MODEL_LOADED = True
            logger.info("✓ Pocket TTS engine loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to load TTS engine: {e}", exc_info=True)
            return False

# --- INITIALIZATION ---
def initialize_extension():
    logger.info("="*80)
    logger.info("🔧 INITIALIZING POCKET TTS EXTENSION")
    logger.info("="*80)
    
    try:
        ensure_deps()
        if not import_required_modules():
            raise ImportError("Could not import required modules")
        ensure_model()
        if load_tts_engine():
            logger.info("="*80)
            logger.info("✅ POCKET TTS EXTENSION READY")
            logger.info("="*80)
        else:
            logger.error("❌ POCKET TTS INITIALIZATION FAILED")
    except Exception as e:
        logger.exception(f"💥 Extension initialization crashed: {e}")

if not MODEL_LOADED:
    try:
        initialize_extension()
    except Exception as e:
        logger.error(f"Initialization failed: {e}")

# --- VOICE PROCESSING ---
def get_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_processed_voice(input_path):
    if not input_path or not Path(input_path).exists():
        return None
    
    if sf is None or signal is None or np is None:
        return None
    
    filename = os.path.basename(input_path)
    output_path = CACHE_VOICE_DIR / f"proc_{filename}"
    
    current_hash = get_file_hash(input_path)
    if output_path.exists() and params['_last_file_hash'] == current_hash:
        return str(output_path.resolve())
    
    try:
        data, sr = sf.read(input_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if sr != 24000:
            data = signal.resample(data, int(len(data) * 24000 / sr))
        
        mask = np.abs(data) > 0.01
        if mask.any():
            indices = np.where(mask)[0]
            if len(indices) >= 2:
                s, e = indices[0], indices[-1]
                data = data[max(0, s-1000):min(len(data), e+1000)]
            else:
                idx = indices[0]
                data = data[max(0, idx-1000):min(len(data), idx+1000)]
        
        data = data[:24000*15]
        sf.write(str(output_path), data, 24000)
        
        params['_last_file_hash'] = current_hash
        return str(output_path.resolve())
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        return None

# --- TEST BUTTON ---
def test_voice():
    if not MODEL_LOADED or tts_engine is None:
        return "❌ TTS engine not loaded."
    
    if not params['_cached_voice_path']:
        return "❌ Upload a voice first."
    
    if not Path(params['_cached_voice_path']).exists():
        return f"❌ Voice file missing."
    
    try:
        out = OUTPUT_DIR / f"test_{int(time.time()*1000)}.wav"
        
        with model_lock:
            audio = tts_engine.generate(text="Hello! This is a test.", voice=params['_cached_voice_path'])
            
            sr = 24000
            if isinstance(audio, tuple):
                audio, sr = audio
            
            tts_engine.save_audio(audio, str(out))
        
        try:
            web_root = Path(__file__).parent.parent.parent
            rel_path = out.relative_to(web_root)
            audio_src = f"file/{rel_path.as_posix()}"
        except ValueError:
            audio_src = f"file={out.as_posix()}"
        
        return f'✅ Success!<br><audio src="{audio_src}" controls autoplay></audio>'
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        return f"❌ Error: {str(e)}"

def clear_cache():
    count = 0
    if CACHE_VOICE_DIR.exists():
        for f in CACHE_VOICE_DIR.glob("*.wav"):
            try: f.unlink(); count += 1
            except: pass
    params['_cached_voice_path'] = None
    params['_last_uploaded_file'] = None
    params['_last_file_hash'] = None
    params['voice_file'] = None
    return f"✅ Cleared {count} files."

# --- MAIN PROCESSING ---
def output_modifier(string, state):
    if not params['activate'] or not MODEL_LOADED or tts_engine is None:
        return string
    
    logger.info("="*80)
    logger.info("🎤 TTS INPUT RECEIVED (RAW FROM OOBABOOGA)")
    logger.info(f"   RAW length: {len(string)} chars")
    logger.info(f"   First 100 chars (repr): {repr(string[:100])}")
    logger.info("="*80)
    
    cleaned_text = clean_text_for_tts(string)
    
    if not cleaned_text or len(cleaned_text) < MIN_TTS_LENGTH:
        logger.warning(f"🔇 TTS SKIPPED: Invalid text after cleaning ({len(cleaned_text)} chars)")
        return string
    
    if len(cleaned_text) > MAX_TTS_LENGTH:
        logger.warning(f"⚠️ TTS TRUNCATED: {len(cleaned_text)} → {MAX_TTS_LENGTH} chars")
        cleaned_text = cleaned_text[:MAX_TTS_LENGTH]
    
    voice_path = None
    if params['voice_file'] and Path(params['voice_file']).exists():
        if not params['_cached_voice_path'] or params['_last_uploaded_file'] != params['voice_file']:
            params['_last_uploaded_file'] = params['voice_file']
            params['_cached_voice_path'] = get_processed_voice(params['voice_file'])
        voice_path = params['_cached_voice_path']
    
    if not voice_path:
        logger.warning("⚠️ TTS SKIPPED: No valid voice file uploaded")
        return "[⚠️ Upload .wav voice]<br><br>" + string
    
    # ✅ SAFE CHUNKED GENERATION (≤200 chars per chunk)
    try:
        out = OUTPUT_DIR / f"{int(time.time()*1000)}.wav"
        logger.info("▶️ TTS GENERATION STARTED (Safe Chunking)")
        logger.info(f"   Voice: {Path(voice_path).name}")
        logger.info(f"   Cleaned text length: {len(cleaned_text)} chars")
        
        text_chunks = safe_chunk_text(cleaned_text, max_chars=MAX_CHARS_PER_CHUNK)
        logger.info(f"   Split into {len(text_chunks)} chunks (≤{MAX_CHARS_PER_CHUNK} chars each)")
        
        all_audio_chunks = []
        total_elapsed = 0
        
        for idx, chunk in enumerate(text_chunks):
            logger.info(f"   ▶️ Generating chunk {idx+1}/{len(text_chunks)} ({len(chunk)} chars)")
            chunk_start = time.time()
            with model_lock:
                audio_chunk = tts_engine.generate(text=chunk, voice=voice_path)
            chunk_elapsed = time.time() - chunk_start
            total_elapsed += chunk_elapsed
            
            sr = 24000
            if isinstance(audio_chunk, tuple):
                audio_chunk, sr = audio_chunk
            
            all_audio_chunks.append(np.array(audio_chunk))
            logger.info(f"   ✅ Chunk {idx+1} done | {chunk_elapsed:.2f}s")
        
        final_audio = np.concatenate(all_audio_chunks)
        elapsed = total_elapsed
        
        tts_engine.save_audio(final_audio, str(out))
        
        logger.info(f"✅ TTS COMPLETE | {elapsed:.2f}s | RTFx: {len(final_audio)/sr/elapsed:.2f}x | Output: {out.name}")
        logger.info("="*80)
        
        # Construct audio URL
        try:
            web_root = Path(__file__).parent.parent.parent
            rel_path = out.relative_to(web_root)
            audio_src = f"file/{rel_path.as_posix()}"
        except ValueError:
            audio_src = f"file={out.as_posix()}"
        
        # Autoplay ONLY on latest message
        autoplay_attr = ""
        if params['autoplay']:
            if state and 'internal' in state:
                internal_msgs = state['internal']
                if len(internal_msgs) >= 2:
                    last_ai_response = internal_msgs[-1][1] if internal_msgs[-1] else ""
                    if string.strip() == last_ai_response.strip():
                        autoplay_attr = "autoplay"
                        logger.info("🔊 Autoplay ENABLED (latest message)")
                    else:
                        logger.info("🔇 Autoplay DISABLED (older message)")
                else:
                    autoplay_attr = "autoplay"
            else:
                autoplay_attr = "autoplay"
        
        return f'<audio src="{audio_src}" controls {autoplay_attr}></audio><br><br>{string}'
        
    except Exception as e:
        logger.error(f"❌ TTS FAILED: {e}", exc_info=True)
        logger.info("="*80)
        return f"[⚠️ {str(e)}]<br><br>{string}"

# --- UI ---
def ui():
    try:
        import gradio as gr
    except ImportError:
        return
    
    with gr.Accordion(f"🔊 Pocket TTS ({'READY' if MODEL_LOADED else 'LOADING'})", open=False):
        status = "✅ Ready" if MODEL_LOADED else "⏳ Initializing..."
        gr.Markdown(f"**Status**: {status}")
        
        if MODEL_LOADED:
            gr.Markdown(
                f"**Optimizations**:<br>"
                f"• INT8-only models (~200 MB)<br>"
                f"• Safe chunking: ≤{MAX_CHARS_PER_CHUNK} chars/chunk (maximum stability)<br>"
                f"• IntraOp threads: {INTRA_OP_THREADS}<br>"
                f"• HTML decoding + invisible char stripping<br>"
                f"• Autoplay: latest message only"
            )
        
        with gr.Row():
            gr.Checkbox(value=params['activate'], label='Activate')
            gr.Checkbox(value=params['autoplay'], label='Autoplay')
        
        voice_upload = gr.File(label='Voice (.wav)', file_types=['.wav'], type='filepath')
        
        with gr.Row():
            test_btn = gr.Button("🔊 Test")
            clear_btn = gr.Button("🗑️ Clear")
        
        test_output = gr.Markdown("")
        
        def _on_upload(p):
            if p:
                params['voice_file'] = p
                params['_cached_voice_path'] = None
                params['_last_file_hash'] = None
                return f"**Voice**: {Path(p).name}"
            return "**Voice**: None"
        
        voice_upload.change(_on_upload, voice_upload, gr.Markdown())
        test_btn.click(test_voice, None, test_output)
        clear_btn.click(clear_cache, None, test_output)
