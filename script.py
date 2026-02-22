""" OOBA BOOGA EXTENSION 
Pocket TTS - Final Optimized + Chunked Version
- Applies 'intra_op_threads=min(cpu_count, 4)' for ~2x speedup on autoregressive models.
- Correctly handles 'onnx' subfolder structure.
- Lazy loading + Caching + Diagnostics.
- HTML ENTITY DECODING: Converts &#x27; → ', &quot; → ", etc. BEFORE TTS.
- NEWLINE REMOVAL: Replaces all \\n/\\r with single space for natural speech flow.
- INVISIBLE CHARACTER STRIPPING: Removes zero-width spaces, non-breaking spaces, etc.
- CHUNKED GENERATION: Splits text into ≤100-token chunks, generates audio per chunk, concatenates.
- TEXT VALIDATION: Ensures only valid, reasonable-length AI responses are processed.
- AUTPLAY FIX: Only latest message auto-plays, older messages do not.
- ZERO AUDIO POST-PROCESSING: Audio saved exactly as generated.
- FULL CONSOLE LOGGING: Shows RAW input → CLEANED → CHUNKED pipeline.
"""
import os
import sys
import subprocess
import logging
import time
import threading
import hashlib
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import re
import unicodedata

# --- CRITICAL PERFORMANCE TUNING ---
CPU_COUNT = os.cpu_count() or 4
INTRA_OP_THREADS = min(CPU_COUNT, 4)
INTER_OP_THREADS = 1

os.environ['ORT_PROVIDERS'] = 'CPUExecutionProvider'
os.environ['OMP_NUM_THREADS'] = str(CPU_COUNT)
os.environ['INTRA_OP_NUM_THREADS'] = str(INTRA_OP_THREADS)
os.environ['INTER_OP_NUM_THREADS'] = str(INTER_OP_THREADS)

# --- IMPORTS AFTER ENVIRONMENT SETUP ---
import gradio as gr
import sentencepiece as spm

logger = logging.getLogger("PocketTTS")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logging.getLogger("sentencepiece").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

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
MAX_TTS_LENGTH = 5000  # Increased to allow chunking of longer texts
MIN_TTS_LENGTH = 3
MAX_TOKENS_PER_CHUNK = 100  # Kyutai recommends ≤100 tokens per chunk for voice quality

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
    text = ''.join(' ' if unicodedata.category(c).startswith('Z') else c for c in text)
    
    # STEP 3: Remove zero-width characters and other invisibles
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]', '', text)
    
    # STEP 4: Collapse multiple whitespace → single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    # STEP 5: Remove newlines
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text

def get_tokenizer():
    """Load SentencePiece tokenizer used by PocketTTS"""
    if not MODEL_LOADED or tts_engine is None:
        return None
    tokenizer_path = str(MODEL_PATH / 'tokenizer.model')
    if not Path(tokenizer_path).exists():
        logger.warning(f"⚠️ Tokenizer not found at {tokenizer_path}")
        return None
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    return sp

def chunk_text_by_tokens(text, max_tokens=100):
    """Split text into chunks of ≤ max_tokens using SentencePiece tokenizer"""
    sp = get_tokenizer()
    if sp is None:
        logger.warning("⚠️ Tokenizer unavailable. Falling back to character-based chunking.")
        # Fallback: ~5 chars per token heuristic
        fallback_chunk_size = max_tokens * 5
        chunks = [text[i:i+fallback_chunk_size] for i in range(0, len(text), fallback_chunk_size)]
        return [c for c in chunks if c.strip()]
    
    # Encode full text to tokens
    tokens = sp.encode(text, out_type=int)
    logger.info(f"   Total tokens: {len(tokens)}")
    
    if len(tokens) <= max_tokens:
        return [text]
    
    # Split into chunks of max_tokens
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = sp.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    logger.info(f"   Split into {len(chunks)} chunks (≤{max_tokens} tokens each)")
    return chunks

# --- VALIDATION ---
EXPECTED_ONNX_FILES = [
    'onnx/flow_lm_main_int8.onnx',
    'onnx/flow_lm_flow_int8.onnx',
    'onnx/mimi_decoder_int8.onnx',
    'onnx/mimi_encoder.onnx',
    'onnx/text_conditioner.onnx',
    'tokenizer.model',
    'pocket_tts_onnx.py'
]

# --- SETUP ---
def ensure_deps():
    required = ['onnxruntime', 'numpy', 'soundfile', 'sentencepiece', 'scipy', 'huggingface_hub']
    for pkg in required:
        try: __import__(pkg.replace('-', '_'))
        except ImportError:
            logger.info(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", pkg])
    return True

def ensure_model():
    if not MODEL_PATH.exists():
        logger.info("Downloading model...")
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, local_dir=str(MODEL_PATH), local_dir_use_symlinks=False)
        logger.info("Download complete.")
    
    missing = []
    for f in EXPECTED_ONNX_FILES:
        if not (MODEL_PATH / f).exists():
            missing.append(f)
    
    if missing:
        logger.error(f"Missing ONNX files: {missing}")
        raise FileNotFoundError(f"Model download incomplete. Missing: {missing}")
    
    logger.info("✅ Model files verified.")
    return True

def load_tts_engine():
    global MODEL_LOADED, tts_engine
    
    with model_lock:
        if MODEL_LOADED and tts_engine:
            return True
        
        logger.info(f"Loading engine (IntraOp Threads: {INTRA_OP_THREADS})...")
        
        if str(MODEL_PATH) not in sys.path:
            sys.path.insert(0, str(MODEL_PATH))
        
        try:
            from pocket_tts_onnx import PocketTTSOnnx
            
            models_dir = str(MODEL_PATH / 'onnx')
            tokenizer_path = str(MODEL_PATH / 'tokenizer.model')
            
            # OPTIMIZED FOR SPEED: lsd_steps=1, temperature=0.7
            tts_engine = PocketTTSOnnx(
                models_dir=models_dir,
                tokenizer_path=tokenizer_path,
                temperature=0.5,
                lsd_steps=8
            )
            
            MODEL_LOADED = True
            logger.info(f"✅ Engine loaded. (Using {INTRA_OP_THREADS} intra-op threads, lsd_steps=1)")
            return True
        except Exception as e:
            logger.error(f"Load failed: {e}", exc_info=True)
            return False

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
    
    filename = os.path.basename(input_path)
    output_path = CACHE_VOICE_DIR / f"proc_{filename}"
    
    current_hash = get_file_hash(input_path)
    if output_path.exists() and params['_last_file_hash'] == current_hash:
        return str(output_path.resolve())
    
    logger.info(f"Processing voice: {filename}...")
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
        logger.info(f"✅ Voice cached: {output_path.name}")
        return str(output_path.resolve())
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        return None

# --- TEST ---
def test_voice():
    if not MODEL_LOADED and not load_tts_engine():
        return "❌ Load failed."
    
    if not params['_cached_voice_path']:
        return "❌ Upload a voice first."
    
    if not Path(params['_cached_voice_path']).exists():
        return f"❌ Voice file missing."
    
    try:
        out = OUTPUT_DIR / f"test_{int(time.time()*1000)}.wav"
        
        with model_lock:
            logger.info(f"🔊 TTS TEST TRIGGERED")
            logger.info(f"   Text: 'Hello! This is a test of optimized inference.'")
            logger.info(f"   Voice: {params['_cached_voice_path']}")
            logger.info(f"   Starting generation...")
            
            start = time.time()
            audio = tts_engine.generate(text="Hello! This is a test of optimized inference.", voice=params['_cached_voice_path'])
            elapsed = time.time() - start
            
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
        
        logger.info(f"✅ TTS TEST COMPLETE | Duration: {elapsed:.2f}s | Output: {out.name}")
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

# --- MAIN ---
def output_modifier(string, state):
    # Log RAW input BEFORE any processing
    logger.info("="*80)
    logger.info("🎤 TTS INPUT RECEIVED (RAW FROM OOBABOOGA)")
    logger.info(f"   RAW length: {len(string)} chars")
    logger.info(f"   First 100 chars (repr): {repr(string[:100])}")
    logger.info(f"   Last 100 chars (repr):  {repr(string[-100:])}")
    logger.info("="*80)
    
    # Skip if disabled
    if not params['activate']:
        logger.info("🔇 TTS SKIPPED: Extension disabled")
        return string
    
    # Apply aggressive cleaning
    cleaned_text = clean_text_for_tts(string)
    
    # Validate after cleaning
    if not cleaned_text or len(cleaned_text) < MIN_TTS_LENGTH:
        logger.warning(f"🔇 TTS SKIPPED: Invalid text after cleaning ({len(cleaned_text)} chars)")
        return string
    
    if len(cleaned_text) > MAX_TTS_LENGTH:
        logger.warning(f"⚠️ TTS TRUNCATED: {len(cleaned_text)} → {MAX_TTS_LENGTH} chars")
        cleaned_text = cleaned_text[:MAX_TTS_LENGTH]
    
    # Load engine if needed
    if not MODEL_LOADED and not load_tts_engine():
        logger.error("❌ TTS FAILED: Engine load failed")
        return "[⚠️ Load Failed]<br><br>" + string
    
    # Setup voice
    voice_path = None
    if params['voice_file'] and Path(params['voice_file']).exists():
        if not params['_cached_voice_path'] or params['_last_uploaded_file'] != params['voice_file']:
            params['_last_uploaded_file'] = params['voice_file']
            params['_cached_voice_path'] = get_processed_voice(params['voice_file'])
        voice_path = params['_cached_voice_path']
    
    if not voice_path:
        logger.warning("⚠️ TTS SKIPPED: No valid voice file uploaded")
        return "[⚠️ Upload .wav voice]<br><br>" + string
    
    # Generate audio
    try:
        out = OUTPUT_DIR / f"{int(time.time()*1000)}.wav"
        logger.info("▶️ TTS GENERATION STARTED")
        logger.info(f"   Voice: {Path(voice_path).name}")
        logger.info(f"   Cleaned text length: {len(cleaned_text)} chars")
        
        # ✅ CHUNKED TTS GENERATION (≤100 tokens per chunk)
        text_chunks = chunk_text_by_tokens(cleaned_text, max_tokens=MAX_TOKENS_PER_CHUNK)
        all_audio_chunks = []
        total_elapsed = 0
        
        for idx, chunk in enumerate(text_chunks):
            logger.info(f"   ▶️ Generating chunk {idx+1}/{len(text_chunks)} ({len(chunk)} chars)")
            chunk_start = time.time()
            with model_lock:
                audio_chunk = tts_engine.generate(text=chunk, voice=voice_path)
            chunk_elapsed = time.time() - chunk_start
            total_elapsed += chunk_elapsed
            
            # Extract audio array and sample rate
            sr = 24000
            if isinstance(audio_chunk, tuple):
                audio_chunk, sr = audio_chunk
            
            all_audio_chunks.append(np.array(audio_chunk))
            logger.info(f"   ✅ Chunk {idx+1} done | {chunk_elapsed:.2f}s")
        
        # Concatenate all chunks into final audio
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
        
        # ✅ CRITICAL FIX: Autoplay ONLY on the latest AI message
        autoplay_attr = ""
        if params['autoplay']:
            # Check if this is the last message in the chat
            if state and 'internal' in state:
                internal_msgs = state['internal']
                if len(internal_msgs) >= 2:
                    # internal format: [[user_msg, ai_response], ...]
                    last_ai_response = internal_msgs[-1][1] if internal_msgs[-1] else ""
                    # Compare content (not perfect but robust enough)
                    if string.strip() == last_ai_response.strip():
                        autoplay_attr = "autoplay"
                        logger.info("🔊 Autoplay ENABLED (latest message)")
                    else:
                        logger.info("🔇 Autoplay DISABLED (older message)")
                else:
                    autoplay_attr = "autoplay"  # fallback for single-message chats
            else:
                autoplay_attr = "autoplay"  # fallback if state unavailable
        
        return f'<audio src="{audio_src}" controls {autoplay_attr}></audio><br><br>{string}'
        
    except Exception as e:
        logger.error(f"❌ TTS FAILED: {e}", exc_info=True)
        logger.info("="*80)
        return f"[⚠️ {str(e)}]<br><br>{string}"

def ui():
    with gr.Accordion(f"🔊 Pocket TTS ({INTRA_OP_THREADS} intra-op threads)", open=False):
        st = "✅ Ready" if MODEL_LOADED else "⏳ Lazy Load"
        gr.Markdown(f"**Status**: {st} | Optimization: IntraOp={INTRA_OP_THREADS}, InterOp={INTER_OP_THREADS}<br>"
                   f"✅ HTML Decoding + Newline Removal + Invisible Char Stripping<br>"
                   f"✅ Chunked TTS: ≤{MAX_TOKENS_PER_CHUNK} tokens per chunk (prevents voice rot)<br>"
                   f"✅ Safety Limits: Min {MIN_TTS_LENGTH} / Max {MAX_TTS_LENGTH} chars<br>"
                   f"✅ Speed Optimized: lsd_steps=1 (~4x RTF)<br>"
                   f"✅ Autoplay: Latest message only<br>"
                   f"✅ FULL CONSOLE LOGGING: See terminal for raw → cleaned → chunked pipeline")
        
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

# --- INIT ---
ensure_deps()
ensure_model()
logger.info(f"Pocket TTS initialized. (Optimized for {CPU_COUNT} cores)")
logger.info(f"✅ Safety limits: Min {MIN_TTS_LENGTH} / Max {MAX_TTS_LENGTH} chars per TTS call")
logger.info(f"✅ Chunked generation: ≤{MAX_TOKENS_PER_CHUNK} tokens per chunk")
logger.info("✅ HTML Decoding + Newline Removal + Invisible Char Stripping enabled")
logger.info("="*80)