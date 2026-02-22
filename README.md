#Warning this is made with AI , Vibecoded , not coded by me , however it works as far as I seen.
# Pocket TTS ONNX Extension for oobabooga/text-generation-webui

CPU-only zero-shot voice cloning TTS using Pocket TTS (ONNX INT8 quantized) inside text-generation-webui.

**Core strengths**  
- ~200 MB footprint, zero VRAM required  
- True zero-shot cloning from 3–10 s reference audio  
- Qwen3-TTS-1.7B phonetic bootstrap: generates dense-coverage synthetic reference → dramatically better cloning fidelity vs raw input  
- Automatic chunking: splits long text at ~100 tokens, generates separate WAVs, concatenates with crossfade + micro-silence → natural breathing-like pauses instead of decoder crashes  

Slower than Piper (~4× real-time factor on mid-range CPU) but cloned voices sound noticeably more natural and speaker-accurate.

This is vibecode / proof-of-concept. Tested on [insert your hardware, e.g. Ryzen 7 laptop, Windows 11]. Bugs, especially ONNX cross-platform issues and long-input seams, are expected.

## Why bother

No public Pocket TTS ONNX extension exists for oobabooga (GitHub/Reddit searches return zero as of Feb 2026). Existing TTS backends:  
- Piper → fast, fixed voices, no cloning  
- XTTS/Coqui → cloning but VRAM hungry  
- Silero/AllTalk/Bark → limited or slow cloning  

This targets lightweight, local, cloning-capable TTS on laptops / low-end hardware.

## Installation
- Just download the zip and install it into the extensions folder in ooba booga , then activate the extension , load a voice clip and you're done.
