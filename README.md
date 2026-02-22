# Warning
This is made with AI, Vibecoded, not coded by me, however it works as far as I seen.

# Pocket TTS ONNX Extension for oobabooga/text-generation-webui
CPU-only zero-shot voice cloning TTS using Pocket TTS (ONNX INT8 quantized) inside text-generation-webui.

**Core strengths**  
- ~200 MB footprint, zero VRAM required  
- True zero-shot cloning from 3–10 s reference audio  
- Qwen3-TTS-1.7B phonetic bootstrap: generates dense-coverage synthetic reference → dramatically better cloning fidelity vs raw input  
- Automatic chunking: splits long text at ~100 tokens, generates separate WAVs, concatenates with crossfade + micro-silence → natural breathing-like pauses instead of decoder crashes  

Slower than Piper (~4× real-time factor on mid-range CPU) but cloned voices sound noticeably more natural and speaker-accurate.

This is vibecode / proof-of-concept. Tested on Ryzen 7 2700. Bugs, especially ONNX cross-platform issues and long-input seams, are expected, however as far as I tested it had none lately.

## Why bother
No public Pocket TTS ONNX extension exists for oobabooga (GitHub/Reddit searches return zero as of Feb 2026). Existing TTS backends:  
- Piper → fast, fixed voices, no cloning  
- XTTS/Coqui → cloning but VRAM hungry  
- Silero/AllTalk/Bark → limited or slow cloning  

This targets lightweight, local, cloning-capable TTS on laptops / low-end hardware.

## Installation
- Download the zip from GitHub → extract to extensions/pocket-tts-onnx/  
- Restart webui or reload extensions  
- Upload a voice clip (there is no official documentation about length but pocketTTS samples are 9s).

## Tips
- For better results use the base clip into a bigger cloning model like Qwen 3 TTS 1.7B for example, from that make a phonetically dense clip and use this clip for pocketTTS.

# DISCLAIMER - READ THIS BEFORE USING

**I (the repository owner / uploader) take zero responsibility for any use, misuse, or consequences of this code.**

This extension includes voice cloning functionality via Pocket TTS (zero-shot from audio reference). Voice cloning can be used to:
- Replicate real people's voices without consent
- Create deepfakes, impersonations, misinformation, fraud, harassment, or illegal content

**I do not endorse, encourage, or condone any such use.**  
If you use this code to clone voices in ways that violate laws (e.g. identity theft, defamation, non-consensual pornography, election interference, fraud, stalking), that is entirely on you.  
You are solely responsible for:
- Obtaining explicit consent for any real person's voice used as reference
- Complying with all applicable laws in your jurisdiction (deepfake regulations, privacy laws, copyright on audio, etc.)
- Any harm, damage, legal consequences, or civil/criminal liability resulting from your use

**No warranty of any kind is provided.**  
This code is vibecode / AI-generated proof-of-concept. It is provided "as is" with no guarantees it works, is safe, secure, bug-free, or compatible with anything.  
Tested only on my hardware (Ryzen 7 2700). Your mileage may vary; crashes, audio artifacts, memory leaks, or silent failures are likely.

**No support will be provided.**  
I will not:
- Answer questions
- Fix bugs
- Add features
- Respond to issues/PRs
- Help with installation, errors, model paths, dependencies, or ethical/legal concerns

If something breaks, figure it out yourself or don't use it.  
Fork it, rewrite it, ignore it—do whatever. I am not maintaining this repo.

By downloading, installing, or using this extension, you acknowledge and agree that:
- You assume all risk
- You release me from any and all liability
- You will not hold me responsible for anything that happens as a result of this code

If you cannot or do not agree to these terms, delete the repo/folder immediately and do not use it.

MIT License applies to the code itself, but this disclaimer overrides any conflicting implication.
