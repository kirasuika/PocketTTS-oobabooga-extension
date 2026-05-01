
# Pocket TTS ONNX Extension for oobabooga/text-generation-webui with lavaSR upscaling

48khz Output & CPU-only zero-shot voice cloning TTS using Pocket TTS (ONNX INT8 quantized) inside text-generation-webui.

**Core strengths**  
- AutoInstall drop run
- ~200 MB footprint download, zero VRAM required  
- True zero-shot cloning from 3–10 s reference audio
- 48Khz Outputs on par with way bigger models → thanks to ysharma3501 https://github.com/ysharma3501/LavaSR
- Recommended :
Qwen3-TTS-1.7B phonetic bootstrap:
Use a bigger model like Qwen3 TTS to input the voice you want to clone or your own from there make a phonetically dense clip and use that clip for PocketTTS.

- Automatic chunking: splits long text at ~100 tokens, generates separate WAVs, concatenates with crossfade + micro-silence → natural breathing-like pauses instead of decoder crashes  

Slower than Piper (~4× real-time factor on mid-range CPU) but cloned voices sound noticeably more natural and speaker-accurate.


## Why bother
No public Pocket TTS ONNX extension exists for oobabooga (GitHub/Reddit searches return zero as of Feb 2026). Existing TTS backends:  
- Piper → fast, fixed voices, no cloning
- XTTS/Coqui → cloning but VRAM hungry  
- Silero/AllTalk/Bark → limited or slow cloning  

This targets lightweight, local, cloning-capable TTS on laptops / low-end hardware.

## Installation
- Download from releases script.py then into extensions make a folder with any name you'd like ideally PocketTTS , move script.py to that folder , start ooba booga activate the extension upload a clip and you're good to go.

- About the voice clip (there is no official documentation about length but pocketTTS samples are 9s).


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
By downloading, installing, or using this extension, you acknowledge and agree that:
- You assume all risk
- You release me from any and all liability
- You will not hold me responsible for anything that happens as a result of this code

If you cannot or do not agree to these terms, delete the repo/folder immediately and do not use it.

MIT License applies to the code itself, but this disclaimer overrides any conflicting implication.

## Disclaimer - AI Aided
