# Qwen3-TTS Voice Cloning Guide (Mac M4)

## How Voice Cloning Works

You need two things from your voice:

1. **A reference audio clip** - a short recording of you speaking (5-15 seconds)
2. **The reference text** - the exact words you said in that recording

Then you provide the **synthesis text** - what you want the cloned voice to say.

## Quick Start

### 1. Record yourself

Say a clear sentence and save it as a WAV file, e.g. `my_voice.wav`.

### 2. Edit the script

In `examples/test_model_12hz_base.py`, replace the remote URLs and text with your own:

```python
ref_audio_single = "my_voice.wav"
ref_text_single = "Whatever you actually said in the recording."
syn_text_single = "This is my cloned voice saying something new."
syn_lang_single = "English"  # or "Chinese", or "Auto"
```

### 3. Run it

```bash
conda activate qwen3-tts
python test_model_12hz_base.py
```

Output WAV files are saved to `qwen3_tts_test_voice_clone_output_wav/`.

## Minimal Script

If you just want a single-sentence clone without all the test cases:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="mps",
)

wavs, sr = tts.generate_voice_clone(
    text="Hello, this is my cloned voice!",
    language="English",
    ref_audio="my_voice.wav",
    ref_text="The exact words from your recording.",
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
)

sf.write("output.wav", wavs[0], sr)
```

The key requirement is that `ref_text` must accurately match what's spoken in `ref_audio` - the model uses the alignment between them to learn your voice characteristics.

## Cloning Modes

The script supports two modes via the `xvec_only` flag:

- **`xvec_only=False` (ICL mode)** - Uses your full audio as in-context learning. Better quality, captures more nuance of your voice, but slower.
- **`xvec_only=True` (x-vector mode)** - Extracts a compact voice embedding from your audio. Faster, but less precise cloning.

## M4-Specific Code Changes

The default scripts are built for NVIDIA GPUs. The following changes have already been applied to `examples/test_model_12hz_base.py`:

- `attn_implementation="sdpa"` instead of `"flash_attention_2"`
- `device_map="mps"` instead of `"cuda:0"`
- `torch.mps.synchronize()` instead of `torch.cuda.synchronize()`

---

# Managing Hugging Face Model Cache

Models downloaded via `from_pretrained` are stored in `~/.cache/huggingface/hub/`.

## List cached models

```bash
# Simple listing with sizes
du -sh ~/.cache/huggingface/hub/models--*/

# Using the HF CLI (more detailed)
huggingface-cli scan-cache
```

## Delete a specific model

```bash
# Manual deletion
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base

# Interactive picker via HF CLI
huggingface-cli delete-cache
```

## Delete all cached models

```bash
rm -rf ~/.cache/huggingface/hub/models--*
```

## Notes

- The model used by this project is `Qwen--Qwen3-TTS-12Hz-1.7B-Base` (~4.2GB)
- Models are shared across all projects - deleting a model here affects any project using it
- Re-running a script will re-download any missing models automatically
