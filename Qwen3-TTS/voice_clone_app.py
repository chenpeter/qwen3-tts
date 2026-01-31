# coding=utf-8
"""
Qwen3-TTS Voice Clone — lightweight Flask UI.

Usage:
    python voice_clone_app.py                          # defaults
    python voice_clone_app.py --checkpoint Qwen/Qwen3-TTS-12Hz-0.6B-Base
    python voice_clone_app.py --device mps             # Apple Silicon
    python voice_clone_app.py --device cuda:0           # NVIDIA GPU
"""

import argparse
import io
import os
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
from flask import Flask, jsonify, request, send_file

from qwen_tts import Qwen3TTSModel

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
tts: Qwen3TTSModel = None  # set in main()

SUPPORTED_LANGUAGES: list[str] = []

UPLOAD_DIR = tempfile.mkdtemp(prefix="qwen3_tts_uploads_")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Qwen3-TTS Voice Clone web UI")
    p.add_argument("--checkpoint", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    help="Model checkpoint path or HuggingFace repo id.")
    p.add_argument("--device", default="mps",
                    help="Device: cpu, mps, cuda, cuda:0, etc.")
    p.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Torch dtype (default: bfloat16).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5000)
    return p.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32}[s]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    with open(os.path.join(HERE, "voice_clone_ui.html")) as f:
        return f.read()


@app.route("/api/languages")
def api_languages():
    return jsonify(SUPPORTED_LANGUAGES)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    # --- collect form data ---
    ref_audio_file = request.files.get("ref_audio")
    ref_text = (request.form.get("ref_text") or "").strip()
    target_text = (request.form.get("target_text") or "").strip()
    language = (request.form.get("language") or "Auto").strip()
    x_vector_only = request.form.get("x_vector_only") == "true"

    if not ref_audio_file:
        return jsonify({"error": "Reference audio is required."}), 400
    if not target_text:
        return jsonify({"error": "Output transcript is required."}), 400
    if not x_vector_only and not ref_text:
        return jsonify({"error": "Reference transcript is required when not using x-vector only mode."}), 400

    # Save uploaded audio to a temp file so the model can read it
    ext = os.path.splitext(ref_audio_file.filename or "audio.wav")[1] or ".wav"
    tmp_path = os.path.join(UPLOAD_DIR, f"ref_{int(time.time()*1000)}{ext}")
    ref_audio_file.save(tmp_path)

    try:
        wavs, sr = tts.generate_voice_clone(
            text=target_text,
            language=language,
            ref_audio=tmp_path,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=x_vector_only,
            max_new_tokens=4096,
            do_sample=True,
            top_k=50,
            top_p=1.0,
            temperature=0.9,
            repetition_penalty=1.05,
            subtalker_dosample=True,
            subtalker_top_k=50,
            subtalker_top_p=1.0,
            subtalker_temperature=0.9,
        )

        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)

        return send_file(buf, mimetype="audio/wav",
                         download_name="voice_clone_output.wav")
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    global tts, SUPPORTED_LANGUAGES

    args = _parse_args()

    print(f"Loading model: {args.checkpoint}  device={args.device}  dtype={args.dtype}")
    tts = Qwen3TTSModel.from_pretrained(
        args.checkpoint,
        device_map=args.device,
        dtype=_dtype_from_str(args.dtype),
        attn_implementation="sdpa",
    )

    if callable(getattr(tts.model, "get_supported_languages", None)):
        SUPPORTED_LANGUAGES = sorted(tts.model.get_supported_languages())
    else:
        SUPPORTED_LANGUAGES = [
            "Chinese", "English", "French", "German", "Italian",
            "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
        ]

    print(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
