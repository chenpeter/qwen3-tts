#!/usr/bin/env bash
# Launch Qwen3-TTS Voice Clone UI
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Setting up virtual environment..."
  python3 -m venv .venv
  source .venv/bin/activate
  pip install torch torchaudio
  pip install -e .
  pip install flask
else
  source .venv/bin/activate
fi

python voice_clone_app.py --device mps --port 8000 "$@"
