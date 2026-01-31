# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn):
    _sync()
    t0 = time.time()

    wavs, sr = call_fn()

    _sync()
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def main():
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_wav"
    ensure_dir(OUT_DIR)

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="mps",
    )

    # Reference audio - Peter's voice
    ref_audio_single = "../source/peter-1.m4a"

    ref_text_single = (
        "Ten more steps. If he could take ten more steps it would be over, but his legs wouldn't move. "
        "He tried to will them to work, but they wouldn't listen to his brain. "
        "Ten more steps and it would be over but it didn't appear he would be able to do it. "
        "His mother had always taught him not to ever think of himself as better than others. "
        "He'd tried to live by this motto. He never looked down on those who were less fortunate or who had less money than him. "
        "But the stupidity of the group of people he was talking to made him change his mind. "
        "He picked up the burnt end of the branch and made a mark on the stone. "
        "Day 52 if the marks on the stone were accurate. He couldn't be sure. "
        "Day and nights had begun to blend together creating confusion, but he knew it was a long time. Much too long."
    )

    # Synthesis targets
    syn_text_single = (
        "あるプログラマーが面接に行きました。面接官が聞きました。あなたの特技は何ですか？"
        "彼は答えました。バグを見つけるのが得意です。"
        "面接官が言いました。例を挙げてもらえますか？"
        "彼は言いました。例えば、御社のWiFiパスワードがホワイトボードに書いてあるのを見つけました。"
        "面接官は三秒間沈黙した後、言いました。明日から来てください。"
    )
    syn_lang_single = "Japanese"

    common_gen_kwargs = dict(
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

    run_case(
        tts, OUT_DIR, "voice_clone",
        lambda: tts.generate_voice_clone(
            text=syn_text_single,
            language=syn_lang_single,
            ref_audio=ref_audio_single,
            ref_text=ref_text_single,
            x_vector_only_mode=False,
            **common_gen_kwargs,
        ),
    )



if __name__ == "__main__":
    main()
