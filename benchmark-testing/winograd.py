import platform
import ctypes.util
import warnings
import os
import csv
import traceback
import time
import json

import torch
import torch.nn as nn
import whisper
from jiwer import wer
from ptflops import get_model_complexity_info

# ─── Monkey-patch ctypes.util.find_library on Windows ─────────────────────────
if platform.system() == "Windows":
    original_find_library = ctypes.util.find_library
    def custom_find_library(name):
        if name == "c":
            return "msvcrt.dll"
        return original_find_library(name)
    ctypes.util.find_library = custom_find_library

# ─── Suppress safe-unpickle warnings ─────────────────────────────────────────
warnings.filterwarnings("ignore", message=".*weights_only.*")

# ─── CSV setup ──────────────────────────────────────────────────────────────
csv_output_path = "winograd_benchmark_results.csv"
csv_headers = ["model", "winograd_mode", "file", "time_sec", "wer", "vram_mb", "bytes_per_instr"]
with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(csv_headers)

# ─── General config ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path = "audio_files"
output_path = "transcripts_winograd"
os.makedirs(output_path, exist_ok=True)

file_list = [f for f in os.listdir(audio_path) if f.lower().endswith((".wav", ".mp3"))]
if not file_list:
    print(f"No audio files found in {audio_path}; please put your .wav/.mp3 there.")
    exit(1)

# Standard Whisper models - removed 'turbo' as it's not an official model
model_list = ['tiny', 'base', 'small', 'medium', 'large-v2']
winograd_opts = [True, False]
model_flops = {}

# ─── FLOPs-wrapper for Whisper ──────────────────────────────────────────────
class WhisperFLOPsWrapper(nn.Module):
    def __init__(self, whisper_model):
        super().__init__()
        self.encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder
        self.n_text_ctx = whisper_model.dims.n_text_ctx

    def forward(self, mel: torch.Tensor):
        enc_out = self.encoder(mel)
        batch = mel.shape[0]
        tokens = torch.zeros((batch, self.n_text_ctx), dtype=torch.long, device=mel.device)
        return self.decoder(tokens, enc_out)

# ─── Precompute FLOPs for each Whisper model ────────────────────────────────
print("\nPrecomputing FLOPs for models...")
for mname in model_list:
    try:
        base = whisper.load_model(mname, device="cpu")
        n_mels = base.dims.n_mels  # Get model-specific mel bands
        wrapper = WhisperFLOPsWrapper(base)
        macs, _ = get_model_complexity_info(
            wrapper,
            (n_mels, 3000),  # Use dynamic n_mels
            as_strings=False,
            verbose=False,
            print_per_layer_stat=False
        )
        flops = 2 * macs if macs is not None else 1e9  # Default to 1 GFLOP if MACs is None
        model_flops[mname] = flops
        print(f" • {mname}: {flops/1e9:.2f} GFLOPs")
        del base, wrapper
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error precomputing FLOPs for {mname}: {str(e)}")
        model_flops[mname] = 1e9  # Default fallback value

print(f"\nRunning on device: {device}")
print(f"{len(file_list)} audio files to process.\n")

# ─── Benchmark loop ─────────────────────────────────────────────────────────
for mname in model_list:
    for use_winograd in winograd_opts:
        print(f"\n--- Benchmarking '{mname}' | cuDNN.benchmark = {use_winograd} ---")

        # FULL isolation: destroy CUDA context and flush everything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = use_winograd

        model = None

        try:
            # Fresh reload of model with error handling
            try:
                model = whisper.load_model(mname, device=device, download_root="models")
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Skipping {mname} due to insufficient GPU memory.")
                    continue
                else:
                    raise

            # Calculate model size and bytes per instruction
            model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            flops = model_flops.get(mname, 1e9)
            bpi = model_bytes / flops if flops else 0

            print(f"Model size: {model_bytes/1e6:.1f} MB  |  Bytes/Instr: {bpi:.4f}")

            # Warmup run to stabilize kernel selection
            warmup_file = os.path.join(audio_path, file_list[0])
            try:
                _ = model.transcribe(warmup_file, language="en", task="transcribe")
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Skipping {mname} due to OOM during warmup")
                    continue
                else:
                    raise

            for fname in file_list:
                try:
                    inp = os.path.join(audio_path, fname)
                    start = time.perf_counter()
                    res = model.transcribe(inp, language="en", task="transcribe")
                    dur = time.perf_counter() - start

                    # Save outputs
                    base = os.path.splitext(fname)[0]
                    odir = os.path.join(output_path, base)
                    os.makedirs(odir, exist_ok=True)

                    with open(os.path.join(odir, f"{base}_{mname}_winograd_{use_winograd}.txt"), "w", encoding="utf-8") as t:
                        t.write(res["text"])

                    with open(os.path.join(odir, f"{base}_{mname}_winograd_{use_winograd}.json"), "w", encoding="utf-8") as j:
                        json.dump(res, j, indent=2)

                    # WER evaluation if real transcript exists
                    wer_score = ""
                    real_path = os.path.join(odir, f"{base}_REAL.txt")
                    if os.path.exists(real_path):
                        ref = open(real_path, encoding="utf-8").read().strip()
                        wer_score = f"{wer(ref, res['text'].strip()):.4f}"

                    # VRAM peak usage
                    peak = ""
                    if torch.cuda.is_available():
                        peak = f"{torch.cuda.max_memory_allocated() / (1024**2):.2f}"

                    # Record to CSV
                    with open(csv_output_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            mname,
                            use_winograd,
                            fname,
                            f"{dur:.3f}",
                            wer_score,
                            peak,
                            f"{bpi:.6f}"
                        ])

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f"OOM during processing {fname}, skipping...")
                        continue
                    else:
                        print(f"Error processing {fname}: {str(e)}")
                        continue

        except Exception as e:
            print(f"Error during benchmarking {mname} with Winograd={use_winograd}: {str(e)}")
            traceback.print_exc()

        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()
            if torch.backends.cudnn.enabled:
                torch.backends.cudnn.benchmark = False

print("\nBenchmarking complete!")