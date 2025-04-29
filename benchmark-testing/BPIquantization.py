import platform
import ctypes.util
import warnings
import os
import time
import json
import csv
import traceback

import torch
import torch.nn as nn
import whisper
from jiwer import wer
from ptflops import get_model_complexity_info

# Monkey-patch ctypes.util.find_library on Windows
if platform.system() == "Windows":
    original_find_library = ctypes.util.find_library
    def custom_find_library(name):
        if name == "c":
            return "msvcrt.dll"
        return original_find_library(name)
    ctypes.util.find_library = custom_find_library

# Suppress warnings
warnings.filterwarnings("ignore", message=".*weights_only.*")

# CSV Setup
csv_output_path = "quant_benchmark_results.csv"
csv_headers = [
    "model", "quant_mode", "file", "time_sec", "wer",
    "vram_mb", "param_mem_mb", "activation_mem_mb", "bytes_per_instr"
]
with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(csv_headers)

# General Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_path = "audio_files"
output_path = "transcripts_quant"
os.makedirs(output_path, exist_ok=True)

file_list = [f for f in os.listdir(audio_path) if f.lower().endswith((".wav", ".mp3"))]
if not file_list:
    print(f"No audio files found in {audio_path}; please add your .wav/.mp3 files.")
    exit(1)

# Standard Whisper models (removed 'turbo' which was causing issues)
model_list = ['tiny', 'base', 'small', 'medium', 'large-v2']
quantization_modes = ['full', 'fp16', 'dynamic']
model_flops = {}

# FLOPs Wrapper
class WhisperFLOPsWrapper(nn.Module):
    def __init__(self, whisper_model):
        super().__init__()
        self.encoder = whisper_model.encoder
        self.decoder = whisper_model.decoder
        self.dims = whisper_model.dims  # Store dims for text context

    def forward(self, mel: torch.Tensor):
        enc_out = self.encoder(mel)
        batch = mel.shape[0]
        tokens = torch.zeros((batch, self.dims.n_text_ctx), dtype=torch.long, device=mel.device)
        return self.decoder(tokens, enc_out)

# Helper Functions
def get_memory_stats(model, device, dummy_input_size=(1, 80, 3000)):
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers())

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        dummy_input = torch.randn(*dummy_input_size).to(device)
        if next(model.parameters()).dtype == torch.float16:
            dummy_input = dummy_input.half()
        _ = model.encoder(dummy_input)
        activation_mem = torch.cuda.max_memory_allocated() - param_mem - buffer_mem

    return param_mem, max(activation_mem, 0)

# Precompute FLOPs
print("\nPrecomputing FLOPs for models...")
for mname in model_list:
    try:
        model = whisper.load_model(mname, device="cpu")
        wrapper = WhisperFLOPsWrapper(model)
        macs, _ = get_model_complexity_info(
            wrapper,
            (80, 3000),
            as_strings=False,
            verbose=False,
            print_per_layer_stat=False
        )
        flops = 2 * macs
        model_flops[mname] = flops
        print(f" • {mname}: {flops/1e9:.2f} GFLOPs")
        del model, wrapper
    except Exception as e:
        print(f" • {mname}: Error during FLOPs calculation - {str(e)}")
        model_flops[mname] = 1e9  # Default value to prevent crashes

print(f"\nRunning on device: {device}")
print(f"{len(file_list)} audio files to process.\n")

# Benchmark Loop
for mname in model_list:
    for qmode in quantization_modes:
        print(f"--- Benchmarking '{mname}' | Quantization = {qmode} ---")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        model = None
        try:
            # Load model with fresh state
            model = whisper.load_model(mname, device=device)

            # Handle quantization modes
            use_fp16 = False
            if qmode == "dynamic":
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif qmode == "fp16":
                use_fp16 = True  # Let Whisper handle FP16 via autocast
                # Don't convert model to half() to avoid LayerNorm issues

            # Get memory statistics
            param_mem, activation_mem = get_memory_stats(model, device)
            total_mem = param_mem + activation_mem
            flops = model_flops.get(mname, 1e9)
            bpi = total_mem / flops if flops else 0

            print(f"Model size: {param_mem/1e6:.1f} MB | Activation: {activation_mem/1e6:.1f} MB | Bytes/Instr: {bpi:.4f}")

            # Warmup run
            warmup_file = os.path.join(audio_path, file_list[0])
            _ = model.transcribe(warmup_file, language="en", task="transcribe", fp16=use_fp16)

            # Process each audio file
            for fname in file_list:
                inp = os.path.join(audio_path, fname)
                base = os.path.splitext(fname)[0]
                odir = os.path.join(output_path, base)
                os.makedirs(odir, exist_ok=True)

                # Benchmark transcription
                start = time.perf_counter()
                res = model.transcribe(inp, language="en", task="transcribe", fp16=use_fp16)
                dur = time.perf_counter() - start

                # Save outputs
                with open(os.path.join(odir, f"{base}_{mname}_{qmode}.txt"), "w", encoding="utf-8") as t:
                    t.write(res["text"])
                with open(os.path.join(odir, f"{base}_{mname}_{qmode}.json"), "w", encoding="utf-8") as j:
                    json.dump(res, j, indent=2)

                # Calculate WER if reference exists
                wer_score = ""
                real_path = os.path.join(odir, f"{base}_REAL.txt")
                if os.path.exists(real_path):
                    ref = open(real_path, encoding="utf-8").read().strip()
                    wer_score = f"{wer(ref, res['text'].strip()):.4f}"

                # Get peak VRAM usage
                peak = ""
                if torch.cuda.is_available():
                    peak = f"{torch.cuda.max_memory_allocated() / (1024**2):.2f}"

                # Record results to CSV
                with open(csv_output_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        mname,
                        qmode,
                        fname,
                        f"{dur:.3f}",
                        wer_score,
                        peak,
                        f"{param_mem/1e6:.2f}",
                        f"{activation_mem/1e6:.2f}",
                        f"{bpi:.6f}"
                    ])

        except Exception as e:
            print(f"Error during benchmarking {mname} with quantization={qmode}:")
            traceback.print_exc()

        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()