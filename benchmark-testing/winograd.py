import platform
import ctypes.util
import warnings
import os
import csv

# Monkey-patch ctypes.util.find_library on Windows
if platform.system() == "Windows":
    original_find_library = ctypes.util.find_library
    def custom_find_library(name):
        if name == "c":
            return "msvcrt.dll"
        return original_find_library(name)
    ctypes.util.find_library = custom_find_library

# Suppress warnings about weights_only safe unpickling (informational only)
warnings.filterwarnings("ignore", message=".*weights_only.*")

import torch
import time
import whisper
import json
from jiwer import wer

# CSV Configuration
csv_output_path = "winograd_benchmark_results.csv"
csv_headers = ["model", "winograd_mode", "file", "time_sec", "wer", "vram_mb"]

# Create CSV and write headers
with open(csv_output_path, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# General Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_path = "audio_files"
output_path = "transcripts_winograd"
os.makedirs(output_path, exist_ok=True)

file_list = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]
if not file_list:
    print(f"No audio files found in {audio_path}")
    exit(1)

model_list = ['base','small']  # Add other models as needed
winograd_options = [True, False]

print(f"Using device: {device}")
print(f"Found {len(file_list)} audio files")

for model_name in model_list:
    for use_winograd in winograd_options:
        print(f"\nBenchmarking {model_name} with cuDNN Benchmarking={use_winograd}")
        torch.cuda.empty_cache()
        
        try:
            torch.backends.cudnn.benchmark = use_winograd
            model = whisper.load_model(
                name=model_name,
                device=device,
                download_root="models"
            )

            # Warmup
            test_file = os.path.join(audio_path, file_list[0])
            warmup_result = model.transcribe(test_file, language='en', task='transcribe')

            # Process each file
            for file in file_list:
                start_time = time.perf_counter()
                result = model.transcribe(
                    os.path.join(audio_path, file),
                    language='en',
                    task='transcribe'
                )

                # Save results
                base_name = os.path.splitext(file)[0]
                output_dir = os.path.join(output_path, base_name)
                os.makedirs(output_dir, exist_ok=True)

                # Save transcript and JSON
                transcript_filename = os.path.join(output_dir, f"{base_name}_{model_name}_winograd_{use_winograd}.txt")
                with open(transcript_filename, 'w') as f:
                    f.write(result['text'])
                
                json_filename = os.path.join(output_dir, f"{base_name}_{model_name}_winograd_{use_winograd}.json")
                with open(json_filename, 'w') as f:
                    json.dump(result, f, indent=2)

                # Calculate metrics
                duration = time.perf_counter() - start_time
                wer_score = None
                peak_mem = None

                # Check for real transcript
                real_transcript_path = os.path.join(output_dir, f"{base_name}_REAL.txt")
                if os.path.exists(real_transcript_path):
                    with open(real_transcript_path, 'r') as f:
                        real_transcript = f.read().strip()
                    wer_score = wer(real_transcript, result['text'].strip())

                # Get VRAM usage
                if torch.cuda.is_available():
                    peak_mem = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB

                # Write to CSV
                with open(csv_output_path, mode='a', newline='', encoding='utf-8') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([
                        model_name,
                        use_winograd,
                        file,
                        f"{duration:.2f}",
                        f"{wer_score:.4f}" if wer_score is not None else "",
                        f"{peak_mem:.2f}" if peak_mem is not None else ""
                    ])

                print(f"Processed {file} | Time: {duration:.2f}s | WER: {wer_score or 'N/A'} | VRAM: {peak_mem or 'N/A'}MB")

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()