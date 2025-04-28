import torch
import time
import whisper
import os
import json
from jiwer import wer
import csv
from ptflops import get_model_complexity_info

# CSV output config
csv_output_path = "benchmark_results.csv"
csv_headers = ["model", "quant_mode", "file", "time_sec", "wer", "vram_mb", "bytes_per_instr"]

# Create CSV file and write headers
with open(csv_output_path, mode="w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

# Configure paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_path = r"audio_files"
output_path = r"transcripts"
os.makedirs(output_path, exist_ok=True)

# Get audio files
file_list = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]

if not file_list:
    print(f"No audio files found in {audio_path}")
    exit(1)

# Benchmark configuration
model_list = ['base', 'small']
quantization_options = ['full', 'fp16', 'dynamic']

# Precompute FLOPs for each model
model_flops = {}
print("\nPrecomputing FLOPs for models...")
for model_name in model_list:
    try:
        # Load model on CPU for FLOPs calculation
        model = whisper.load_model(model_name, device='cpu')
        
        # Create patched forward function for FLOPs calculation
        def patched_forward(mel_features):
            return model.decoder(model.encoder(mel_features))
        
        # Input shape: (batch_size, n_mels, n_frames)
        input_shape = (1, 80, 3000)  # Typical mel spectrogram shape
        macs, _ = get_model_complexity_info(
            patched_forward, 
            input_shape, 
            as_strings=False, 
            verbose=False,
            input_constructor=lambda x: {'mel_features': torch.randn(*x)}
        )
        
        flops = 2 * macs  # 1 MAC = 2 FLOPs
        model_flops[model_name] = flops
        print(f"Computed FLOPs for {model_name}: {flops / 1e9:.2f} GFLOPs")
        del model
    except Exception as e:
        print(f"Error precomputing FLOPs for {model_name}: {str(e)}")
        model_flops[model_name] = 1e9  # Default to 1 GFLOP to avoid division by zero

print(f"\nUsing device: {device}")
print(f"Found {len(file_list)} audio files")

for model_name in model_list:
    for quant_mode in quantization_options:
        print(f"\nBenchmarking {model_name} with Quantization Mode={quant_mode}")
        torch.cuda.empty_cache()
        try:
            # Load model
            model = whisper.load_model(model_name, device=device)
            
            # Apply quantization
            if quant_mode == 'dynamic':
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            elif quant_mode == 'fp16':
                model = model.half()

            # Calculate model memory footprint
            model_size_bytes = 0
            for param in model.parameters():
                if quant_mode == 'dynamic' and isinstance(param, torch.nn.Linear):
                    model_size_bytes += param.numel() * 1  # qint8: 1 byte/param
                elif param.dtype == torch.float16:
                    model_size_bytes += param.numel() * 2
                else:
                    model_size_bytes += param.numel() * 4

            # Calculate bytes per instruction
            flops = model_flops.get(model_name, 1e9)
            bytes_per_instr = model_size_bytes / flops if flops else 0

            print(f"Model size: {model_size_bytes / 1e6:.2f} MB")
            print(f"Bytes per instruction: {bytes_per_instr:.4f}")

            # Warmup
            use_fp16 = quant_mode == 'fp16'
            test_file = os.path.join(audio_path, file_list[0])
            model.transcribe(test_file, language='en', task='transcribe', fp16=use_fp16)

            # Benchmark files
            for file in file_list:
                try:
                    print(f"Processing {file} with {model_name} ({quant_mode})")
                    start_time = time.perf_counter()
                    
                    # Transcribe
                    result = model.transcribe(
                        os.path.join(audio_path, file),
                        language='en',
                        task='transcribe',
                        fp16=use_fp16
                    )
                    
                    # Calculate metrics
                    time_per = time.perf_counter() - start_time
                    peak_memory = torch.cuda.max_memory_allocated() / 1e6
                    
                    # Create output directory
                    file_name = os.path.splitext(file)[0]
                    output_dir = os.path.join(output_path, file_name)
                    os.makedirs(output_dir, exist_ok=True)

                    # Save results
                    output_file = os.path.join(output_dir, f"{file_name}_{model_name}_{quant_mode}.txt")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result['text'])

                    # Load real transcript
                    real_transcript_path = os.path.join(output_dir, f"{file_name}_REAL.txt")
                    with open(real_transcript_path, 'r', encoding='utf-8') as f:
                        real_transcript = f.read().strip()

                    # Compute WER
                    wer_score = wer(real_transcript, result['text'])
                    
                    # Write to CSV
                    with open(csv_output_path, mode="a", newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            model_name,
                            quant_mode,
                            file,
                            f"{time_per:.2f}",
                            f"{wer_score:.4f}",
                            f"{peak_memory:.2f}",
                            f"{bytes_per_instr:.4f}"
                        ])

                    print(f"Processed {file} | Time: {time_per:.2f}s | WER: {wer_score:.2%} | VRAM: {peak_memory:.2f}MB")

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

        except Exception as e:
            print(f"Error initializing {model_name} {quant_mode}: {str(e)}")
        finally:
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()