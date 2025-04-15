import torch
import time
import whisper
import os
import subprocess
import json
from jiwer import wer

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
model_list = ['base','small']#,'tiny','medium','large-v2','turbo']
quantization_options = ['full', 'fp16','dynamic']

print(f"Using device: {device}")
print(f"Found {len(file_list)} audio files")

for model_name in model_list:
    # Test with and without FP16 Quantization
    for quant_mode in quantization_options:
        print(f"\nBenchmarking {model_name} with Quantization Mode={quant_mode}")
        torch.cuda.empty_cache()
        try:
            # Load model
            model = whisper.load_model(
                name=model_name,
                device=device,
                download_root=r"models"
            )
            if quant_mode == 'Dynamic':
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            # Warmup
            use_fp16 = True if quant_mode == 'fp16' else False
            test_file = os.path.join(audio_path, file_list[0])
            model.transcribe(test_file, language='en', task='transcribe', fp16=use_fp16)
            
            # Benchmark and save transcripts
            total_time = 0.0
            for file in file_list:
                print("Testing the speed of: " + str(file) + " with Model Type: " + str(model_name))
                start = time.perf_counter()
                
                # Transcribe
                result = model.transcribe(
                    os.path.join(audio_path, file),
                    language='en',
                    task='transcribe',
                    fp16=use_fp16
                )
                
                file_name = os.path.splitext(file)[0]
                output_path2 = output_path + "/" + file_name

                # Save transcript
                output_file = os.path.join(output_path2, f"{os.path.splitext(file)[0]}_{model_name}_{quant_mode}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                # Save full results as JSON
                json_file = os.path.join(output_path2, f"{os.path.splitext(file)[0]}_{model_name}_{quant_mode}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                total_time += time.perf_counter() - start
                time_per = time.perf_counter() - start
                
                # Load real transcript
                real_transcript_path = os.path.join(output_path2, f"{os.path.splitext(file)[0]}_REAL.txt")
                with open(real_transcript_path, 'r', encoding='utf-8') as f:
                    real_transcript = f.read().strip()  # Read and remove extra spaces
                
                # Load real transcript
                predicted_transcript_path = os.path.join(output_path2, f"{os.path.splitext(file)[0]}_{model_name}_{quant_mode}.txt")
                with open(predicted_transcript_path, 'r', encoding='utf-8') as f:
                    predicted_transcript = f.read().strip()  # Read and remove extra spaces

                # Compute WER
                wer2 = wer(real_transcript, predicted_transcript)
                print(f"WER: {wer2:.2%}")
                print(f"Time per file: {time_per:.2f}s")
                peak_memory = torch.cuda.max_memory_allocated() / 1e6
                print(f"Peak VRAM Usage: {peak_memory:.2f} MB")

            avg_time = total_time / len(file_list)
            print(f"Average time per file: {avg_time:.2f}s")
            print(f"Transcripts saved to: {output_path2}")
            
        except Exception as e:
            print(f"Error with {model_name} FP16={quant_mode}: {str(e)}")
        finally:
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()