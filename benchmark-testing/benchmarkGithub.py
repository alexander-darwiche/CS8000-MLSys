import torch
import time
import whisper
import os
import subprocess
import json

# Configure paths
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
audio_path = r"C:\Users\willi\Documents\GitHub\CS8000-MLSys\benchmark-testing\audio_files"
output_path = r"C:\Users\willi\Documents\GitHub\CS8000-MLSys\benchmark-testing\transcripts"
os.makedirs(output_path, exist_ok=True)

# Get audio files
file_list = [f for f in os.listdir(audio_path) if f.endswith(('.wav', '.mp3'))]

if not file_list:
    print(f"No audio files found in {audio_path}")
    exit(1)

# Benchmark configuration
model_list = ['medium', 'large-v2']
fp16_options = [True, False]

print(f"Using device: {device}")
print(f"Found {len(file_list)} audio files")

for model_name in model_list:
    for use_fp16 in fp16_options:
        print(f"\nBenchmarking {model_name} with FP16={use_fp16}")
        
        try:
            # Load model
            model = whisper.load_model(
                name=model_name,
                device=device,
                download_root=r"C:\Users\willi\Documents\GitHub\CS8000-MLSys\models"
            )
            
            # Warmup
            test_file = os.path.join(audio_path, file_list[0])
            model.transcribe(test_file, language='en', task='transcribe', fp16=use_fp16)
            
            # Benchmark and save transcripts
            total_time = 0.0
            for file in file_list:
                start = time.perf_counter()
                
                # Transcribe
                result = model.transcribe(
                    os.path.join(audio_path, file),
                    language='en',
                    task='transcribe',
                    fp16=use_fp16
                )
                
                # Save transcript
                output_file = os.path.join(output_path, f"{os.path.splitext(file)[0]}_{model_name}_fp16_{use_fp16}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['text'])
                
                # Save full results as JSON
                json_file = os.path.join(output_path, f"{os.path.splitext(file)[0]}_{model_name}_fp16_{use_fp16}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                
                total_time += time.perf_counter() - start
            
            avg_time = total_time / len(file_list)
            print(f"Average time per file: {avg_time:.2f}s")
            print(f"Transcripts saved to: {output_path}")
            
        except Exception as e:
            print(f"Error with {model_name} FP16={use_fp16}: {str(e)}")
        finally:
            if 'model' in locals():
                del model
                torch.cuda.empty_cache()