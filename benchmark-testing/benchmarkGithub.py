import torch
import time
import whisper
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_list = ['medium', 'large-v2']
fp16_bool = [True, False]
# Correct path to directory and use raw string
path = r"C:\Users\willi\Documents\GitHub\CS8000-MLSys\benchmark-testing"
file_list = os.listdir(path)

for model_name in model_list:
    for use_fp16 in fp16_bool:
        print(f"\nLoading {model_name} with fp16={use_fp16}...")
        
        # Load model with device_map
        model = whisper.load_model(
            name=model_name,
            device=device,
            download_root="./models"
        )
        
        # Warmup
        audio = whisper.load_audio(os.path.join(path, file_list[0]), sr=16000)
        model.transcribe(audio, language='en', task='transcribe', fp16=use_fp16)
        
        # Benchmark
        total_time = 0.0
        for file in file_list:
            audio_path = os.path.join(path, file)
            audio = whisper.load_audio(audio_path, sr=16000)
            start = time.perf_counter()
            result = model.transcribe(audio, language='en', task='transcribe', fp16=use_fp16)
            total_time += time.perf_counter() - start
        
        avg_time = total_time / len(file_list)
        print(f"Model: {model_name} | FP16: {use_fp16} | Avg time per file: {avg_time:.2f}s")
        
        del model  # Clean up memory
        torch.cuda.empty_cache()