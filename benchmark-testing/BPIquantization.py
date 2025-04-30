import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from jiwer import wer
import time
import csv

def measure_vram():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated() / (1024**2)
    return 0

# Setup
dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation").select(range(5))
models = ["tiny", "small", "base", "medium", "large-v2"]
quant_modes = ['full', 'fp16', 'dynamic']

with open("quant_benchmark_results2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["model", "quant_mode", "file", "time_sec", "wer", "vram_mb"])

    for model_size in models:
        model_name = f"openai/whisper-{model_size}"
        processor = WhisperProcessor.from_pretrained(model_name)
        
        for qmode in quant_modes:
            print(f"Processing {model_name} with {qmode}...")
            
            # Load and quantize model
            model = WhisperForConditionalGeneration.from_pretrained(model_name)
            if qmode == "fp16":
                model = model.half().cuda()
            elif qmode == "dynamic":
                model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).cpu()
            else:
                model = model.cuda()
            
            # Clear VRAM before each sample
            torch.cuda.empty_cache()
            
            for idx, sample in enumerate(dataset):
                try:
                    # Measure baseline VRAM
                    _ = measure_vram()
                    
                    # Process audio
                    inputs = processor(
                        sample["audio"]["array"],
                        sampling_rate=sample["audio"]["sampling_rate"],
                        return_tensors="pt"
                    ).to(model.device)
                    
                    if qmode == "fp16":
                        inputs.input_features = inputs.input_features.half()
                    
                    # Warmup
                    with torch.no_grad():
                        _ = model.generate(inputs.input_features)
                    
                    # Time and measure VRAM
                    start = time.time()
                    predicted_ids = model.generate(inputs.input_features)
                    duration = time.time() - start
                    vram = measure_vram()
                    
                    # Calculate WER
                    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                    wer_score = wer(
                        processor.tokenizer._normalize(sample["text"]),
                        processor.tokenizer._normalize(transcription)
                    )
                    
                    writer.writerow([
                        model_name,
                        qmode,
                        f"sample_{idx}",
                        f"{duration:.4f}",
                        f"{wer_score:.4f}",
                        f"{vram:.2f}"
                    ])
                    
                except Exception as e:
                    print(f"Error on {model_name} {qmode} sample {idx}: {str(e)}")
                finally:
                    torch.cuda.empty_cache()
            
            del model
            torch.cuda.empty_cache()