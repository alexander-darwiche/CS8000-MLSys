import warnings
import csv
import time
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
from ptflops import get_model_complexity_info

# Suppress warnings
warnings.filterwarnings("ignore")

# CSV setup
csv_output_path = "winograd_benchmark_results.csv"
csv_headers = [
    "model", "winograd_mode", "file", "time_sec", "wer",
    "vram_mb", "theoretical_gflops", "actual_gflops"
]
with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(csv_headers)

# Load dataset
dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.select(range(5))  # First 5 samples

# Model list to benchmark
models_to_test = [
    "tiny", "small", "base", "medium", "large-v2"
]

# Config
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed FLOPs calculation wrapper
class WhisperFLOPsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder
        self.config = model.config
        
    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_input_ids = torch.tensor(
            [[self.config.decoder_start_token_id]],
            dtype=torch.long,
            device=x.device
        ).repeat(x.size(0), 1)
        return self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state
        )

for model_size in models_to_test:
    model_name = f"openai/whisper-{model_size}"
    print(f"\n=== Starting Benchmark for {model_name} ===")
    
    try:
        # Load processor
        processor = WhisperProcessor.from_pretrained(model_name)
        
        # Calculate theoretical FLOPS
        flops_model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        wrapper = WhisperFLOPsWrapper(flops_model)
        
        macs, _ = get_model_complexity_info(
            wrapper,
            (1, 80, 3000),
            as_strings=False,
            verbose=False,
            input_constructor=lambda _: {"x": torch.randn(1, 80, 3000).to(device)}
        )
        theoretical_gflops = 2 * macs / 1e9 if macs else 0
    except Exception as e:
        print(f"FLOPs calculation failed for {model_name}: {str(e)}")
        theoretical_gflops = 0
    finally:
        del flops_model, wrapper
        torch.cuda.empty_cache()

    print(f"Precomputed theoretical GFLOPS for {model_name}: {theoretical_gflops:.2f}")

    winograd_opts = [True, False]
    for use_winograd in winograd_opts:
        print(f"\n--- Benchmarking {model_name} with cuDNN.benchmark = {use_winograd} ---")
        
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = use_winograd
        
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        model.config.forced_decoder_ids = None

        for idx, sample in enumerate(dataset):
            try:
                audio = sample["audio"]["array"]
                sampling_rate = sample["audio"]["sampling_rate"]
                
                inputs = processor(
                    audio, 
                    sampling_rate=sampling_rate, 
                    return_tensors="pt"
                ).to(device)
                
                # Warmup
                _ = model.generate(inputs.input_features)

                # Benchmark
                start_time = time.perf_counter()
                predicted_ids = model.generate(inputs.input_features)
                duration = time.perf_counter() - start_time
                
                # Metrics
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                ref_norm = processor.tokenizer._normalize(sample["text"])
                hyp_norm = processor.tokenizer._normalize(transcription)
                wer_score = wer(ref_norm, hyp_norm)
                vram_usage = torch.cuda.max_memory_allocated() / (1024**2)
                actual_gflops = theoretical_gflops / duration if duration > 0 else 0
                
                # Write results
                with open(csv_output_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([
                        model_name,
                        use_winograd,
                        f"sample_{idx}",
                        f"{duration:.3f}",
                        f"{wer_score:.4f}",
                        f"{vram_usage:.2f}",
                        f"{theoretical_gflops:.2f}",
                        f"{actual_gflops:.2f}"
                    ])
                    
            except Exception as e:
                print(f"Error processing sample {idx}: {str(e)}")
            finally:
                del predicted_ids
                torch.cuda.empty_cache()

        del model
        torch.cuda.empty_cache()

print("\nBenchmarking complete!")