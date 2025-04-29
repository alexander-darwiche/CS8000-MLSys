import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig
)
from jiwer import wer

# Configuration
MODEL_NAME = "openai/whisper-small"
NUM_SAMPLES = 5
WARMUP_RUNS = 3

def get_model_size(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def run_inference(model, input_features, generation_config):
    for _ in range(WARMUP_RUNS):
        _ = model.generate(inputs=input_features, generation_config=generation_config)
    start_time = time.time()
    outputs = model.generate(inputs=input_features, generation_config=generation_config)
    inference_time = time.time() - start_time
    return outputs, inference_time

def main():
    # 1. Load components
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).half().eval()

    # Model size
    print("\n=== Model Size ===")
    total_size = get_model_size(model)
    print(f"Full Whisper-small model size: {total_size:.2f} MB\n")

    # Generation config
    generation_config = GenerationConfig(
        max_length=200,
        num_beams=5,
        early_stopping=True
    )

    # Metrics
    total_wer = 0
    total_preprocess_time = 0
    total_inference_time = 0

    print(f"\n=== Processing {NUM_SAMPLES} samples ===")
    for i in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}: {sample['text']}")

        # Preprocessing
        start_time = time.time()
        processed = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        )
        input_features = processed.input_features.half()
        preprocess_time = time.time() - start_time
        total_preprocess_time += preprocess_time

        # Inference
        outputs, inference_time = run_inference(model, input_features, generation_config)
        total_inference_time += inference_time

        # Decode & Evaluate
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ground_truth = sample["text"].lower()
        wer_score = wer(ground_truth, transcription.lower())
        total_wer += wer_score

        # Results
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction:   {transcription}")
        print(f"WER: {wer_score:.4f}")
        print(f"Preprocess: {preprocess_time:.3f}s")
        print(f"Inference:  {inference_time:.3f}s")
        print(f"Total:      {preprocess_time + inference_time:.3f}s")

    # Summary
    print("\n=== Summary ===")
    print(f"Average WER: {total_wer/NUM_SAMPLES:.4f}")
    print(f"Average Preprocess Time: {total_preprocess_time/NUM_SAMPLES:.3f}s")
    print(f"Average Inference Time: {total_inference_time/NUM_SAMPLES:.3f}s")
    print(f"Average Total Time: {(total_preprocess_time + total_inference_time)/NUM_SAMPLES:.3f}s")

if __name__ == "__main__":
    main()
