import os
import time
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,  # Fixed typo in class name
    GenerationConfig
)
from jiwer import wer
import onnxruntime as ort

# Configuration
MODEL_NAME = "openai/whisper-small"
QUANT_ENCODER_PATH = "whisper_encoder_quantized.onnx"
NUM_SAMPLES = 5  # Number of samples to process
WARMUP_RUNS = 3   # Number of warmup runs before timing

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def get_onnx_model_size(model_path):
    """Calculate ONNX model size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def create_whisper_attention_mask(seq_len):
    """Create attention mask matching Whisper's architecture"""
    mask = torch.zeros((1, 1, seq_len, seq_len), dtype=torch.float16)
    if seq_len > 0:
        causal = torch.triu(torch.ones((1, 1, seq_len, seq_len)), diagonal=1)
        mask = mask + causal
    return torch.where(mask == 1, -torch.inf, 0)

def run_inference(processor, model, input_features, generation_config):
    """Run inference and measure timing"""
    # Warmup runs
    for _ in range(WARMUP_RUNS):
        _ = model.generate(inputs=input_features, generation_config=generation_config)
    
    # Timed runs
    start_time = time.time()
    outputs = model.generate(inputs=input_features, generation_config=generation_config)
    inference_time = time.time() - start_time
    
    return outputs, inference_time

def main():
    # 1. Load components
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    
    # Model size comparison
    print("\n=== Model Size Comparison ===")
    original_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    original_size = get_model_size(original_model)
    
    quant_encoder_size = get_onnx_model_size(QUANT_ENCODER_PATH)
    model = original_model.half().eval()
    decoder_size = get_model_size(model.model.decoder)
    proj_size = get_model_size(model.proj_out)
    hybrid_size = quant_encoder_size + decoder_size + proj_size
    
    print(f"Original Model: {original_size:.2f} MB")
    print(f"Quantized Encoder: {quant_encoder_size:.2f} MB")
    print(f"Decoder: {decoder_size:.2f} MB")
    print(f"Projection Layer: {proj_size:.2f} MB")
    print(f"Hybrid Model Total: {hybrid_size:.2f} MB")
    print(f"Size Reduction: {original_size-hybrid_size:.2f} MB ({(original_size-hybrid_size)/original_size*100:.1f}%)\n")
    
    # Generation config
    generation_config = GenerationConfig(
        max_length=200,
        num_beams=5,
        early_stopping=True
    )
    
    # Initialize metrics
    total_wer = 0
    total_preprocess_time = 0
    total_encode_time = 0
    total_decode_time = 0
    total_inference_time = 0
    
    # Process multiple samples
    print(f"\n=== Processing {NUM_SAMPLES} samples ===")
    for i in range(min(NUM_SAMPLES, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i+1}: {sample['text']}")
        
        # 1. Preprocess audio
        start_time = time.time()
        processed = processor(
            sample["audio"]["array"],
            sampling_rate=sample["audio"]["sampling_rate"],
            return_tensors="pt"
        )
        input_features = processed.input_features.half()
        preprocess_time = time.time() - start_time
        total_preprocess_time += preprocess_time
        
        # 2. Run quantized encoder
        start_time = time.time()
        ort_session = ort.InferenceSession(QUANT_ENCODER_PATH)
        encoder_outputs = ort_session.run(
            None, {"input_features": input_features.numpy().astype(np.float16)}
        )
        encoder_hidden_states = torch.from_numpy(encoder_outputs[0]).half()
        encode_time = time.time() - start_time
        total_encode_time += encode_time
        
        # 3. Run decoder with timing
        outputs, decode_time = run_inference(processor, model, input_features, generation_config)
        total_decode_time += decode_time
        
        # 4. Calculate metrics
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        ground_truth = sample["text"].lower()
        wer_score = wer(ground_truth, transcription.lower())
        total_wer += wer_score
        
        # 5. Print sample results
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction:   {transcription}")
        print(f"WER: {wer_score:.4f}")
        print(f"Preprocess: {preprocess_time:.3f}s")
        print(f"Encoding:   {encode_time:.3f}s")
        print(f"Decoding:   {decode_time:.3f}s")
        print(f"Total:      {preprocess_time+encode_time+decode_time:.3f}s")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average WER: {total_wer/NUM_SAMPLES:.4f}")
    print(f"Average Preprocess Time: {total_preprocess_time/NUM_SAMPLES:.3f}s")
    print(f"Average Encoding Time: {total_encode_time/NUM_SAMPLES:.3f}s")
    print(f"Average Decoding Time: {total_decode_time/NUM_SAMPLES:.3f}s")
    print(f"Average Total Time: {(total_preprocess_time+total_encode_time+total_decode_time)/NUM_SAMPLES:.3f}s")

if __name__ == "__main__":
    main()