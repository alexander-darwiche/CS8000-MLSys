import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer
import onnxruntime as ort

# Configuration
MODEL_NAME = "openai/whisper-small"
QUANT_ENCODER_PATH = "whisper_encoder_quantized.onnx"

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_onnx_model_size(model_path):
    """Calculate ONNX model size in MB"""
    return os.path.getsize(model_path) / (1024 * 1024)

def create_whisper_attention_mask(seq_len):
    """Create attention mask matching Whisper's exact architecture requirements"""
    mask = torch.zeros((1, 1, seq_len, 32), dtype=torch.float16)  # Fixed 32-wide window
    if seq_len > 0:
        # Create causal mask for the active portion
        causal = torch.triu(torch.ones((1, 1, seq_len, seq_len), dtype=torch.float16), diagonal=1)
        mask[:, :, :seq_len, :seq_len] = causal
    mask = torch.where(mask == 1, -torch.inf, 0)
    return mask

def main():
    # 1. Load dataset, processor and model
    dataset = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    
    # Load original model and measure size
    original_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    original_model_size = get_model_size(original_model)
    
    # Convert to half precision for comparison
    model = original_model.half().eval()
    quant_model_size = get_onnx_model_size(QUANT_ENCODER_PATH) + get_model_size(model.model.decoder) + get_model_size(model.proj_out)
    
    print(f"Model Size Comparison:")
    print(f"Original Whisper model size: {original_model_size:.2f} MB")
    print(f"Quantized encoder + normal decoder size: {quant_model_size:.2f} MB")
    print(f"Reduction: {original_model_size - quant_model_size:.2f} MB ({((original_model_size - quant_model_size)/original_model_size)*100:.1f}% smaller)\n")
    
    # 2. Get first sample
    sample = dataset[0]
    print(f"Processing sample: {sample['text']}")

    # 3. Preprocess audio (ensure float16)
    processed = processor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt"
    )
    input_features = processed.input_features.half()

    # 4. Run encoder (using quantized encoder)
    ort_encoder = ort.InferenceSession(QUANT_ENCODER_PATH)
    encoder_outputs = ort_encoder.run(None, {
        "input_features": input_features.numpy().astype(np.float16)
    })
    encoder_hidden_states = torch.from_numpy(encoder_outputs[0]).half()

    # 5. Decoding loop using PyTorch decoder
    decoded_tokens = []
    decoder_input_ids = torch.tensor([[processor.tokenizer.bos_token_id]], dtype=torch.long, device="cpu")
    
    for step in range(200):  # max_length
        current_len = decoder_input_ids.shape[1]
        attention_mask = create_whisper_attention_mask(current_len)
        
        with torch.no_grad():
            outputs = model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask
            )
            # Get the last hidden state and project to vocabulary size
            last_hidden_state = outputs.last_hidden_state
            logits = model.proj_out(last_hidden_state)
            
        next_token = torch.argmax(logits[0, -1]).item()
        
        if next_token == processor.tokenizer.eos_token_id:
            break
            
        decoded_tokens.append(next_token)
        decoder_input_ids = torch.cat([
            decoder_input_ids,
            torch.tensor([[next_token]], dtype=torch.long, device="cpu")
        ], dim=1)

    # 6. Results
    if decoded_tokens:
        transcription = processor.decode(decoded_tokens)
        ground_truth = sample["text"].lower()
        wer_score = wer(ground_truth, transcription.lower())
        
        print(f"\nResults:")
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction: {transcription}")
        print(f"WER: {wer_score:.4f}")
    else:
        print("\nERROR: No tokens generated!")

if __name__ == "__main__":
    main()