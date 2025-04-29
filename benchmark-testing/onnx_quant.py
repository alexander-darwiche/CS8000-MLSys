import torch
import numpy as np
import onnx
import onnxruntime as ort
import os
from transformers import WhisperForConditionalGeneration
from onnxruntime.quantization import (
    QuantType, QuantFormat,
    CalibrationDataReader,
    quantize_static, quantize_dynamic
)
from onnxruntime.transformers.float16 import convert_float_to_float16

# Configuration
MODEL_NAME = "openai/whisper-small"
ENCODER_ONNX_PATH = "whisper_encoder.onnx"
QUANT_ENCODER_PATH = "whisper_encoder_quantized.onnx"
OPSET_VERSION = 15  # Recommended for best compatibility

class WhisperCalibrator(CalibrationDataReader):
    def __init__(self, num_samples=32):
        self.num_samples = num_samples
        self.current_sample = 0
        self.samples = [
            np.random.randn(1, 80, 3000).astype(np.float32) * 2.5 + 12.5  # ~N(12.5, 2.5)
            for _ in range(num_samples)
        ]
    
    def get_next(self):
        if self.current_sample >= self.num_samples:
            return None
        sample = {"input_features": np.clip(self.samples[self.current_sample], -100, 100)}
        self.current_sample += 1
        return sample

def export_encoder():
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    
    dummy_input = torch.randn(1, 80, 3000, dtype=torch.float32)
    
    torch.onnx.export(
        model.model.encoder,
        dummy_input,
        ENCODER_ONNX_PATH,
        input_names=["input_features"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "input_features": {0: "batch_size", 2: "n_frames"},
            "encoder_hidden_states": {0: "batch_size", 1: "n_frames"}
        },
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        verbose=True
    )
    
    # Validate export
    onnx.checker.check_model(onnx.load(ENCODER_ONNX_PATH))
    print(f"Encoder successfully exported to {ENCODER_ONNX_PATH}")

def quantize_encoder():
    strategies = [
        {
            'name': 'FP16',
            'quant_fn': lambda: onnx.save_model(
                convert_float_to_float16(onnx.load_model(ENCODER_ONNX_PATH)),
                QUANT_ENCODER_PATH
            ),
            'is_fp16': True
        },
        {
            'name': 'QDQ INT8',
            'quant_fn': lambda: quantize_static(
                ENCODER_ONNX_PATH,
                QUANT_ENCODER_PATH,
                WhisperCalibrator(64),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                op_types_to_quantize=['MatMul', 'LayerNormalization'],
                extra_options={
                    "WeightSymmetric": True,
                    "AddQDQPairToWeight": True
                }
            ),
            'is_fp16': False
        }
    ]

    for strategy in strategies:
        print(f"\nTrying strategy: {strategy['name']}")
        try:
            strategy['quant_fn']()
            print(f"✅ Successfully quantized with {strategy['name']}")
            return
        except Exception as e:
            print(f"⚠️ Failed with {strategy['name']}: {str(e)}")
    
    raise RuntimeError("All quantization strategies failed")

if __name__ == "__main__":
    print("=== Exporting Whisper Encoder ===")
    export_encoder()
    
    print("\n=== Quantizing Encoder ===")
    quantize_encoder()
    print(f"Quantized encoder saved to {QUANT_ENCODER_PATH}")