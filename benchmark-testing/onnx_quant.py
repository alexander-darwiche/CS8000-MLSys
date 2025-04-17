from onnxruntime.quantization import quantize_dynamic, QuantType

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")  # or tiny/small/etc.
model.eval()

# Dummy input
dummy_input = torch.randint(0, 1_000, (1, 300))  # (batch_size, sequence_length)

torch.onnx.export(
    model,
    (dummy_input,),
    "whisper.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=13,
    do_constant_folding=True,
    dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}}
)