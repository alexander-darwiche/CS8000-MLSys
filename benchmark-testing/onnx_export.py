import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os

output_dir = "onnx_models"
os.makedirs(output_dir, exist_ok=True)

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

model.eval()

# ---------------------- Export Encoder ----------------------
print("📤 Exporting encoder...")

class WhisperEncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features):
        return self.encoder(input_features)

encoder_wrapper = WhisperEncoderWrapper(model)
encoder_input = torch.randn(1, 80, 3000)  # batch, feature, time

torch.onnx.export(
    encoder_wrapper,
    encoder_input,
    os.path.join(output_dir, "whisper_base_encoder.onnx"),
    input_names=["input_features"],
    output_names=["encoder_output"],
    dynamic_axes={"input_features": {0: "batch", 2: "time"}, "encoder_output": {0: "batch", 1: "time"}},
    opset_version=17
)

# ---------------------- Export Decoder (with past) ----------------------
print("📤 Exporting decoder with past...")

class WhisperDecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.model.decoder

    def forward(self, decoder_input_ids, encoder_outputs):
        return self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
        ).last_hidden_state

decoder_wrapper = WhisperDecoderWrapper(model)

# Dummy inputs
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
encoder_outputs = torch.randn(1, 1500, model.config.d_model)  # seq_len, d_model

torch.onnx.export(
    decoder_wrapper,
    (decoder_input_ids, encoder_outputs),
    os.path.join(output_dir, "whisper_base_decoder.onnx"),
    input_names=["decoder_input_ids", "encoder_outputs"],
    output_names=["logits"],
    dynamic_axes={
        "decoder_input_ids": {0: "batch", 1: "seq_len"},
        "encoder_outputs": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=17
)

print("✅ Done exporting both encoder and decoder.")
