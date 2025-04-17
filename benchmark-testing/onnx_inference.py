import os
import time
import csv
import numpy as np
import torch
import onnxruntime as ort
from transformers import WhisperProcessor
from datasets import load_dataset
import soundfile as sf
from pathlib import Path
import evaluate  # for WER

# Paths to exported ONNX models
ENCODER_PATH = "onnx_models/whisper_base_encoder.onnx"
DECODER_PATH = "onnx_models/whisper_base_decoder.onnx"
SAMPLES_DIR = "audio_files"
CSV_OUTPUT = "results.csv"

# Load processor and WER metric
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
wer_metric = evaluate.load("wer")

# ONNX runtime sessions
encoder_session = ort.InferenceSession(ENCODER_PATH)
decoder_session = ort.InferenceSession(DECODER_PATH)

def load_audio(path):
    audio, _ = sf.read(path)
    return audio

def greedy_decode(encoder_hidden_states, max_length=64):
    decoder_input_ids = np.array([[50258]], dtype=np.int64)  # decoder_start_token_id
    generated_ids = decoder_input_ids.tolist()[0]

    for _ in range(max_length):
        ort_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_hidden_states
        }
        ort_outs = decoder_session.run(None, ort_inputs)
        logits = ort_outs[0]

        next_token = np.argmax(logits[0, -1])
        if next_token == 50257:  # end-of-transcription
            break

        generated_ids.append(int(next_token))
        decoder_input_ids = np.array([generated_ids], dtype=np.int64)

    return generated_ids

def transcribe(audio_path):
    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    input_features = inputs.input_features
    encoder_outs = encoder_session.run(None, {"input_features": input_features})[0]
    token_ids = greedy_decode(encoder_outs)
    return processor.tokenizer.decode(token_ids, skip_special_tokens=True)

def load_references():
    # Optional: customize this if you have references in another format
    refs = {}
    for ref_file in Path(SAMPLES_DIR).glob("*.txt"):
        refs[ref_file.stem] = ref_file.read_text().strip().lower()
    return refs

if __name__ == "__main__":
    files = list(Path(SAMPLES_DIR).glob("*.mp3"))
    references = load_references()

    print(f"📍 Found {len(files)} files in `{SAMPLES_DIR}`")

    rows = []
    for file in files:
        name = file.stem
        print(f"🔍 Processing {file.name}...")

        try:
            start = time.time()
            pred = transcribe(str(file)).lower()
            import pdb;pdb.set_trace()
            duration = time.time() - start
            ref = references.get(name, "")
            wer = wer_metric.compute(predictions=[pred], references=[ref]) if ref else None

            print(f"📝 Prediction: {pred}")
            print(f"✅ Reference: {ref}")
            print(f"📉 WER: {wer:.3f}" if wer is not None else "⚠️ No reference")
            print(f"⏱️  Duration: {duration:.2f}s\n")

            rows.append({
                "file": file.name,
                "reference": ref,
                "prediction": pred,
                "wer": round(wer, 3) if wer is not None else "",
                "duration": round(duration, 2)
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            rows.append({
                "file": file.name,
                "reference": "",
                "prediction": "",
                "wer": "",
                "duration": ""
            })

    # Save results
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "reference", "prediction", "wer", "duration"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Results saved to `{CSV_OUTPUT}`")
