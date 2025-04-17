import os
import csv
import numpy as np
import soundfile as sf
import onnxruntime as ort
from transformers import WhisperProcessor
from datasets import load_dataset
import evaluate

# Load processor and WER metric
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
wer_metric = evaluate.load("wer")

# Load ONNX sessions
encoder_session = ort.InferenceSession("onnx_models/whisper_base_encoder.onnx")
decoder_session = ort.InferenceSession("onnx_models/whisper_base_decoder.onnx")

# Greedy decoding loop
def greedy_decode(encoder_output, max_length=128):
    decoder_input_ids = np.array([[processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]], dtype=np.int64)
    output_ids = []

    for _ in range(max_length):
        ort_inputs = {
            "input_features": encoder_output,
            "decoder_input_ids": decoder_input_ids
        }
        logits = decoder_session.run(None, ort_inputs)[0]
        next_token_id = int(np.argmax(logits[0, -1, :]))

        if next_token_id == processor.tokenizer.eos_token_id:
            break

        output_ids.append(next_token_id)
        decoder_input_ids = np.concatenate([decoder_input_ids, [[next_token_id]]], axis=1)

    return output_ids

# Load and preprocess audio
def load_audio(path):
    audio, rate = sf.read(path)
    if rate != 16000:
        raise ValueError(f"Expected 16kHz audio, got {rate}")
    return audio

# Transcription pipeline
def transcribe(audio_path):
    audio = load_audio(audio_path)
    inputs = processor(audio, sampling_rate=16000, return_tensors="np")
    encoder_outs = encoder_session.run(None, {"input_features": inputs.input_features})[0]
    token_ids = greedy_decode(encoder_outs)
    return processor.tokenizer.decode(token_ids, skip_special_tokens=True)

# Main evaluation loop
def evaluate_directory(audio_dir, ground_truth_file, output_csv="results.csv"):
    

    rows = [("filename", "prediction", "ground_truth", "wer")]

    for file in os.listdir(audio_dir):

        with open('transcripts/'+str(file)+'/'+str(file)+'_REAL', newline='') as gt_file:
            gt_data = dict(csv.reader(gt_file))

        if not file.endswith(".mp3") and not file.endswith(".wav"):
            continue

        path = os.path.join(audio_dir, file)
        prediction = transcribe(path)
        reference = gt_data.get(file, "")

        single_wer = wer_metric.compute(predictions=[prediction], references=[reference])
        rows.append((file, prediction, reference, round(single_wer, 4)))

        print(f"📝 {file} — WER: {single_wer:.2f}")

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"\n📄 Results saved to {output_csv}")

# Example usage:
evaluate_directory("audio_files", "ground_truth.csv")
