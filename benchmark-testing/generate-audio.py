import os
import torchaudio
from tqdm import tqdm

dataset = torchaudio.datasets.LIBRISPEECH(".", url="test-clean", download=True)
dataset = dataset[:10]

out_audio_dir = "audio_files"
out_ref_dir = "transcripts"
os.makedirs(out_audio_dir, exist_ok=True)
os.makedirs(out_ref_dir, exist_ok=True)

for i, (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id) in tqdm(enumerate(dataset)):
    fname = f"{speaker_id}-{chapter_id}-{utterance_id}"
    wav_path = os.path.join(out_audio_dir, f"{fname}.wav")
    ref_path = os.path.join(out_ref_dir, f"{fname}_REAL.txt")

    # Save .wav
    torchaudio.save(wav_path, waveform, sample_rate)

    # Save reference transcript
    with open(ref_path, "w", encoding="utf-8") as f:
        f.write(transcript)
