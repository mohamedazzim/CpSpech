# data_processing/preprocess.py

import librosa
import soundfile as sf
from pathlib import Path
import sys

TARGET_SR = 16000       # Target sample rate for all audio
TARGET_DURATION = 3.0   # Target duration in seconds
TARGET_LENGTH = int(TARGET_SR * TARGET_DURATION)  # Number of samples (16,000 * 3)


def preprocess_wav(file_path: Path, target_sr=TARGET_SR, target_length=TARGET_LENGTH):
    try:
        # Load audio, resample and convert to mono
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)

        # Pad or trim to target length
        if len(y) < target_length:
            y = librosa.util.fix_length(y, size=target_length)
        else:
            y = y[:target_length]

        # Normalize audio amplitude (peak normalization)
        y = librosa.util.normalize(y)

        # Save back to file with target sampling rate
        sf.write(file_path, y, target_sr)
        print(f"[OK] Processed {file_path}")

    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")


def preprocess_dataset(dataset_path: str):
    root = Path(dataset_path)
    if not root.exists():
        print(f"[ERROR] Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    print(f"Starting preprocessing on all WAV files under: {dataset_path}")
    wav_files = list(root.rglob("*.wav"))

    if not wav_files:
        print(f"[WARN] No .wav files found under {dataset_path}")
        return

    for wav_file in wav_files:
        preprocess_wav(wav_file)

    print("Preprocessing completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio dataset: resample, normalize, pad/trim.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset",
        help="Path to the dataset folder containing phrase subfolders with .wav files."
    )
    args = parser.parse_args()

    preprocess_dataset(args.dataset_path)
