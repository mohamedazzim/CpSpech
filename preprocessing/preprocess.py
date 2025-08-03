import librosa
import soundfile as sf
from pathlib import Path

TARGET_SR = 16000
TARGET_DURATION = 3.0  # seconds

def preprocess_wav(file_path, target_sr=TARGET_SR, target_duration=TARGET_DURATION):
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
        samples = int(target_sr * target_duration)

        if len(y) < samples:
            y = librosa.util.fix_length(y, size=samples)
        else:
            y = y[:samples]

        y = librosa.util.normalize(y)
        sf.write(file_path, y, target_sr)
        print(f"[OK] Processed {file_path}")
    except Exception as e:
        print(f"[ERROR] Failed to process {file_path}: {e}")

def preprocess_all(dataset_root="dataset"):
    root = Path(dataset_root)
    print(f"Preprocessing WAV files in {root} ...")
    for label_dir in root.iterdir():
        if label_dir.is_dir():
            for wav_file in label_dir.glob("*.wav"):
                preprocess_wav(str(wav_file))
    print("Preprocessing completed.")

if __name__ == "__main__":
    preprocess_all()
