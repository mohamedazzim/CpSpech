# Speech Phrase Recognition Project

## Project Overview
This project is a robust, production-grade speech phrase recognition system. It is designed to recognize short spoken phrases in Tamil (or any language) using state-of-the-art deep learning (Wav2Vec2). The system supports data augmentation, noise robustness, speaker adaptation, live inference, and detailed evaluation.

---

## Workflow: Phase by Phase

### 1. **Data Preparation**
- **Raw Data:** Place your WAV files in `dataset/`, organized by phrase (each phrase in its own folder).
- **Preprocessing:** Normalize, resample, and pad/trim all audio to 16kHz, 3 seconds.
- **Manifest Creation:** Create train/val manifests for supervised learning.
- **(Optional) Noise Data:** Place real-world noise WAVs in `dataset/noise/` for advanced augmentation.
- **(Optional) Speaker Labels:** Add `speaker_id` to manifest entries for speaker adaptation.

### 2. **Model Training**
- Uses Wav2Vec2 as the backbone.
- On-the-fly data augmentation (pitch, time stretch, synthetic and real noise).
- Early stopping, learning rate scheduling, and best model saving.
- Per-epoch metrics logging (CSV) and confusion matrix (CSV + PNG).
- Optional speaker adaptation if speaker labels are present.

### 3. **Inference**
- Predict from WAV files or live microphone input.
- Optional text-to-speech (TTS) output.

### 4. **Evaluation**
- Confusion matrix and metrics for validation set.
- (Optional) Evaluate on noisy or out-of-domain data by creating a new manifest.

---

## File-by-File Explanation

### **configs/**
- `project_config.yaml`, `phrases.json`: Reserved for project-wide or phrase-specific configuration (currently empty, can be extended).

### **data_processing/**
- `preprocess.py`: Resamples, normalizes, and pads/trims all WAV files in the dataset. Run this before training.
- `split_manifest.py`: Splits the dataset into train/val manifests, ensuring class balance. Warns if any class is underrepresented.

### **dataset/**
- Contains all your phrase-labeled WAV files. Each subfolder is a phrase label. Add a `noise/` subfolder for real-world noise WAVs.

### **model/**
- `dataset.py`: Loads data, applies all augmentations, supports real noise mixing and speaker adaptation.
- `wav2vec_classifier.py`: Defines the Wav2Vec2-based classifier. Supports optional speaker embedding.
- `train.py`: Main training script. Handles all best practices: augmentation, early stopping, LR scheduling, metrics logging, confusion matrix, noise/speaker support.
- `export_onnx.py`: (Stub) For exporting the trained model to ONNX for deployment (extend as needed).
- `__init__.py`: (Empty, for module structure).

### **inference/**
- `real_time.py`: Predicts the phrase from a given WAV file using a trained model.
- `live_inference.py`: Records from the microphone, predicts the phrase, and can optionally speak the result using TTS. Production-grade, robust to errors.
- `tts_engine.py`: Simple text-to-speech utility using `pyttsx3`.

### **scripts/**
- `record_phrases.py`: CLI tool to record new phrase samples from the microphone and save them to the dataset.
- `check_wavs.py`: (Not fully detailed here) Likely checks dataset integrity or WAV file properties.

### **logs/**
- Stores training logs, per-epoch metrics (`metrics.csv`), and confusion matrix (`confusion_matrix.csv`, `confusion_matrix.png`).

### **manifests/**
- `train_manifest.json`, `val_manifest.json`: Lists of all training/validation samples and their labels (and optionally speaker IDs).

### **requirements.txt**
- Lists all Python dependencies (torch, torchaudio, transformers, librosa, soundfile, pyaudio, pyttsx3, etc.).

---

## Features & Best Practices
- **Wav2Vec2 backbone** for state-of-the-art speech recognition.
- **On-the-fly data augmentation:** pitch shift, time stretch, synthetic and real noise.
- **Speaker adaptation:** Optional, via speaker embeddings if `speaker_id` is present.
- **Early stopping** and **learning rate scheduling** for robust training.
- **Metrics logging** and **confusion matrix** for detailed evaluation.
- **Live/interactive inference** with microphone and TTS.
- **Error handling** and **modular code structure**.

---

## How to Run the Project (Step-by-Step)

### **1. Install Requirements**
```bash
pip install -r requirements.txt
```

### **2. Prepare Data**
- Place your WAV files in `dataset/<phrase>/`.
- (Optional) Place real noise WAVs in `dataset/noise/`.
- (Optional) Add `speaker_id` to manifest entries for speaker adaptation.

### **3. Preprocess Audio**
```bash
python data_processing/preprocess.py --dataset_path dataset
```

### **4. Create Train/Val Manifests**
```bash
python data_processing/split_manifest.py --dataset_root dataset --out_dir manifests
```

### **5. Train the Model**
```bash
python model/train.py
```
- Training logs, metrics, and confusion matrix will be saved in `logs/`.

### **6. Inference**
- **From WAV file:**
  ```bash
  python inference/real_time.py <audio_file.wav> models/best_model.pth manifests/train_manifest.json
  ```
- **Live from microphone (with optional TTS):**
  ```bash
  python inference/live_inference.py --model_ckpt models/best_model.pth --manifest_json manifests/train_manifest.json --speak --repeat 3
  ```

### **7. Evaluate on Noisy/Out-of-Domain Data**
- Create a new manifest for your noisy/out-of-domain set and use the validation logic in `train.py` or a custom script.

---

## Advanced Usage
- **Noise Robustness:** Add real noise WAVs to `dataset/noise/` for realistic augmentation.
- **Speaker Adaptation:** Add `speaker_id` to manifest entries and the model will use speaker embeddings.
- **Confusion Matrix:** After training, check `logs/confusion_matrix.png` and `logs/confusion_matrix.csv` for detailed error analysis.
- **Export for Deployment:** Extend `model/export_onnx.py` to export your trained model for production.

---

## Example Manifest Entry (with Speaker ID)
```json
{
  "sample_001": {
    "audio_path": "dataset/Amaravathi/amaravathi_1.wav",
    "phrase": "Amaravathi",
    "speaker_id": "spk1"
  }
}
```

---

## Contact & Contribution
- For questions, improvements, or contributions, please open an issue or pull request. 