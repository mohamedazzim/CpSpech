import json
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
import random
import numpy as np
import librosa
import os
import glob
import torchaudio.functional as F_audio

class SpeechPhraseDataset(Dataset):
    def __init__(self, manifest_path, feature_extractor: Wav2Vec2FeatureExtractor, target_sr=16000, max_length_sec=3.0, augment=False, noise_dir=None):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.sample_ids = list(self.manifest.keys())
        self.feature_extractor = feature_extractor
        self.target_sr = target_sr
        self.max_length = int(target_sr * max_length_sec)
        self.augment = augment
        self.noise_dir = noise_dir
        self.noise_files = []
        if noise_dir is not None and os.path.isdir(noise_dir):
            self.noise_files = glob.glob(os.path.join(noise_dir, '*.wav'))

        unique_phrases = sorted({entry["phrase"] for entry in self.manifest.values()})
        self.phrase2label = {ph: i for i, ph in enumerate(unique_phrases)}
        self.label2phrase = {i: ph for ph, i in self.phrase2label.items()}

        # Speaker ID support
        self.has_speaker = any('speaker_id' in entry for entry in self.manifest.values())
        if self.has_speaker:
            unique_speakers = sorted({entry['speaker_id'] for entry in self.manifest.values() if 'speaker_id' in entry})
            self.speaker2idx = {spk: i for i, spk in enumerate(unique_speakers)}
        else:
            self.speaker2idx = None

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        max_retries = 5
        current_try = 0
        while current_try < max_retries:
            item = self.manifest[self.sample_ids[idx]]
            try:
                wav, sr = torchaudio.load(item["audio_path"])
                break  # Success: exit retry loop
            except Exception as e:
                print(f"[ERROR] Cannot load audio file '{item['audio_path']}': {e}")
                current_try += 1
                idx = random.randint(0, len(self) - 1)  # Try a different sample
        else:
            # After max retries, return dummy data with random label
            print(f"[WARNING] Skipping audio at index {idx} after {max_retries} failed loads.")
            dummy_input = torch.zeros(self.max_length, dtype=torch.float32)
            dummy_label = random.randint(0, self.num_classes() - 1)
            features = self.feature_extractor(
                dummy_input.numpy(),
                sampling_rate=self.target_sr,
                return_tensors="pt"
            ).input_values[0]
            return {"input_values": features, "label": torch.tensor(dummy_label)}

        # Mono channel if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.target_sr:
            wav = torchaudio.transforms.Resample(sr, self.target_sr)(wav)

        y = wav[0].numpy()
        # --- DATA AUGMENTATION (only if enabled) ---
        if self.augment:
            # Randomly apply one or more augmentations
            if np.random.rand() < 0.5:
                # Pitch shift by -2 to +2 semitones
                n_steps = np.random.uniform(-2, 2)
                y = librosa.effects.pitch_shift(y, sr=self.target_sr, n_steps=n_steps)
            if np.random.rand() < 0.5:
                # Time stretch between 0.9x and 1.1x using resampling approach
                rate = np.random.uniform(0.9, 1.1)
                # Resample to achieve time stretch effect
                new_length = int(len(y) / rate)
                y_resampled = librosa.util.fix_length(y, size=new_length)
                # Pad or trim to original length
                if len(y_resampled) < self.max_length:
                    y = np.pad(y_resampled, (0, self.max_length - len(y_resampled)))
                else:
                    y = y_resampled[:self.max_length]
            if np.random.rand() < 0.5:
                # Add Gaussian noise
                noise_amp = 0.005 * np.random.uniform() * np.amax(y)
                y = y + noise_amp * np.random.normal(size=y.shape)
            if self.noise_files and np.random.rand() < 0.7:
                # Add real-world noise with random SNR between 5 and 20 dB
                noise_path = random.choice(self.noise_files)
                noise, nsr = librosa.load(noise_path, sr=self.target_sr, mono=True)
                if len(noise) < self.max_length:
                    noise = np.pad(noise, (0, self.max_length - len(noise)))
                else:
                    noise = noise[:self.max_length]
                # Compute SNR
                snr_db = np.random.uniform(5, 20)
                signal_power = np.mean(y ** 2)
                noise_power = np.mean(noise ** 2)
                factor = np.sqrt(signal_power / (10 ** (snr_db / 10) * noise_power + 1e-8))
                y = y + factor * noise
        # --- END DATA AUGMENTATION ---

        # Pad or trim
        if len(y) < self.max_length:
            y = np.pad(y, (0, self.max_length - len(y)))
        else:
            y = y[:self.max_length]

        features = self.feature_extractor(
            y,
            sampling_rate=self.target_sr,
            return_tensors="pt"
        ).input_values[0]

        label = self.phrase2label[item["phrase"]]
        sample = {"input_values": features, "label": torch.tensor(label, dtype=torch.long)}
        if self.has_speaker and 'speaker_id' in item:
            sample["speaker_id"] = torch.tensor(self.speaker2idx[item["speaker_id"]], dtype=torch.long)
        return sample

    def num_classes(self):
        return len(self.phrase2label)
