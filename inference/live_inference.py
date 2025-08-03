import argparse
import torch
import torchaudio
import pyaudio
import wave
from transformers import Wav2Vec2FeatureExtractor
from model.dataset import SpeechPhraseDataset
from model.wav2vec_classifier import Wav2Vec2Classifier
from inference.tts_engine import speak
import tempfile
import os

# Audio recording parameters
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
RECORD_SECONDS = 3

def record_audio(temp_wav_path, duration=RECORD_SECONDS):
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
        print(f"Recording for {duration} seconds. Please speak now...")
        frames = []
        for _ in range(0, int(SAMPLE_RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Recording finished.")
    except Exception as e:
        print(f"[ERROR] Failed to record audio: {e}")
        return False
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except:
            pass
        p.terminate()
    try:
        wf = wave.open(temp_wav_path, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
        wf.close()
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save audio file: {e}")
        return False

def load_model_and_dataset(model_ckpt, manifest_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    dataset = SpeechPhraseDataset(manifest_json, feat_extractor)
    num_classes = dataset.num_classes()
    model = Wav2Vec2Classifier(num_classes=num_classes)
    try:
        state_dict = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load model checkpoint: {e}")
        return None, None, None, None
    model = model.to(device).eval()
    return model, dataset, feat_extractor, device

def predict_from_wav(wav_path, model, dataset, feat_extractor, device):
    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as e:
        print(f"[ERROR] Unable to load audio file '{wav_path}': {e}")
        return None
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    x = wav[0]
    length = SAMPLE_RATE * RECORD_SECONDS
    if len(x) < length:
        x = torch.nn.functional.pad(x, (0, length - len(x)))
    else:
        x = x[:length]
    inputs = feat_extractor(x.numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        logits = model(inputs)
        pred = logits.argmax(dim=1).item()
    phrase = dataset.label2phrase.get(pred, "Unknown")
    return phrase

def main():
    parser = argparse.ArgumentParser(description="Live/Interactive Speech Phrase Recognition")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--manifest_json", type=str, required=True, help="Path to manifest JSON file")
    parser.add_argument("--speak", action="store_true", help="Speak the predicted phrase using TTS")
    parser.add_argument("--repeat", type=int, default=1, help="Number of live predictions to run (default: 1)")
    args = parser.parse_args()

    model, dataset, feat_extractor, device = load_model_and_dataset(args.model_ckpt, args.manifest_json)
    if model is None:
        return

    for i in range(args.repeat):
        print(f"\n--- Live Inference Iteration {i+1} ---")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            temp_wav_path = tmpfile.name
        try:
            if not record_audio(temp_wav_path):
                continue
            phrase = predict_from_wav(temp_wav_path, model, dataset, feat_extractor, device)
            if phrase is not None:
                print(f"Predicted phrase: {phrase}")
                if args.speak:
                    speak(phrase)
            else:
                print("Prediction failed.")
        finally:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

if __name__ == "__main__":
    main() 