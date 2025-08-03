import sys
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from model.dataset import SpeechPhraseDataset
from model.wav2vec_classifier import Wav2Vec2Classifier

def predict(audio_file, model_ckpt, manifest_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # Load dataset info to get label mapping
    dataset = SpeechPhraseDataset(manifest_json, feat_extractor)
    num_classes = dataset.num_classes()

    # Load model
    model = Wav2Vec2Classifier(num_classes=num_classes)
    try:
        state_dict = torch.load(model_ckpt, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"[ERROR] Failed to load model checkpoint: {e}")
        return
    model = model.to(device).eval()

    try:
        wav, sr = torchaudio.load(audio_file)
    except Exception as e:
        print(f"[ERROR] Unable to load audio file '{audio_file}': {e}")
        return

    # Mono + resample
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    # Pad/trim to 3 seconds
    x = wav[0]
    length = 16000 * 3
    if len(x) < length:
        x = torch.nn.functional.pad(x, (0, length - len(x)))
    else:
        x = x[:length]

    inputs = feat_extractor(x.numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(device)

    with torch.no_grad():
        logits = model(inputs)
        pred = logits.argmax(dim=1).item()

    phrase = dataset.label2phrase.get(pred, "Unknown")

    print(f"Predicted phrase: {phrase}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python real_time.py <audio_file.wav> <model_checkpoint.pth> <manifest.json>")
    else:
        predict(sys.argv[1], sys.argv[2], sys.argv[3])
