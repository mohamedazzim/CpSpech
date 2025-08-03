import os
import json
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from model.dataset import SpeechPhraseDataset
from model.wav2vec_classifier import Wav2Vec2Classifier
from collections import defaultdict, Counter
import argparse

def load_model_and_dataset(model_ckpt, manifest_json):
    """Load the trained model and dataset for testing."""
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

def predict_audio_file(wav_path, model, dataset, feat_extractor, device):
    """Predict the phrase for a single audio file."""
    try:
        wav, sr = torchaudio.load(wav_path)
    except Exception as e:
        print(f"[ERROR] Unable to load audio file '{wav_path}': {e}")
        return None
    
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
    return phrase

def test_all_audio_files(model_ckpt, manifest_json, dataset_root="dataset"):
    """Test all audio files in the dataset and report results."""
    print("Loading model and dataset...")
    model, dataset, feat_extractor, device = load_model_and_dataset(model_ckpt, manifest_json)
    if model is None:
        return
    
    print(f"Testing all audio files in {dataset_root}...")
    print("=" * 80)
    
    # Get all audio files from dataset
    all_files = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                # Extract expected phrase from path
                relative_path = os.path.relpath(file_path, dataset_root)
                expected_phrase = relative_path.split(os.sep)[0]
                all_files.append((file_path, expected_phrase))
    
    print(f"Found {len(all_files)} audio files to test")
    print("=" * 80)
    
    # Test each file
    results = []
    class_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})
    
    for i, (file_path, expected_phrase) in enumerate(all_files, 1):
        print(f"Testing {i}/{len(all_files)}: {os.path.basename(file_path)}")
        print(f"  Expected: {expected_phrase}")
        
        predicted_phrase = predict_audio_file(file_path, model, dataset, feat_extractor, device)
        
        if predicted_phrase is None:
            print(f"  [ERROR] Failed to predict")
            continue
        
        is_correct = predicted_phrase == expected_phrase
        results.append({
            'file': file_path,
            'expected': expected_phrase,
            'predicted': predicted_phrase,
            'correct': is_correct
        })
        
        class_results[expected_phrase]['total'] += 1
        if is_correct:
            class_results[expected_phrase]['correct'] += 1
        class_results[expected_phrase]['predictions'].append(predicted_phrase)
        
        status = "✅ CORRECT" if is_correct else "❌ WRONG"
        print(f"  Predicted: {predicted_phrase}")
        print(f"  Result: {status}")
        print()
    
    # Calculate overall statistics
    total_files = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    overall_accuracy = correct_predictions / total_files if total_files > 0 else 0
    
    # Print overall results
    print("=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    print(f"Total files tested: {total_files}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print()
    
    # Print per-class results
    print("PER-CLASS RESULTS")
    print("=" * 80)
    for phrase, stats in sorted(class_results.items()):
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{phrase}:")
        print(f"  Accuracy: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
        if stats['total'] > 0:
            prediction_counts = Counter(stats['predictions'])
            print(f"  Predictions: {dict(prediction_counts)}")
        print()
    
    # Print confusion matrix summary
    print("CONFUSION SUMMARY")
    print("=" * 80)
    confusion = defaultdict(lambda: defaultdict(int))
    for result in results:
        confusion[result['expected']][result['predicted']] += 1
    
    for expected in sorted(confusion.keys()):
        print(f"{expected} ->")
        for predicted, count in sorted(confusion[expected].items()):
            if expected != predicted:
                print(f"  {predicted}: {count}")
        print()
    
    # Save detailed results to file
    results_file = "test_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'overall_accuracy': overall_accuracy,
            'total_files': total_files,
            'correct_predictions': correct_predictions,
            'per_class_results': dict(class_results),
            'detailed_results': results
        }, f, indent=2)
    
    print(f"Detailed results saved to {results_file}")
    return overall_accuracy, results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all audio files in the dataset")
    parser.add_argument("--model_ckpt", type=str, default="models/best_model.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--manifest_json", type=str, default="manifests/train_manifest.json",
                       help="Path to manifest JSON file")
    parser.add_argument("--dataset_root", type=str, default="dataset",
                       help="Path to dataset root directory")
    
    args = parser.parse_args()
    
    test_all_audio_files(args.model_ckpt, args.manifest_json, args.dataset_root) 