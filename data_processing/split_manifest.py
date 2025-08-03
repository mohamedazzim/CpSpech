import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def build_manifest(dataset_root="dataset", out_dir="manifests", test_size=0.25, random_state=42):
    dataset_root = Path(dataset_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for label_dir in dataset_root.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            for wav_path in label_dir.glob("*.wav"):
                samples.append({"audio_path": str(wav_path), "phrase": label})

    num_classes = len(set([x["phrase"] for x in samples]))

    total_samples = len(samples)
    if isinstance(test_size, float):
        val_size = int(total_samples * test_size)
    else:
        val_size = test_size

    if val_size < num_classes:
        print(f"[WARN] Adjusting val set size from {test_size} to {num_classes} to satisfy stratification.")
        test_size = num_classes if num_classes < total_samples else int(total_samples * 0.2)

    try:
        train, val = train_test_split(
            samples,
            test_size=test_size,
            stratify=[x["phrase"] for x in samples],
            random_state=random_state
        )
    except ValueError as e:
        print(f"[ERROR] Stratified split failed: {e}")
        print("Falling back to random split without stratification (may cause imbalance).")
        train, val = train_test_split(
            samples,
            test_size=test_size,
            random_state=random_state
        )

    train_manifest = {f"sample_{i}": ex for i, ex in enumerate(train)}
    val_manifest = {f"sample_{i}": ex for i, ex in enumerate(val)}

    with open(out_dir / "train_manifest.json", "w") as f:
        json.dump(train_manifest, f, indent=2)

    with open(out_dir / "val_manifest.json", "w") as f:
        json.dump(val_manifest, f, indent=2)

    print(f"Manifest created: Train samples={len(train)}, Val samples={len(val)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="dataset", help="Path to dataset root")
    parser.add_argument("--out_dir", type=str, default="manifests", help="Output directory for manifests")
    args = parser.parse_args()
    build_manifest(dataset_root=args.dataset_root, out_dir=args.out_dir)

    # Print class/sample stats
    from collections import Counter
    dataset_root = Path(args.dataset_root)
    samples_per_class = Counter()
    for label_dir in dataset_root.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            n = len(list(label_dir.glob("*.wav")))
            samples_per_class[label] = n
    print("\nSamples per class:")
    for label, n in samples_per_class.items():
        print(f"  {label}: {n}")
        if n < 5:
            print(f"    [WARN] Class '{label}' has very few samples! Minimum recommended: 5-10.")
    print(f"Total classes: {len(samples_per_class)}")
    print(f"Total samples: {sum(samples_per_class.values())}")
