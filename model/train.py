import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import csv
import matplotlib.pyplot as plt
import numpy as np

from .dataset import SpeechPhraseDataset
from .wav2vec_classifier import Wav2Vec2Classifier

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use project-root-relative paths for manifests
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_manifest_path = os.path.join(project_root, "manifests", "train_manifest.json")
    val_manifest_path = os.path.join(project_root, "manifests", "val_manifest.json")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    noise_dir = os.path.join(project_root, "dataset", "noise")
    train_dataset = SpeechPhraseDataset(train_manifest_path, feature_extractor, augment=True, noise_dir=noise_dir)
    val_dataset = SpeechPhraseDataset(val_manifest_path, feature_extractor, augment=False)

    num_classes = train_dataset.num_classes()
    # Speaker adaptation support
    num_speakers = len(train_dataset.speaker2idx) if train_dataset.has_speaker else None
    model = Wav2Vec2Classifier(num_classes=num_classes, num_speakers=num_speakers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0)

    best_val_acc = 0
    patience = 5
    patience_counter = 0
    metrics_path = os.path.join(project_root, "logs", "metrics.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_acc", "val_f1", "lr"])
        try:
            for epoch in range(1, 51):  # Up to 50 epochs
                model.train()
                running_loss = 0.0
                batch_count = 0
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        x = batch["input_values"].to(device)
                        y = batch["label"].to(device)
                    except Exception as e:
                        print(f"[ERROR] Batch {batch_idx} data loading error: {e}")
                        continue
                    try:
                        optimizer.zero_grad()
                        if train_dataset.has_speaker and "speaker_id" in batch:
                            logits = model(x, batch["speaker_id"].to(device))
                        else:
                            logits = model(x)
                        loss = F.cross_entropy(logits, y)
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"[ERROR] Batch {batch_idx} training error: {e}")
                        continue
                    running_loss += loss.item()
                    batch_count += 1
                avg_loss = running_loss / batch_count if batch_count > 0 else float('inf')
                print(f"Epoch {epoch} train loss: {avg_loss:.4f}")
                # Validation
                model.eval()
                preds, targets = [], []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        try:
                            x = batch["input_values"].to(device)
                            y = batch["label"].to(device)
                            if val_dataset.has_speaker and "speaker_id" in batch:
                                logits = model(x, batch["speaker_id"].to(device))
                            else:
                                logits = model(x)
                            pred = logits.argmax(dim=1)
                            preds.extend(pred.cpu().tolist())
                            targets.extend(y.cpu().tolist())
                        except Exception as e:
                            print(f"[ERROR] Validation batch {batch_idx} skipped due to error: {e}")
                if len(targets) == 0:
                    print("[WARNING] No validation samples evaluated this epoch.")
                    continue
                acc = accuracy_score(targets, preds)
                f1 = f1_score(targets, preds, average="weighted")
                print(f"Epoch {epoch}: val accuracy={acc:.4f}, weighted F1={f1:.4f}")
                writer.writerow([epoch, avg_loss, acc, f1, optimizer.param_groups[0]["lr"]])
                csvfile.flush()
                scheduler.step(acc)
                model_save_path = os.path.join(project_root, "models", "best_model.pth")
                if acc > best_val_acc:
                    best_val_acc = acc
                    patience_counter = 0
                    torch.save(model.state_dict(), model_save_path)
                    print(f"Saved best model at epoch {epoch} with accuracy {acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"No improvement. Early stopping patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("Early stopping triggered.")
                        break
            # --- After training: Confusion Matrix ---
            print("\nComputing confusion matrix on validation set...")
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["input_values"].to(device)
                    y = batch["label"].to(device)
                    if val_dataset.has_speaker and "speaker_id" in batch:
                        logits = model(x, batch["speaker_id"].to(device))
                    else:
                        logits = model(x)
                    pred = logits.argmax(dim=1)
                    all_preds.extend(pred.cpu().tolist())
                    all_targets.extend(y.cpu().tolist())
            cm = confusion_matrix(all_targets, all_preds)
            np.savetxt(os.path.join(project_root, "logs", "confusion_matrix.csv"), cm, delimiter=",", fmt="%d")
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            phrase_labels = list(val_dataset.phrase2label.keys())
            tick_marks = np.arange(len(phrase_labels))
            plt.xticks(tick_marks, phrase_labels, rotation=45, ha="right", fontsize=8)
            plt.yticks(tick_marks, phrase_labels, fontsize=8)
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(os.path.join(project_root, "logs", "confusion_matrix.png"), bbox_inches="tight")
            print("Confusion matrix saved to logs/confusion_matrix.png and logs/confusion_matrix.csv")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

if __name__ == "__main__":
    train()
