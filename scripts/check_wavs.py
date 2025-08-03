import soundfile as sf
from pathlib import Path
import torchaudio
# Manifest consistency check
import json
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
manifest_paths = [
    os.path.join(project_root, "manifests", "train_manifest.json"),
    os.path.join(project_root, "manifests", "val_manifest.json")
]
missing_files = []

bad_files = []

dataset_root = Path("dataset")
for wav_path in dataset_root.glob('**/*.wav'):
    try:
        sf.read(str(wav_path))
        # Try loading with torchaudio as well
        torchaudio.load(str(wav_path))
    except Exception as e:
        print(f"Bad file: {wav_path} -- {e}")
        bad_files.append(wav_path)

print(f"Total bad files found: {len(bad_files)}")
if bad_files:
    print("Consider deleting or re-recording these files.")
    print("BAD_FILES = [")
    for f in bad_files:
        print(f'    "{f.as_posix()}",')
    print("]")

# Manifest consistency check
for manifest_path in manifest_paths:
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        for entry in manifest.values():
            audio_path = entry["audio_path"].replace("\\", "/")
            if not Path(audio_path).exists():
                print(f"Missing file: {audio_path} (referenced in {manifest_path})")
                missing_files.append(audio_path)
    except Exception as e:
        print(f"[ERROR] Could not check manifest {manifest_path}: {e}")
if missing_files:
    print("MISSING_FILES = [")
    for f in missing_files:
        print(f'    "{f}",')
    print("]")
else:
    print("All manifest files exist on disk.")

def clean_manifest(manifest_path, missing_files):
    import json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    new_manifest = {}
    for k, v in manifest.items():
        audio_path = v["audio_path"].replace("\\", "/")
        if audio_path not in missing_files:
            new_manifest[k] = v
    with open(manifest_path, "w") as f:
        json.dump(new_manifest, f, indent=2)
    print(f"Cleaned manifest: {manifest_path} (removed {len(manifest)-len(new_manifest)} entries)")

if missing_files:
    for manifest_path in manifest_paths:
        clean_manifest(manifest_path, missing_files)
