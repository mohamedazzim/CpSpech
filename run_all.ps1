# 1. Preprocess audio
python data_processing/preprocess.py --dataset_path dataset

# 2. Create train/val manifests
python data_processing/split_manifest.py --dataset_root dataset --out_dir manifests

# 3. Train the model
python -m model.train

# 4. Example: Run inference on a sample file (replace with your own .wav)
# python -m inference.real_time dataset/Amaravathi/amaravathi_1.wav models/best_model.pth manifests/train_manifest.json

# 5. Example: Live inference from microphone (uncomment to use)
# python -m inference.live_inference --model_ckpt models/best_model.pth --manifest_json manifests/train_manifest.json --speak --repeat 1