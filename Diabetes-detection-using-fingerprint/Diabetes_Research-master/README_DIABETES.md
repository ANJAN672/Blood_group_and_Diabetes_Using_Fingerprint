Diabetes detection from fingerprints — guide

Files added:
- `train_diabetes.py` — training script. Supports `--dry-run` for synthetic data or real dataset via `DIABETES_DATA_PATH` env var or `--data-dir`.
- `predict_diabetes.py` — inference script to predict diabetes probability for single images or a directory.
- `train_smoke.py` — earlier smoke-test that loads dataset if present or uses synthetic data.

How to provide your dataset:
1. Copy your fingerprint images into `f:\diabetes\Diabetes_Research-master\dataset`.
   - Preferred structure (recommended):
       dataset/\
         diabetic/\
           img1.png
           img2.png
         non_diabetic/\
           imgA.png
           imgB.png
   - If your dataset is a flat folder of images, the script will treat it as a single class.

2. Train locally (example, PowerShell):
   1) Install minimal requirements:
      .venv\Scripts\python.exe -m pip install -r .\Diabetes_Research-master\requirements-min.txt
   2) Run training:
      .venv\Scripts\python.exe .\Diabetes_Research-master\train_diabetes.py --epochs 5 --batch-size 16

3. Predict for new fingerprints:
   .venv\Scripts\python.exe .\Diabetes_Research-master\predict_diabetes.py --image path\to\fp.png --model models\diabetes_model.h5

Notes:
- MobileNetV2 is used as a lightweight feature extractor to get reasonable results quickly. For serious experiments you may want to fine-tune the base model or use a larger backbone and GPU.
- If your dataset class mapping differs from `diabetic`/`non_diabetic`, pass the folder with subfolders named for each class; the training script infers class names.
- I can update the original notebooks so they use the same loader and the trained model; say the word and I'll patch them.
