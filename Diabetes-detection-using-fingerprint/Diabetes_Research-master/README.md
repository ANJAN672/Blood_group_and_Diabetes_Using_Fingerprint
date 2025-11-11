# Diabetes detection from fingerprint images

This repository contains a demo pipeline that explores diabetes risk prediction from fingerprint images. It includes a CNN classifier (MobileNetV2), handcrafted feature extraction, a Mamdani fuzzy inference prototype, explainability (Grad‑CAM) and utilities to build membership functions, plots and a small Streamlit UI.

Use this README as the single place to find how to run training, inference, feature extraction, fuzzy inference, and to reproduce the plots used in presentations.

## What is in this project (short)
- A CNN classifier (MobileNetV2-based) trained to predict diabetic vs non-diabetic from fingerprint image crops. Outputs a probability (0..1).
- Grad‑CAM explanation utility to visualize salient image regions affecting the CNN output.
- Handcrafted feature extractor that computes interpretable features per image (edge_density, ridge_density, mean_intensity, entropy, gradient_mean, etc.).
- A Mamdani fuzzy inference prototype that maps features → linguistic labels (not_diabetic, not_likely, likely, diabetic, highly_diabetic) and produces a fuzzy score.
- Evaluation scripts and plotting utilities to compare CNN probabilities vs fuzzy outputs, and to build membership function plots used in slides.

## Quick setup (Windows PowerShell)
1. Create and activate a virtual environment (recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Install common packages used by the scripts (adjust versions as needed):
```powershell
pip install tensorflow matplotlib scikit-learn pillow opencv-python scikit-image pandas streamlit fastapi uvicorn
```
Note: Some scripts use only the Python standard library + Pillow/matplotlib. If you don't need TF or Streamlit, skip those installs.

## Important files and scripts
- `train_diabetes.py` — training pipeline (MobileNetV2 backbone). Use to train the CNN.
- `predict_diabetes.py` — CLI for single-image prediction using the saved model.
- `streamlit_predict.py` — Streamlit web UI for quick single-image prediction.
- `evaluate.py` — evaluate model on a labeled folder and write `predictions_eval.csv`, ROC and calibration plots.
- `explain.py` — Grad‑CAM explainer that saves overlay heatmaps.
- `extract_features.py` — compute interpretable image features and write `features.csv`.
- `compute_feature_mf_params.py` — compute trapezoidal MF params from `features.csv` (percentile-driven) and save JSON.
- `fuzzy_inference.py` — prototype Mamdani fuzzy inference that reads feature CSV and produces `fuzzy_outputs.csv`.
- `plot_membership_from_stats.py`, `plot_membership.py`, `generate_all_membership.py` — build and save membership plots.
- `apply_simple_thresholds.py` — apply fixed numeric cutoffs to CNN probabilities and write `predictions_thresholded.csv`.
- `create_membership_slide.py` — compose the membership plot and textual thresholds into `membership_slide.png` for presentation.

Generated artifacts added in this demo
- `features.csv` — extracted features for the demo images.
- `models/diabetes_model.h5` — trained Keras model (if created by you). Note: not committed by default unless present.
- Various plots and PNGs: `membership_*.png`, `membership_slide.png`, `fuzzy_vs_neural_plots.png`, `roc_curve.png`, `calibration.png`.

## How to run the main flows

1) Feature extraction (build `features.csv`) — run on a folder of labeled images:
```powershell
python extract_features.py --data-dir dataset_labeled --out features.csv
```

2) Train CNN (example quick run):
```powershell
python train_diabetes.py --data-dir dataset_labeled --epochs 10 --batch-size 8
```
This saves a model under `models/` (see script output).

3) Evaluate model:
```powershell
python evaluate.py --data-dir dataset_labeled --model models/diabetes_model.h5 --out predictions_eval.csv
```

4) Apply simple threshold rules to CNN probabilities (presentation style):
```powershell
python apply_simple_thresholds.py --in "predictions_eval.csv"
```
This writes `predictions_thresholded.csv` with `threshold_label` column.

5) Compute MF params from data (percentiles) and save JSON:
```powershell
python compute_feature_mf_params.py
```

6) Run the fuzzy prototype (reads `features.csv`) and produce `fuzzy_outputs.csv`:
```powershell
python fuzzy_inference.py
```

7) Create membership plots and slide (presentation):
```powershell
python plot_membership_from_stats.py
python create_membership_slide.py
```

8) Run Streamlit UI for single image demo:
```powershell
streamlit run streamlit_predict.py
```

## Fuzzy system notes
- We used trapezoidal membership functions for the five linguistic labels. The prototype scripts compute MF parameters from data percentiles or use fixed thresholds for presentation (e.g., 0.40,0.60,0.70,0.85).
- The fuzzy inference engine uses Mamdani-style rules, min for AND/implication, max for aggregation and centroid defuzzification.

## Models used (one-line each) — for slides
- MobileNetV2 CNN — fingerprint → diabetes probability.  
- Grad‑CAM — saliency overlays for model explanation.  
- Handcrafted feature extractor — edge/ridge/mean intensity, entropy, gradient mean.  
- Mamdani fuzzy inference — feature → linguistic risk labels.  
- Evaluation utilities — ROC/AUC, calibration, threshold mapping.

## Caution & validation
- The demo labels and short training runs in this workspace are for demonstration only. Do NOT use the outputs for clinical decisions. Any clinical use requires proper validated labels, stronger cross-validation, calibration, and regulatory review.

## Contributing / pushing updates
- This repo already contains a Git remote (`origin`) pointing to your GitHub. To push local commits to GitHub use:
```powershell
git -C "F:/diabetes/Diabetes_Research-master" push origin main
```
Use SSH or a GitHub personal access token if prompted.

## License & contact
Check the `LICENSE` file in the repo root. For questions about the code or results, open an issue or contact the project owner.

---
If you want, I can also:
- add a `requirements.txt` file with exact package versions; or
- create a short one-slide PNG summarizing the models + 3 formulas you requested; or
- commit this `README.md` into the repo for you to push. Tell me which next.
# Identifying diabetes from fingerprints.

Our objective was to test if the fingerprints excluding the thumb could carries any information regarding the probabilty of diabetes, either type 1 and type 2, to the person being tested. We have used Deep Convolutional Neural Networks and obtained performance metric of prediction on 50 cross validations which revealed the fact that in the 4 finger analyzed there is infact a great deal of embedded information regarding diabetes.<br>
<br>

## Data<br>
The data for this research was collected from hospitals in Romania, by  Nicoleta Dragana, C. Vulpe, L. Guja.<br>
The data is divided into three parts, the first part consists of fingerprints from Non-diabetic patients, the second part consists of fingerprints from Type-1 diabetic patients, and the last part consits of data from Type-2 diabetic patients.

## Artificial Neural Network <br>
In order to find the difference in patterns of fingerprints for Diabetic and Non-diabetic patients, we designed a Convonutional Neural Network and a Residual Neural Network. Out of these two Convonutional Neural Network worked the best.<br>
The code for Convonutional Neural Network is [here](https://www.github.com/Sid2697/Diabetes_Research/blob/master/Type1VersusHealty.ipynb), and the code for Residual Neural Network is [here](https://github.com/Sid2697/Diabetes_Research/blob/master/ResNet%20Type1%20Vs.%20Healthy%20.ipynb).

## Publication
The results of successfully identifying Diabetic patients were published [here](https://www.researchgate.net/publication/324219743_Unexpected_Results_Embedded_Information_in_Fingerprints_Regarding_Diabetes) with @alexandrudaia.
<br>
Feel free to contribute to the code for increasing the accuracy.

## License

*I'm providing the codes in this repository to you under an open source license*

MIT License

Copyright (c) 2018 Siddhant Bansal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

