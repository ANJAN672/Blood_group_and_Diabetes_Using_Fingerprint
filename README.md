# Fingerprint Screening Suite

Unified **Streamlit** experience for fingerprint-based **blood group detection** and **diabetes risk screening** powered by pre-trained **TensorFlow/Keras** models.

## ✨ Features
- Single Streamlit UI with tabs for both blood group classification and diabetes probability estimation.
- Fingerprint preprocessing optimized for existing CNN models.
- Works with local virtual environments managed by [uv](https://docs.astral.sh/uv/).
- Docker image based on the official `uv` runtime for reproducible deployments.

## 📦 Prerequisites
- Python 3.10 (added to `PATH`)
- [uv CLI](https://docs.astral.sh/uv/getting-started/installation/) for dependency management
- Git (if cloning)
- Optional: Docker

## 🚀 Quickstart with uv
```powershell
git clone https://github.com/yourusername/blood_group_detection_using_fingerprint_cnn.git
cd blood_group_detection_using_fingerprint_cnn

# Create/refresh the project virtual environment in .venv
uv sync

# Launch the unified Streamlit app
uv run streamlit run app.py
```

By default `uv sync` provisions a `.venv/` alongside the project and installs the dependencies defined in `pyproject.toml`.

### Diabetes model checkpoint
Place `diabetes_model.h5` (and optional `class_names.json`) inside  
`Diabetes-detection-using-fingerprint/Diabetes_Research-master/models/`.  
If the file is missing the diabetes tab will show a warning.

## 🐳 Docker Workflow
```sh
docker build -t fingerprint-screening .
docker run -p 8501:8501 fingerprint-screening
```
Visit `http://localhost:8501` in your browser.

## 📂 Project Layout
```
├── app.py
├── pyproject.toml
├── requirements.txt              # legacy fallback
├── Dockerfile
├── Model/
│   ├── keras_model.h5
│   └── labels.txt
├── Diabetes-detection-using-fingerprint/
│   └── Diabetes_Research-master/
│       └── models/
│           └── diabetes_model.h5 (provide your trained checkpoint)
└── …
```

## 🛠 Troubleshooting
- Run `uv pip check` if dependency resolution fails.
- Delete `.venv/` and rerun `uv sync` to refresh the environment.
- Use `UV_LINK_MODE=copy uv sync` on Windows if symlinks are restricted.
- If you must use pip, `pip install -r requirements.txt` remains available but is not the preferred workflow.

## 📜 License
This project is **open-source**. Contributions are welcome! 🎉
