# Churn Prediction

This repository contains a compact end-to-end churn prediction project: data, training pipeline, saved model artifacts, and a small Flask-based web app for serving predictions.

It is organized to be MLOps-friendly — models and metadata are stored under `mlartifacts/` (MLflow-style layout), training code lives in `model_pipeline.py`, and a simple demo web UI is available via `app.py` (root) and the `flask/` subfolder.

---

**Quick links**

- **Project root app**: `app.py`
- **Alternate Flask app & templates**: `flask/app.py`, `flask/templates/`
- **Training pipeline**: `model_pipeline.py`
- **Sample dataset**: `merged_churn1.csv`
- **Saved model artifacts**: `mlartifacts/`

**Why this repo**

This repo is useful as a learning/example project demonstrating how to train a churn model, persist artifacts for reproducibility, and run a simple web service that performs predictions.

**Audience**: data scientists, MLOps engineers, and developers who want a small end-to-end churn-demo to extend or deploy.

**Table of contents**

- **Project Overview** — high-level description
- **Setup** — dependencies and how to run locally
- **Usage** — how to run the web app and make predictions
- **Model artifacts** — where models live and how to load them
- **Development** — training, testing, and extending the project
- **Contributing & Next steps**

---

**Project Overview**

- **Purpose**: Predict whether a customer will churn using historical features.
- **Modeling**: The repo contains a training pipeline (`model_pipeline.py`) that prepares data and fits a model (for example, a Random Forest). Trained models and environment metadata are stored under `mlartifacts/`.
- **Serving**: A small Flask app demonstrates how to collect inputs and return churn probabilities.

**Repository Structure**

- **`app.py`**: Root-level example web app / entry script for quick demos.
- **`main.py`**: Auxiliary entrypoint (check contents to determine role in your workflow).
- **`model_pipeline.py`**: Training and evaluation pipeline — preprocesses the data and trains the model.
- **`merged_churn1.csv`**: Example dataset used in development and experiments.
- **`requirements.txt`**: Python dependencies for the project / environment.
- **`flask/`**: Alternative Flask app and `templates/` used by the web UI.
- **`mlartifacts/`**: Saved model runs/artifacts (MLflow-style) with multiple runs inside; each run contains an `artifacts/model/` folder.

---

**Setup (local, Windows PowerShell)**

Note: this repository was developed for Python 3.8+ — adjust virtual environment and commands to your platform and preferred Python version.

- **Install dependencies**: Use a virtual environment and install packages from `requirements.txt`.

```powershell
# create and activate virtual environment (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

- **Notes about dependencies**: Some model artifacts include their own `requirements.txt` inside `mlartifacts/*/artifacts/model/` — use those to reproduce the exact training/serving environment for a given run.

---

**Run the demo web app**

There are two small web app entrypoints in this repo. Choose one to try.

- **Root demo app** (quick test):

```powershell
# from repo root
python app.py
```

- **Flask app with templates** (if present / preferred):

```powershell
# from repo root
python ./flask/app.py

# or, if the project uses flask CLI, set FLASK_APP and run
$env:FLASK_APP = 'flask.app'; flask run
```

- **Access**: The app will normally be available at `http://127.0.0.1:5000` unless configured otherwise. Check console output for exact host/port.

**Making predictions**

- The demo app accepts customer feature values and returns a churn probability and/or label. Use the web form in the browser, or call the HTTP endpoint (if the app exposes one) with JSON payloads.

---

**Model artifacts & reproducibility**

- **Location**: All saved model runs are under `mlartifacts/`. Each run folder contains an `artifacts/model/` directory that typically includes `MLmodel`, `requirements.txt`, and (optionally) `input_example.json` and `serving_input_example.json`.
- **How to load a model**: If artifacts are MLflow-style, you can load them with MLflow's `pyfunc` API; otherwise inspect `MLmodel` and the included files to determine how to load the model (e.g., `joblib`, `pickle`, or `mlflow.pyfunc`). Example using MLflow:

```python
import mlflow.pyfunc

# path to an artifacts model folder (example)
model_path = 'mlartifacts/1/3300fa6de8b54be1a357ca7bd9cb73fe/artifacts/model'
model = mlflow.pyfunc.load_model(model_path)

# predict on a pandas DataFrame
preds = model.predict(df)
```

- **Environment**: To reproduce the exact environment for a model run, inspect `mlartifacts/<run-id>/artifacts/model/requirements.txt` or `conda.yaml`.

---

**Data**

- **`merged_churn1.csv`**: Example dataset used by the project. It contains features and the churn target — examine it with pandas before running training.

Example quick inspect:

```powershell
python -c "import pandas as pd; df = pd.read_csv('merged_churn1.csv'); print(df.shape); print(df.head())"
```

---

**Training & development**

- **Training script**: `model_pipeline.py` contains the data preparation, feature engineering, training, and evaluation logic. Open it to learn which features were used and how preprocessing is applied.
- **Re-train**: To re-run training, ensure dependencies are installed and run the pipeline script. Consider creating a dedicated run directory or using MLflow to track experiments.

---

**Troubleshooting**

- **Model load errors**: If loading a saved model fails due to missing packages, check `mlartifacts/*/artifacts/model/requirements.txt` or `conda.yaml` and install the required packages into your environment.
- **Port in use**: If `5000` is occupied, set `FLASK_RUN_PORT` or pass a different port to `app.run()`.

---

**Contributing & Next steps**

- **Extend**: Improve the model (new features, hyperparameter tuning), add CI, or create a Dockerfile for containerized serving.
- **Add tests**: Add unit tests for data processing and model inference.
- **Packaging**: Provide a `Makefile`, Dockerfile, or CI pipeline to reproduce training and serving steps automatically.

---

**License**

- This project does not include an explicit license file. Add a `LICENSE` with your preferred license to clarify reuse terms.

---

If you'd like, I can also:

- add a small `docker-compose` or `Dockerfile` for containerized serving,
- create a short example script that loads a model from `mlartifacts/` and runs a demo prediction,
- or run basic static checks on `model_pipeline.py` to extract exact dependency hints.

If you want any of those next steps, tell me which and I'll proceed.
