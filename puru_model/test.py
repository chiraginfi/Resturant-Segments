import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

from config import Paths, FeatureConfig
from feature_engineering import engineer

# -----------------------------------------------------------------------------
# Load manifest (single source of truth)
# -----------------------------------------------------------------------------
ARTIFACT_DIR = Path(Paths.OUTPUT_DIR) / "artifacts"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

BEST_MODEL_NAME = manifest["best_model_name"]
MODEL_PATHS = manifest["models"]

# -----------------------------------------------------------------------------
# Load common artifacts
# -----------------------------------------------------------------------------
feature_cols = joblib.load(manifest["feature_cols_path"])
imputer = joblib.load(manifest["imputer_path"])
le = joblib.load(manifest["label_encoder_path"])

USER_REJECT_COLUMNS = list(getattr(FeatureConfig, "USER_REJECT_COLUMNS", []))

# -----------------------------------------------------------------------------
# Load model bundle
# -----------------------------------------------------------------------------
def load_model_bundle(model_name):
    mapping = {
        "GBM (tuned)": "gbm_tuned",
        "RF (tuned)": "rf_tuned",
        "Extra Trees": "extra_trees",
        "k-NN (tuned)": "knn_tuned",
        "Semi-Supervised": "semi_supervised_gbm",
        "Stacking": "stacking",
        "Majority Vote": "majority_vote",
    }
    key = mapping[model_name]
    return joblib.load(MODEL_PATHS[key])

model_bundle = load_model_bundle(BEST_MODEL_NAME)

# -----------------------------------------------------------------------------
# Prediction logic per model type
# -----------------------------------------------------------------------------
def predict_with_model(bundle, X):
    name = bundle["model_name"]
    print("Running model:", name)
    if name in ["GBM (tuned)", "RF (tuned)", "Extra Trees", "k-NN (tuned)", "Semi-Supervised"]:
        model = bundle["model"]
        preds = model.predict(X)
        proba = model.predict_proba(X)

    elif name == "Stacking":
        base_models = bundle["base_models"]
        meta_model = bundle["meta_model"]

        nc = len(meta_model.classes_)
        meta = np.zeros((len(X), len(base_models) * nc))
        votes = []

        for i, (nm, m, _) in enumerate(base_models):
            p = m.predict_proba(X)
            meta[:, i * nc:(i + 1) * nc] = p
            votes.append(m.predict(X))

        meta_votes = np.column_stack(votes)
        meta_all = np.column_stack([meta, meta_votes])

        preds = meta_model.predict(meta_all)
        proba = meta_model.predict_proba(meta_all)

    elif name == "Majority Vote":
        members = bundle["members"]

        preds_list = []
        proba_list = []

        for mname, m in members.items():
            if mname == "stacking":
                # recursive call
                p, _ = predict_with_model(m, X)
            else:
                p = m.predict(X)
            preds_list.append(p)

            if hasattr(m, "predict_proba"):
                proba_list.append(m.predict_proba(X))

        pred_matrix = np.column_stack(preds_list)
        preds = stats.mode(pred_matrix, axis=1)[0].ravel()

        if proba_list:
            proba = np.mean(proba_list, axis=0)
        else:
            proba = None

    else:
        raise ValueError(f"Unknown model type: {name}")

    return preds, proba


# -----------------------------------------------------------------------------
# Load input (can be test.csv OR new data)
# -----------------------------------------------------------------------------
def load_input(path=None):
    if path:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(manifest["test_split_csv"])
    return df


# -----------------------------------------------------------------------------
# Main predict function
# -----------------------------------------------------------------------------
def run_prediction(input_path=None, output_path=None):
    df = load_input(input_path)

    # Drop rejected columns
    df = df.drop(columns=[c for c in USER_REJECT_COLUMNS if c in df.columns], errors="ignore")

    # Feature engineering
    df_eng, *_ = engineer(df.copy())

    # Align features
    valid_cols = [f for f in feature_cols if f in df_eng.columns]
    X = imputer.transform(df_eng[valid_cols].astype(float))

    # Predict
    preds, proba = predict_with_model(model_bundle, X)

    df["prediction"] = le.inverse_transform(preds)

    if proba is not None:
        df["confidence"] = proba.max(axis=1)
        epsilon = 1e-10
        if proba.shape[1] >= 2:
            sorted_proba = np.sort(proba, axis=1)[:, ::-1]
            df["recall_proxy"] = sorted_proba[:, 0] / (sorted_proba[:, 0] + sorted_proba[:, 1] + epsilon)
    else:
        df["confidence"] = None
        df["recall_proxy"] = None
    # Save
    if output_path is None:
        output_path = Path(Paths.OUTPUT_DIR) / "predictions.csv"

    df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved → {output_path}")
    print(df["prediction"].value_counts())

    return df


# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_prediction(input_path="/mnt/data/image_recognition/brown_forman_req/puru_input/training_data_bf_22_may.csv", output_path="/mnt/data/image_recognition/brown_forman_req/puru_output/predictions_full.csv")  # default → test split