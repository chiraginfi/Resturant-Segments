# training.py
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config import Paths, data_cfg, split_cfg, cv_cfg, model_cfg, ss_cfg, analysis_cfg, FeatureConfig
from label_corrections import apply_label_corrections
from feature_engineering import (
    engineer,
    select_features,
    build_feature_matrix,
    run_correlation_analysis,
    run_distribution_analysis,
    CORE_FEATURES,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
Paths.ensure_output_dir()
OUTPUT_DIR = Path(Paths.OUTPUT_DIR)
ARTIFACT_DIR = OUTPUT_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
SPLIT_DIR = ARTIFACT_DIR / "splits"
METRIC_DIR = ARTIFACT_DIR / "metrics"

for p in [ARTIFACT_DIR, MODEL_DIR, SPLIT_DIR, METRIC_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT_CSV = SPLIT_DIR / "train_split.csv"
TEST_SPLIT_CSV = SPLIT_DIR / "test_split.csv"
METRICS_CSV = METRIC_DIR / "model_metrics.csv"
METRICS_JSON = METRIC_DIR / "model_metrics.json"
BEST_MODEL_JSON = METRIC_DIR / "best_model.json"

FEATURE_COLS_PKL = ARTIFACT_DIR / "feature_cols.pkl"
IMPUTER_PKL = ARTIFACT_DIR / "imputer.pkl"
LABEL_ENCODER_PKL = ARTIFACT_DIR / "label_encoder.pkl"

MANIFEST_PKL = ARTIFACT_DIR / "manifest.pkl"
MANIFEST_JSON = ARTIFACT_DIR / "manifest.json"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
USER_REJECT_COLUMNS = list(getattr(FeatureConfig, "USER_REJECT_COLUMNS", []))


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df.drop(columns=cols)
    return df


def save_json(obj, path: Path):
    def _default(x):
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        return str(x)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)


def class_metrics(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    out = {"accuracy": accuracy_score(y_true, y_pred)}

    for cls in le.classes_:
        idx = list(le.classes_).index(cls)
        tp = cm[idx, idx]
        col_sum = cm[:, idx].sum()
        row_sum = cm[idx, :].sum()

        precision = tp / col_sum if col_sum > 0 else 0.0
        recall = tp / row_sum if row_sum > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        out[f"{cls}_precision"] = precision
        out[f"{cls}_recall"] = recall
        out[f"{cls}_f1"] = f1

    return out


def save_model_bundle(name: str, bundle: dict):
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(bundle, path)
    return path


def print_results_table(rows: list[dict], le, title: str):
    print(f"\n{'=' * 110}")
    print(title)
    print(f"{'=' * 110}")
    header = (
        f"{'Model':<22s} {'CV Acc':>8s} {'Test Acc':>9s} │ "
        f"{'Mainstream F1':>13s} {'Premium F1':>11s} {'Super Premium F1':>17s}"
    )
    print(header)
    print("─" * 110)

    for r in rows:
        print(
            f"{r['model']:<22s} "
            f"{r.get('cv_acc', np.nan):>7.1%} "
            f"{r.get('test_acc', np.nan):>8.1%} │ "
            f"{r.get('Mainstream_f1', 0.0):>12.1%} "
            f"{r.get('Premium_f1', 0.0):>10.1%} "
            f"{r.get('Super Premium_f1', 0.0):>16.1%}"
        )


def tune_gbm(X_train, y_train, skf):
    g = model_cfg.GBM_GRID
    fixed = model_cfg.GBM_FIXED

    best = {
        "cv_acc": -1.0,
        "params": None,
        "oof_pred": None,
    }

    for n_est in g["n_estimators"]:
        for max_d in g["max_depth"]:
            for lr in g["learning_rate"]:
                for ss in g["subsample"]:
                    preds = np.zeros_like(y_train)
                    for tri, tei in skf.split(X_train, y_train):
                        sw = compute_sample_weight("balanced", y_train[tri])
                        m = GradientBoostingClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            learning_rate=lr,
                            subsample=ss,
                            random_state=model_cfg.RANDOM_STATE,
                            **fixed,
                        )
                        m.fit(X_train[tri], y_train[tri], sample_weight=sw)
                        preds[tei] = m.predict(X_train[tei])

                    acc = accuracy_score(y_train, preds)
                    if acc > best["cv_acc"]:
                        best.update(
                            {
                                "cv_acc": acc,
                                "params": {
                                    "n_estimators": n_est,
                                    "max_depth": max_d,
                                    "learning_rate": lr,
                                    "subsample": ss,
                                },
                                "oof_pred": preds.copy(),
                            }
                        )

    return best


def tune_rf(X_train, y_train, skf):
    g = model_cfg.RF_GRID
    fixed = model_cfg.RF_FIXED

    best = {
        "cv_acc": -1.0,
        "params": None,
        "oof_pred": None,
    }

    for n_est in g["n_estimators"]:
        for max_d in g["max_depth"]:
            for msl in g["min_samples_leaf"]:
                preds = np.zeros_like(y_train)
                for tri, tei in skf.split(X_train, y_train):
                    m = RandomForestClassifier(
                        n_estimators=n_est,
                        max_depth=max_d,
                        min_samples_leaf=msl,
                        random_state=model_cfg.RANDOM_STATE,
                        **fixed,
                    )
                    m.fit(X_train[tri], y_train[tri])
                    preds[tei] = m.predict(X_train[tei])

                acc = accuracy_score(y_train, preds)
                if acc > best["cv_acc"]:
                    best.update(
                        {
                            "cv_acc": acc,
                            "params": {
                                "n_estimators": n_est,
                                "max_depth": max_d,
                                "min_samples_leaf": msl,
                            },
                            "oof_pred": preds.copy(),
                        }
                    )

    return best


def tune_knn(X_train, y_train, skf):
    g = model_cfg.KNN_GRID
    fixed = model_cfg.KNN_FIXED

    best = {
        "cv_acc": -1.0,
        "params": None,
        "oof_pred": None,
    }

    for n in g["n_neighbors"]:
        for w in g["weights"]:
            for metric in g["metric"]:
                preds = np.zeros_like(y_train)
                for tri, tei in skf.split(X_train, y_train):
                    m = KNeighborsClassifier(
                        n_neighbors=n,
                        weights=w,
                        metric=metric,
                        **fixed,
                    )
                    m.fit(X_train[tri], y_train[tri])
                    preds[tei] = m.predict(X_train[tei])

                acc = accuracy_score(y_train, preds)
                if acc > best["cv_acc"]:
                    best.update(
                        {
                            "cv_acc": acc,
                            "params": {
                                "n_neighbors": n,
                                "weights": w,
                                "metric": metric,
                            },
                            "oof_pred": preds.copy(),
                        }
                    )

    return best


def fit_stacking_bundle(X_train, y_train, X_unlab, best_gbm_params, best_knn_params, skf):
    # Base models used in the original script
    models_stack = [
        ("GBM1", GradientBoostingClassifier(
            n_estimators=best_gbm_params["n_estimators"],
            max_depth=best_gbm_params["max_depth"],
            learning_rate=best_gbm_params["learning_rate"],
            subsample=best_gbm_params["subsample"],
            random_state=model_cfg.RANDOM_STATE,
            **model_cfg.GBM_FIXED
        ), True),
        ("GBM2", GradientBoostingClassifier(
            **model_cfg.STACK_GBM2
        ), True),
        ("GBM3", GradientBoostingClassifier(
            **model_cfg.STACK_GBM3
        ), True),
        ("RF", RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=3,
            random_state=model_cfg.RANDOM_STATE,
            **model_cfg.RF_FIXED
        ), False),
        ("ET", ExtraTreesClassifier(
            random_state=model_cfg.RANDOM_STATE,
            **model_cfg.ET_PARAMS
        ), False),
        ("KNN", KNeighborsClassifier(
            **best_knn_params,
            **model_cfg.KNN_FIXED
        ), False),
    ]

    nc = len(np.unique(y_train))
    meta = np.zeros((len(y_train), len(models_stack) * nc))
    fitted_base_models = []

    for mi, (nm, mod, use_sw) in enumerate(models_stack):
        for tri, tei in skf.split(X_train, y_train):
            mm = type(mod)(**mod.get_params())
            if use_sw:
                sw = compute_sample_weight("balanced", y_train[tri])
                mm.fit(X_train[tri], y_train[tri], sample_weight=sw)
            else:
                mm.fit(X_train[tri], y_train[tri])
            meta[tei, mi * nc:(mi + 1) * nc] = mm.predict_proba(X_train[tei])

    # Use the same vote-style meta features from the original script
    base_oof_preds = []
    for nm, mod, use_sw in models_stack:
        preds = np.zeros_like(y_train)
        for tri, tei in skf.split(X_train, y_train):
            mm = type(mod)(**mod.get_params())
            if use_sw:
                sw = compute_sample_weight("balanced", y_train[tri])
                mm.fit(X_train[tri], y_train[tri], sample_weight=sw)
            else:
                mm.fit(X_train[tri], y_train[tri])
            preds[tei] = mm.predict(X_train[tei])
        base_oof_preds.append(preds)

    meta_votes = np.column_stack(base_oof_preds)
    meta_all = np.column_stack([meta, meta_votes])

    lr_meta = LogisticRegression(
        **model_cfg.META_LR,
        random_state=model_cfg.RANDOM_STATE,
    )
    preds_stack = cross_val_predict(lr_meta, meta_all, y_train, cv=skf)
    stack_cv_acc = accuracy_score(y_train, preds_stack)

    # Fit base models on full train for saving
    fitted_base_models = []
    for nm, mod, use_sw in models_stack:
        mm = type(mod)(**mod.get_params())
        if use_sw:
            sw = compute_sample_weight("balanced", y_train)
            mm.fit(X_train, y_train, sample_weight=sw)
        else:
            mm.fit(X_train, y_train)
        fitted_base_models.append((nm, mm, use_sw))

    # Fit final meta model on training meta-features built from CV OOF outputs
    lr_meta.fit(meta_all, y_train)

    return {
        "cv_acc": stack_cv_acc,
        "base_models": fitted_base_models,
        "meta_model": lr_meta,
        "meta_all_train": meta_all,
    }


def predict_stacking(bundle: dict, X):
    base_models = bundle["base_models"]
    meta_model = bundle["meta_model"]

    nc = len(meta_model.classes_)
    meta = np.zeros((len(X), len(base_models) * nc))
    votes = []

    for mi, (nm, mod, use_sw) in enumerate(base_models):
        proba = mod.predict_proba(X)
        meta[:, mi * nc:(mi + 1) * nc] = proba
        votes.append(mod.predict(X))

    meta_votes = np.column_stack(votes)
    meta_all = np.column_stack([meta, meta_votes])
    return meta_model.predict(meta_all)


# -----------------------------------------------------------------------------
# 1) Load data
# -----------------------------------------------------------------------------
print("=" * 100)
print("RESTAURANT CLASSIFIER TRAINING")
print("=" * 100)

df = pd.read_csv(str(Paths.INPUT_CSV), low_memory=False)

# Drop leakage columns from config
df = drop_columns(df, list(getattr(data_cfg, "LEAKAGE_COLUMNS", [])))

# Drop user rejected columns
df = drop_columns(df, USER_REJECT_COLUMNS)

df["Category"] = df["Category"].astype(str).str.strip().str.title()

labeled_raw = df[df["Category"].isin(data_cfg.VALID_CATEGORIES)].copy()
unlabeled = df[~df["Category"].isin(data_cfg.VALID_CATEGORIES)].copy()

print(f"Full labeled: {len(labeled_raw)} | Unlabeled: {len(unlabeled)}")
print(f"Distribution: {dict(labeled_raw['Category'].value_counts())}")

# -----------------------------------------------------------------------------
# 2) Blind train/test split
# -----------------------------------------------------------------------------
train_idx, test_idx = train_test_split(
    labeled_raw.index,
    test_size=split_cfg.TEST_SIZE,
    stratify=labeled_raw["Category"],
    random_state=split_cfg.RANDOM_STATE,
)

train_raw = labeled_raw.loc[train_idx].copy()
test_raw = labeled_raw.loc[test_idx].copy()

train_raw.to_csv(TRAIN_SPLIT_CSV, index=False)
test_raw.to_csv(TEST_SPLIT_CSV, index=False)

print(f"\nTrain: {len(train_raw)} | Test: {len(test_raw)}")
print(f"Train dist: {dict(train_raw['Category'].value_counts())}")
print(f"Test  dist: {dict(test_raw['Category'].value_counts())}")

# -----------------------------------------------------------------------------
# 3) Label corrections
# -----------------------------------------------------------------------------
train_corrected, correction_log = apply_label_corrections(train_raw, verbose=True)
test_corrected, _ = apply_label_corrections(test_raw, verbose=False)

correction_log.to_csv(str(Paths.correction_log_csv()), index=False)

# -----------------------------------------------------------------------------
# 4) Feature engineering
# -----------------------------------------------------------------------------
print("\nEngineering train set...")
train_eng, binary_cols, drink_avg_cols, drink_count_cols = engineer(train_corrected.copy())

print("Engineering test set...")
test_eng, _, _, _ = engineer(test_corrected.copy())

print("Engineering unlabeled set...")
unlabeled_eng, _, _, _ = engineer(unlabeled.copy())

feature_cols = list(dict.fromkeys(
    f for f in (CORE_FEATURES + drink_avg_cols + drink_count_cols + binary_cols)
    if f in train_eng.columns
))

# Apply user-rejected columns again after engineering
feature_cols = [f for f in feature_cols if f not in USER_REJECT_COLUMNS]

print(f"Original features: {len(feature_cols)}")
feature_cols = select_features(train_eng, feature_cols)
print(f"Features after reduction: {len(feature_cols)}")


X_train, feature_cols, le, y_train, imputer = build_feature_matrix(
    train_eng, feature_cols
)

def align_to_feature_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Return a DataFrame with exactly `cols` columns in order.
    Columns missing from df are filled with NaN for the imputer to handle."""
    aligned = pd.DataFrame(index=df.index)
    for c in cols:
        aligned[c] = df[c] if c in df.columns else float("nan")
    return aligned

X_test  = imputer.transform(align_to_feature_cols(test_eng,      feature_cols).astype(float))
y_test  = le.transform(test_corrected["Category"])
X_unlab = imputer.transform(align_to_feature_cols(unlabeled_eng, feature_cols).astype(float))

joblib.dump(feature_cols, FEATURE_COLS_PKL)
joblib.dump(imputer, IMPUTER_PKL)
joblib.dump(le, LABEL_ENCODER_PKL)

# -----------------------------------------------------------------------------
# 5) Correlation and distribution analysis
# -----------------------------------------------------------------------------
corr_df = run_correlation_analysis(
    train_eng, feature_cols, y_train, le, X_train,
    top_n=analysis_cfg.CORR_TOP_N,
    output_dir=str(OUTPUT_DIR),
)
corr_df.to_csv(str(Paths.correlation_csv()), index=False)

dist_df = run_distribution_analysis(
    train_eng, unlabeled_eng, feature_cols, imputer,
    output_dir=str(OUTPUT_DIR),
)
dist_df.to_csv(str(Paths.distribution_csv()), index=False)

# -----------------------------------------------------------------------------
# 6) CV setup
# -----------------------------------------------------------------------------
skf = StratifiedKFold(
    n_splits=cv_cfg.N_SPLITS,
    shuffle=cv_cfg.SHUFFLE,
    random_state=cv_cfg.RANDOM_STATE,
)

results = []

# -----------------------------------------------------------------------------
# 7) Tune / fit models
# -----------------------------------------------------------------------------
print(f"\n{'=' * 100}")
print(f"MODEL TRAINING — {cv_cfg.N_SPLITS}-fold CV on TRAIN set")
print(f"{'=' * 100}")

# 7a) GBM
print("\nTuning GBM...")
gbm_best = tune_gbm(X_train, y_train, skf)
gbm_params = gbm_best["params"]

gbm_cv_pred = gbm_best["oof_pred"]
gbm_cv_acc = gbm_best["cv_acc"]

gbm_final = GradientBoostingClassifier(
    **gbm_params,
    random_state=model_cfg.RANDOM_STATE,
    **model_cfg.GBM_FIXED,
)
gbm_sw = compute_sample_weight("balanced", y_train)
gbm_final.fit(X_train, y_train, sample_weight=gbm_sw)

gbm_test_pred = gbm_final.predict(X_test)
gbm_test_acc = accuracy_score(y_test, gbm_test_pred)

results.append({
    "model": "GBM (tuned)",
    "cv_acc": gbm_cv_acc,
    "test_acc": gbm_test_acc,
    **class_metrics(y_test, gbm_test_pred, le),
    "params": json.dumps(gbm_params),
})

save_model_bundle("gbm_tuned", {
    "model_name": "GBM (tuned)",
    "model": gbm_final,
    "params": gbm_params,
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Best GBM params: {gbm_params} | CV {gbm_cv_acc:.1%} | Test {gbm_test_acc:.1%}")

# 7b) RF
print("Tuning RF...")
rf_best = tune_rf(X_train, y_train, skf)
rf_params = rf_best["params"]

rf_final = RandomForestClassifier(
    **rf_params,
    random_state=model_cfg.RANDOM_STATE,
    **model_cfg.RF_FIXED,
)
rf_final.fit(X_train, y_train)

rf_cv_pred = rf_best["oof_pred"]
rf_cv_acc = rf_best["cv_acc"]
rf_test_pred = rf_final.predict(X_test)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

results.append({
    "model": "RF (tuned)",
    "cv_acc": rf_cv_acc,
    "test_acc": rf_test_acc,
    **class_metrics(y_test, rf_test_pred, le),
    "params": json.dumps(rf_params),
})

save_model_bundle("rf_tuned", {
    "model_name": "RF (tuned)",
    "model": rf_final,
    "params": rf_params,
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Best RF params: {rf_params} | CV {rf_cv_acc:.1%} | Test {rf_test_acc:.1%}")

# 7c) Extra Trees
et_final = ExtraTreesClassifier(
    random_state=model_cfg.RANDOM_STATE,
    **model_cfg.ET_PARAMS,
)
et_cv_pred = np.zeros_like(y_train)
for tri, tei in skf.split(X_train, y_train):
    m = ExtraTreesClassifier(
        random_state=model_cfg.RANDOM_STATE,
        **model_cfg.ET_PARAMS,
    )
    m.fit(X_train[tri], y_train[tri])
    et_cv_pred[tei] = m.predict(X_train[tei])
et_cv_acc = accuracy_score(y_train, et_cv_pred)

et_final.fit(X_train, y_train)
et_test_pred = et_final.predict(X_test)
et_test_acc = accuracy_score(y_test, et_test_pred)

results.append({
    "model": "Extra Trees",
    "cv_acc": et_cv_acc,
    "test_acc": et_test_acc,
    **class_metrics(y_test, et_test_pred, le),
    "params": json.dumps(model_cfg.ET_PARAMS),
})

save_model_bundle("extra_trees", {
    "model_name": "Extra Trees",
    "model": et_final,
    "params": model_cfg.ET_PARAMS,
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Extra Trees | CV {et_cv_acc:.1%} | Test {et_test_acc:.1%}")

# 7d) k-NN
print("Tuning k-NN...")
knn_best = tune_knn(X_train, y_train, skf)
knn_params = knn_best["params"]

knn_final = KNeighborsClassifier(
    **knn_params,
    **model_cfg.KNN_FIXED,
)
knn_final.fit(X_train, y_train)

knn_cv_pred = knn_best["oof_pred"]
knn_cv_acc = knn_best["cv_acc"]
knn_test_pred = knn_final.predict(X_test)
knn_test_acc = accuracy_score(y_test, knn_test_pred)

results.append({
    "model": "k-NN (tuned)",
    "cv_acc": knn_cv_acc,
    "test_acc": knn_test_acc,
    **class_metrics(y_test, knn_test_pred, le),
    "params": json.dumps(knn_params),
})

save_model_bundle("knn_tuned", {
    "model_name": "k-NN (tuned)",
    "model": knn_final,
    "params": knn_params,
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Best k-NN params: {knn_params} | CV {knn_cv_acc:.1%} | Test {knn_test_acc:.1%}")

# 7e) Semi-supervised GBM
print("Running semi-supervised GBM...")

# Use the best tuned GBM model to pseudo-label unlabeled data
gbm_seed = GradientBoostingClassifier(
    **gbm_params,
    random_state=model_cfg.RANDOM_STATE,
    **model_cfg.GBM_FIXED,
)
gbm_seed.fit(X_train, y_train, sample_weight=gbm_sw)

proba_u = gbm_seed.predict_proba(X_unlab)
pseudo_u = gbm_seed.predict(X_unlab)
conf_mask = proba_u.max(axis=1) >= ss_cfg.CONFIDENCE_THRESHOLD

pseudo_labels = pseudo_u[conf_mask]
X_pseudo = X_unlab[conf_mask]

print(f"  Pseudo-labels kept (≥{ss_cfg.CONFIDENCE_THRESHOLD}): {conf_mask.sum()}")

ss_model = GradientBoostingClassifier(
    **gbm_params,
    random_state=model_cfg.RANDOM_STATE,
    **model_cfg.GBM_FIXED,
)

X_ss = np.vstack([X_train, X_pseudo]) if len(X_pseudo) else X_train
y_ss = np.concatenate([y_train, pseudo_labels]) if len(pseudo_labels) else y_train
sw_ss = compute_sample_weight("balanced", y_ss)

ss_model.fit(X_ss, y_ss, sample_weight=sw_ss)

ss_cv_pred = np.zeros_like(y_train)
for tri, tei in skf.split(X_train, y_train):
    X_tr = X_train[tri]
    y_tr = y_train[tri]

    # pseudo-label the unlabeled set using a fold-specific model
    fold_seed = GradientBoostingClassifier(
        **gbm_params,
        random_state=model_cfg.RANDOM_STATE,
        **model_cfg.GBM_FIXED,
    )
    sw_fold = compute_sample_weight("balanced", y_tr)
    fold_seed.fit(X_tr, y_tr, sample_weight=sw_fold)

    proba_fold = fold_seed.predict_proba(X_unlab)
    pseudo_fold = fold_seed.predict(X_unlab)
    mask_fold = proba_fold.max(axis=1) >= ss_cfg.CONFIDENCE_THRESHOLD

    X_tr2 = np.vstack([X_tr, X_unlab[mask_fold]])
    y_tr2 = np.concatenate([y_tr, pseudo_fold[mask_fold]])
    sw_tr2 = compute_sample_weight("balanced", y_tr2)

    m = GradientBoostingClassifier(
        **gbm_params,
        random_state=model_cfg.RANDOM_STATE,
        **model_cfg.GBM_FIXED,
    )
    m.fit(X_tr2, y_tr2, sample_weight=sw_tr2)
    ss_cv_pred[tei] = m.predict(X_train[tei])

ss_cv_acc = accuracy_score(y_train, ss_cv_pred)
ss_test_pred = ss_model.predict(X_test)
ss_test_acc = accuracy_score(y_test, ss_test_pred)

results.append({
    "model": "Semi-Supervised",
    "cv_acc": ss_cv_acc,
    "test_acc": ss_test_acc,
    **class_metrics(y_test, ss_test_pred, le),
    "params": json.dumps({
        "gbm_params": gbm_params,
        "confidence_threshold": ss_cfg.CONFIDENCE_THRESHOLD,
    }),
})

save_model_bundle("semi_supervised_gbm", {
    "model_name": "Semi-Supervised",
    "model": ss_model,
    "params": {
        "gbm_params": gbm_params,
        "confidence_threshold": ss_cfg.CONFIDENCE_THRESHOLD,
    },
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Semi-Supervised | CV {ss_cv_acc:.1%} | Test {ss_test_acc:.1%}")

# 7f) Stacking
print("Building stacking ensemble...")
stack_bundle = fit_stacking_bundle(
    X_train=X_train,
    y_train=y_train,
    X_unlab=X_unlab,
    best_gbm_params=gbm_params,
    best_knn_params=knn_params,
    skf=skf,
)

stack_cv_acc = stack_bundle["cv_acc"]
stack_test_pred = predict_stacking(stack_bundle, X_test)
stack_test_acc = accuracy_score(y_test, stack_test_pred)

results.append({
    "model": "Stacking",
    "cv_acc": stack_cv_acc,
    "test_acc": stack_test_acc,
    **class_metrics(y_test, stack_test_pred, le),
    "params": json.dumps({
        "base_models": ["GBM1", "GBM2", "GBM3", "RF", "ET", "KNN"],
        "meta": model_cfg.META_LR,
    }),
})

save_model_bundle("stacking", {
    "model_name": "Stacking",
    "base_models": stack_bundle["base_models"],
    "meta_model": stack_bundle["meta_model"],
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Stacking | CV {stack_cv_acc:.1%} | Test {stack_test_acc:.1%}")

# 7g) Majority vote
all_cv_preds = {
    "GBM (tuned)": gbm_cv_pred,
    "RF (tuned)": rf_cv_pred,
    "Extra Trees": et_cv_pred,
    "k-NN (tuned)": knn_cv_pred,
    "Semi-Supervised": ss_cv_pred,
    "Stacking": stack_test_pred,  # for evaluation reporting only
}

# For proper CV majority vote, keep only OOF vectors of same length as train
majority_cv_matrix = np.column_stack([
    gbm_cv_pred,
    rf_cv_pred,
    et_cv_pred,
    knn_cv_pred,
    ss_cv_pred,
])
majority_cv_pred = stats.mode(majority_cv_matrix, axis=1, keepdims=False)[0].astype(int)
majority_cv_acc = accuracy_score(y_train, majority_cv_pred)

majority_test_matrix = np.column_stack([
    gbm_test_pred,
    rf_test_pred,
    et_test_pred,
    knn_test_pred,
    ss_test_pred,
    stack_test_pred,
])
majority_test_pred = stats.mode(majority_test_matrix, axis=1, keepdims=False)[0].astype(int)
majority_test_acc = accuracy_score(y_test, majority_test_pred)

results.append({
    "model": "Majority Vote",
    "cv_acc": majority_cv_acc,
    "test_acc": majority_test_acc,
    **class_metrics(y_test, majority_test_pred, le),
    "params": json.dumps({"members": ["GBM", "RF", "ET", "KNN", "Semi-Supervised", "Stacking"]}),
})

save_model_bundle("majority_vote", {
    "model_name": "Majority Vote",
    "members": {
        "gbm": gbm_final,
        "rf": rf_final,
        "et": et_final,
        "knn": knn_final,
        "semi_supervised": ss_model,
        "stacking": stack_bundle,
    },
    "feature_cols": feature_cols,
    "label_encoder": le,
    "imputer": imputer,
})

print(f"  Majority Vote | CV {majority_cv_acc:.1%} | Test {majority_test_acc:.1%}")

# -----------------------------------------------------------------------------
# 8) Results table
# -----------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(METRICS_CSV, index=False)

metrics_json = {r["model"]: r for r in results}
save_json(metrics_json, METRICS_JSON)

print_results_table(results, le, "MODEL RESULTS")

best_row = results_df.sort_values(["test_acc", "cv_acc"], ascending=False).iloc[0].to_dict()
best_model_name = best_row["model"]

save_json(best_row, BEST_MODEL_JSON)

print(f"\nBest model by test accuracy: {best_model_name} | {best_row['test_acc']:.1%}")

# -----------------------------------------------------------------------------
# 9) Save manifest
# -----------------------------------------------------------------------------
manifest = {
    "best_model_name": best_model_name,
    "feature_cols_path": str(FEATURE_COLS_PKL),
    "imputer_path": str(IMPUTER_PKL),
    "label_encoder_path": str(LABEL_ENCODER_PKL),
    "train_split_csv": str(TRAIN_SPLIT_CSV),
    "test_split_csv": str(TEST_SPLIT_CSV),
    "correlation_csv": str(Paths.correlation_csv()),
    "correlation_plot": str(Paths.correlation_plot()),
    "distribution_csv": str(Paths.distribution_csv()),
    "metrics_csv": str(METRICS_CSV),
    "metrics_json": str(METRICS_JSON),
    "best_model_json": str(BEST_MODEL_JSON),
    "models": {
        "gbm_tuned": str(MODEL_DIR / "gbm_tuned.joblib"),
        "rf_tuned": str(MODEL_DIR / "rf_tuned.joblib"),
        "extra_trees": str(MODEL_DIR / "extra_trees.joblib"),
        "knn_tuned": str(MODEL_DIR / "knn_tuned.joblib"),
        "semi_supervised_gbm": str(MODEL_DIR / "semi_supervised_gbm.joblib"),
        "stacking": str(MODEL_DIR / "stacking.joblib"),
        "majority_vote": str(MODEL_DIR / "majority_vote.joblib"),
    },
}

joblib.dump(manifest, MANIFEST_PKL)
save_json(manifest, MANIFEST_JSON)

# -----------------------------------------------------------------------------
# 10) Final evaluation summary
# -----------------------------------------------------------------------------
print(f"\n{'─' * 80}")
print("OVERFITTING CHECK")
print(f"{'─' * 80}")
gap = float(best_row["cv_acc"] - best_row["test_acc"])
flag = (
    f"⚠  Gap > {analysis_cfg.OVERFIT_GAP_THRESHOLD:.0%} — consider more regularisation"
    if gap > analysis_cfg.OVERFIT_GAP_THRESHOLD
    else "✓  Gap is within acceptable range"
)
print(f"  Best CV accuracy  : {best_row['cv_acc']:.1%}")
print(f"  Best Test accuracy: {best_row['test_acc']:.1%}")
print(f"  Gap               : {gap:+.1%}  ({flag})")

print(f"\n{'=' * 100}")
print("SAVED ARTIFACTS")
print(f"{'=' * 100}")
print(f"  Train split   : {TRAIN_SPLIT_CSV}")
print(f"  Test split    : {TEST_SPLIT_CSV}")
print(f"  Metrics CSV   : {METRICS_CSV}")
print(f"  Metrics JSON  : {METRICS_JSON}")
print(f"  Best model    : {BEST_MODEL_JSON}")
print(f"  Manifest      : {MANIFEST_JSON}")
print(f"  Correlation   : {Paths.correlation_csv()}")
print(f"  Plot          : {Paths.correlation_plot()}")
print(f"  Distribution  : {Paths.distribution_csv()}")
print(f"  Model dir     : {MODEL_DIR}")
print(f"{'=' * 100}")