"""
models.py
────────────────────────────────────────────────────────────────────────────────
Restaurant Classifier — All Models
  · Blind test split (stratified, held out BEFORE any fitting)
  · Label corrections on train fold only  (from label_corrections.py)
  · Feature engineering + imputation      (from feature_engineering.py)
  · Correlation analysis  → CSV + PNG plot
  · Distribution analysis → CSV
  · GBM (grid-tuned), RF (grid-tuned), Extra Trees,
    Semi-Supervised GBM, Stacking Ensemble, Majority Vote
  · CV results  +  Blind test results  +  Overfitting gap check
  · Final production predictions on all 4323 restaurants

All paths / hyperparameters / thresholds live in config.py.
────────────────────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.neighbors import KNeighborsClassifier


# ── Project modules ───────────────────────────────────────────────────────────
from config import (
    Paths, data_cfg, split_cfg, cv_cfg,
    model_cfg, ss_cfg, analysis_cfg,
)
from label_corrections  import apply_label_corrections
from feature_engineering import (
    engineer, select_features, build_feature_matrix,
    run_correlation_analysis, run_distribution_analysis,
    CORE_FEATURES,
)

# ── Ensure output directory exists ────────────────────────────────────────────
Paths.ensure_output_dir()
OUTPUT_DIR = str(Paths.OUTPUT_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 100)
print("RESTAURANT CLASSIFIER V10-CLEAN  |  NO SHERLOCK / ST  |  BLIND TEST SPLIT")
print("=" * 100)

df = pd.read_csv(str(Paths.INPUT_CSV), low_memory=False)

for col in data_cfg.LEAKAGE_COLUMNS:
    if col in df.columns:
        df = df.drop(columns=[col])
        print(f"Dropped leakage column: '{col}'")

df["Category"] = df["Category"].str.strip().str.title()

labeled_raw = df[df["Category"].isin(data_cfg.VALID_CATEGORIES)].copy()
unlabeled   = df[~df["Category"].isin(data_cfg.VALID_CATEGORIES)].copy()

print(f"\nFull labeled: {len(labeled_raw)} | Unlabeled: {len(unlabeled)}")
print(f"Distribution: {dict(labeled_raw['Category'].value_counts())}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. BLIND TEST SPLIT  — happens before ANYTHING else
# ══════════════════════════════════════════════════════════════════════════════
train_idx, test_idx = train_test_split(
    labeled_raw.index,
    test_size=split_cfg.TEST_SIZE,
    stratify=labeled_raw["Category"],
    random_state=split_cfg.RANDOM_STATE,
)
train_raw = labeled_raw.loc[train_idx].copy()
test_raw  = labeled_raw.loc[test_idx].copy()

print(f"\nTrain: {len(train_raw)} | Test (blind): {len(test_raw)}")
print(f"Train dist: {dict(train_raw['Category'].value_counts())}")
print(f"Test  dist: {dict(test_raw['Category'].value_counts())}")
print(f"\n⚠  Test set LOCKED — not touched until final evaluation.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LABEL CORRECTIONS  — train fold only
# ══════════════════════════════════════════════════════════════════════════════
train_corrected, correction_log = apply_label_corrections(train_raw, verbose=True)
correction_log.to_csv(str(Paths.correction_log_csv()), index=False)
print(f"Correction log → {Paths.correction_log_csv()}  ({len(correction_log)} rows)\n")

# Same rules applied to test (domain rules, not data-fitted — acceptable)
test_corrected, _ = apply_label_corrections(test_raw, verbose=False)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("Engineering train set...")
train_eng, binary_cols, drink_avg_cols, drink_count_cols = engineer(train_corrected.copy())

print("Engineering test set...")
test_eng, _, _, _ = engineer(test_corrected.copy())

print("Engineering unlabeled set...")
unlabeled_eng, _, _, _ = engineer(unlabeled.copy())

# ── Build feature list ────────────────────────────────────────────────────────
feature_cols = list(dict.fromkeys(
    f for f in CORE_FEATURES + drink_avg_cols + drink_count_cols + binary_cols
    if f in train_eng.columns
))

# After: feature_cols = list(dict.fromkeys(...))
print(f"Orignal Features: {len(feature_cols)}")
feature_cols = select_features(train_eng, feature_cols)
print(f"Features after reduction: {len(feature_cols)}")

# ── Build train matrix (imputer fitted on train only) ─────────────────────────
X_train, feature_cols, le, y_train, imputer = build_feature_matrix(
    train_eng, feature_cols
)
print(f"\nFeatures: {len(feature_cols)} | Train samples: {len(y_train)}")
print(f"Labels  : {dict(zip(le.classes_, np.bincount(y_train)))}")

# ── Transform test & unlabeled with train imputer ─────────────────────────────
valid_test  = [f for f in feature_cols if f in test_eng.columns]
X_test      = imputer.transform(test_eng[valid_test].astype(float))
y_test      = le.transform(test_corrected["Category"])

valid_unlab = [f for f in feature_cols if f in unlabeled_eng.columns]
X_unlab     = imputer.transform(unlabeled_eng[valid_unlab].astype(float))


# ══════════════════════════════════════════════════════════════════════════════
# 5. CORRELATION & DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
corr_df = run_correlation_analysis(
    train_eng, feature_cols, y_train, le, X_train,
    top_n=analysis_cfg.CORR_TOP_N, output_dir=OUTPUT_DIR,
)
dist_df = run_distribution_analysis(
    train_eng, unlabeled_eng, feature_cols, imputer,
    output_dir=OUTPUT_DIR,
)

skf = StratifiedKFold(
    n_splits=cv_cfg.N_SPLITS,
    shuffle=cv_cfg.SHUFFLE,
    random_state=cv_cfg.RANDOM_STATE,
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. TRAIN MODELS  (CV on train set only)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("MODEL TRAINING — {}-fold CV on TRAIN set".format(cv_cfg.N_SPLITS))
print(f"{'=' * 100}")

all_preds_cv = {}

# ── 6a. Tuned GBM ─────────────────────────────────────────────────────────────
print("\nTuning GBM...")
best_acc, best_pred, best_params = 0, None, None
g = model_cfg.GBM_GRID
f = model_cfg.GBM_FIXED

for n_est in g["n_estimators"]:
    for max_d in g["max_depth"]:
        for lr in g["learning_rate"]:
            for ss in g["subsample"]:
                preds = np.zeros_like(y_train)
                for tri, tei in skf.split(X_train, y_train):
                    sw = compute_sample_weight("balanced", y_train[tri])
                    m  = GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=max_d,
                        learning_rate=lr, subsample=ss,
                        random_state=model_cfg.RANDOM_STATE,
                        **f,
                    )
                    m.fit(X_train[tri], y_train[tri], sample_weight=sw)
                    preds[tei] = m.predict(X_train[tei])
                a = accuracy_score(y_train, preds)
                if a > best_acc:
                    best_acc   = a
                    best_pred  = preds.copy()
                    best_params= (n_est, max_d, lr, ss)

all_preds_cv["GBM (tuned)"] = best_pred
print(f"  Best GBM: n={best_params[0]}, d={best_params[1]}, "
      f"lr={best_params[2]}, ss={best_params[3]} → CV {best_acc:.1%}")

# ── 6b. Tuned RF ──────────────────────────────────────────────────────────────
print("Tuning RF...")
best_rf, best_rf_pred = 0, None
rg = model_cfg.RF_GRID
rf = model_cfg.RF_FIXED

for n_est in rg["n_estimators"]:
    for max_d in rg["max_depth"]:
        for msl in rg["min_samples_leaf"]:
            preds = np.zeros_like(y_train)
            for tri, tei in skf.split(X_train, y_train):
                m = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_d,
                    min_samples_leaf=msl,
                    random_state=model_cfg.RANDOM_STATE,
                    **rf,
                )
                m.fit(X_train[tri], y_train[tri])
                preds[tei] = m.predict(X_train[tei])
            a = accuracy_score(y_train, preds)
            if a > best_rf:
                best_rf, best_rf_pred = a, preds.copy()

all_preds_cv["RF (tuned)"] = best_rf_pred
print(f"  Best RF CV: {best_rf:.1%}")

# ── 6c. Extra Trees ───────────────────────────────────────────────────────────
preds = np.zeros_like(y_train)
for tri, tei in skf.split(X_train, y_train):
    m = ExtraTreesClassifier(
        random_state=model_cfg.RANDOM_STATE,
        **model_cfg.ET_PARAMS,
    )
    m.fit(X_train[tri], y_train[tri])
    preds[tei] = m.predict(X_train[tei])
all_preds_cv["Extra Trees"] = preds
print(f"  Extra Trees CV: {accuracy_score(y_train, preds):.1%}")

# Add to your model training loop (after Extra Trees)
print("Tuning k-NN...")
best_knn, best_knn_pred = 0, None
kg = model_cfg.KNN_GRID
for n in kg["n_neighbors"]:
    for w in kg["weights"]:
        for metric in kg["metric"]:
            preds = np.zeros_like(y_train)
            for tri, tei in skf.split(X_train, y_train):
                m = KNeighborsClassifier(
                    n_neighbors=n, weights=w, metric=metric,
                    **model_cfg.KNN_FIXED,
                )
                m.fit(X_train[tri], y_train[tri])
                preds[tei] = m.predict(X_train[tei])
            a = accuracy_score(y_train, preds)
            if a > best_knn:
                best_knn, best_knn_pred = a, preds.copy()

all_preds_cv["k-NN (tuned)"] = best_knn_pred
print(f"  Best k-NN CV: {best_knn:.1%}")

# ── 6d. Semi-Supervised GBM ───────────────────────────────────────────────────
print("Running semi-supervised...")
sw = compute_sample_weight("balanced", y_train)
clf_ss = GradientBoostingClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2], subsample=best_params[3],
    random_state=model_cfg.RANDOM_STATE,
    **model_cfg.GBM_FIXED,
)
clf_ss.fit(X_train, y_train, sample_weight=sw)
proba_u   = clf_ss.predict_proba(X_unlab)
pseudo    = clf_ss.predict(X_unlab)
conf_mask = proba_u.max(axis=1) >= ss_cfg.CONFIDENCE_THRESHOLD
dist_ps   = pd.Series(le.inverse_transform(pseudo[conf_mask])).value_counts()
print(f"  Pseudo-labels (≥{ss_cfg.CONFIDENCE_THRESHOLD}): "
      f"{conf_mask.sum()} → {dict(dist_ps)}")

preds_ss = np.zeros_like(y_train)
for tri, tei in skf.split(X_train, y_train):
    X_tr = np.vstack([X_train[tri], X_unlab[conf_mask]])
    y_tr = np.concatenate([y_train[tri], pseudo[conf_mask]])
    sw2  = compute_sample_weight("balanced", y_tr)
    m    = GradientBoostingClassifier(
        n_estimators=best_params[0], max_depth=best_params[1],
        learning_rate=best_params[2], subsample=best_params[3],
        random_state=model_cfg.RANDOM_STATE,
        **model_cfg.GBM_FIXED,
    )
    m.fit(X_tr, y_tr, sample_weight=sw2)
    preds_ss[tei] = m.predict(X_train[tei])
all_preds_cv["Semi-Supervised"] = preds_ss
print(f"  Semi-Supervised CV: {accuracy_score(y_train, preds_ss):.1%}")

# ── 6e. Stacking Ensemble ─────────────────────────────────────────────────────
print("Building stacking ensemble...")
models_stack = [
    ("GBM1", GradientBoostingClassifier(
        n_estimators=best_params[0], max_depth=best_params[1],
        learning_rate=best_params[2], subsample=best_params[3],
        random_state=model_cfg.RANDOM_STATE, **model_cfg.GBM_FIXED), True),
    ("GBM2", GradientBoostingClassifier(
        **model_cfg.STACK_GBM2), True),
    ("GBM3", GradientBoostingClassifier(
        **model_cfg.STACK_GBM3), True),
    ("RF",   RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        random_state=model_cfg.RANDOM_STATE, **model_cfg.RF_FIXED), False),
    ("ET",   ExtraTreesClassifier(
        random_state=model_cfg.RANDOM_STATE, **model_cfg.ET_PARAMS), False),
]

nc   = len(le.classes_)
meta = np.zeros((len(y_train), len(models_stack) * nc))
for mi, (nm, mod, usw) in enumerate(models_stack):
    for tri, tei in skf.split(X_train, y_train):
        mm = type(mod)(**mod.get_params())
        if usw:
            sw = compute_sample_weight("balanced", y_train[tri])
            mm.fit(X_train[tri], y_train[tri], sample_weight=sw)
        else:
            mm.fit(X_train[tri], y_train[tri])
        meta[tei, mi*nc:(mi+1)*nc] = mm.predict_proba(X_train[tei])

meta_votes  = np.column_stack(list(all_preds_cv.values()))
meta_all    = np.column_stack([meta, meta_votes])
lr_meta     = LogisticRegression(**model_cfg.META_LR, random_state=model_cfg.RANDOM_STATE)
preds_stack = cross_val_predict(lr_meta, meta_all, y_train, cv=skf)
all_preds_cv["Stacking"] = preds_stack
print(f"  Stacking CV: {accuracy_score(y_train, preds_stack):.1%}")

# ── 6f. Majority Vote ─────────────────────────────────────────────────────────
pred_matrix = np.column_stack(list(all_preds_cv.values()))
preds_maj   = stats.mode(pred_matrix, axis=1)[0].ravel().astype(int)
all_preds_cv["Majority Vote"] = preds_maj
print(f"  Majority Vote CV: {accuracy_score(y_train, preds_maj):.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS TABLE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def _print_results_table(all_preds: dict, y_true: np.ndarray, le, title: str):
    print(f"\n{'=' * 100}")
    print(title)
    print(f"{'=' * 100}")
    header = (
        f"{'Model':<25s} {'Acc':>7s} │ "
        f"{'M Pre':>6s} {'M Rec':>6s} {'M F1':>6s} │ "
        f"{'P Pre':>6s} {'P Rec':>6s} {'P F1':>6s} │ "
        f"{'SP Pre':>7s} {'SP Rec':>7s} {'SP F1':>7s}"
    )
    print(f"\n{header}")
    print("─" * 105)

    max_acc = max(accuracy_score(y_true, p) for p in all_preds.values())
    for name, pred in all_preds.items():
        acc  = accuracy_score(y_true, pred)
        cm   = confusion_matrix(y_true, pred)
        sp_i = list(le.classes_).index("Super Premium")
        p_i  = list(le.classes_).index("Premium")
        m_i  = list(le.classes_).index("Mainstream")

        def prf(i):
            pre = cm[i,i] / cm[:,i].sum() if cm[:,i].sum() > 0 else 0
            rec = cm[i,i] / cm[i,:].sum() if cm[i,:].sum() > 0 else 0
            f1  = 2*pre*rec / (pre+rec) if (pre+rec) > 0 else 0
            return pre, rec, f1

        mp,mr,mf = prf(m_i)
        pp,pr,pf = prf(p_i)
        sp,sr,sf = prf(sp_i)
        mark     = " ◀ BEST" if acc >= max_acc else ""
        print(
            f"{name:<25s} {acc:>6.1%} │ "
            f"{mp:>5.1%} {mr:>5.1%} {mf:>5.1%} │ "
            f"{pp:>5.1%} {pr:>5.1%} {pf:>5.1%} │ "
            f"{sp:>6.1%} {sr:>6.1%} {sf:>6.1%}{mark}"
        )


# ── CV results ────────────────────────────────────────────────────────────────
_print_results_table(all_preds_cv, y_train, le, "CV RESULTS (TRAIN SET)")
best_cv_name = max(all_preds_cv, key=lambda k: accuracy_score(y_train, all_preds_cv[k]))
best_cv_acc  = accuracy_score(y_train, all_preds_cv[best_cv_name])
print(f"\nBest CV: {best_cv_name} → {best_cv_acc:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. BLIND TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print(f"BLIND TEST EVALUATION  (n={len(y_test)})")
print(f"{'=' * 100}")

test_preds = {}

# GBM
sw = compute_sample_weight("balanced", y_train)
gbm_final = GradientBoostingClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2], subsample=best_params[3],
    random_state=model_cfg.RANDOM_STATE, **model_cfg.GBM_FIXED,
)
gbm_final.fit(X_train, y_train, sample_weight=sw)
test_preds["GBM (tuned)"] = gbm_final.predict(X_test)

# RF
rf_final = RandomForestClassifier(
    n_estimators=800, max_depth=8, min_samples_leaf=2,
    random_state=model_cfg.RANDOM_STATE, **model_cfg.RF_FIXED,
)
rf_final.fit(X_train, y_train)
test_preds["RF (tuned)"] = rf_final.predict(X_test)

# Extra Trees
et_final = ExtraTreesClassifier(
    random_state=model_cfg.RANDOM_STATE, **model_cfg.ET_PARAMS,
)
et_final.fit(X_train, y_train)
test_preds["Extra Trees"] = et_final.predict(X_test)

# Majority vote
tv_matrix = np.column_stack(list(test_preds.values()))
test_preds["Majority Vote"] = stats.mode(tv_matrix, axis=1)[0].ravel().astype(int)

_print_results_table(test_preds, y_test, le, "BLIND TEST SET RESULTS")

best_test_name = max(test_preds, key=lambda k: accuracy_score(y_test, test_preds[k]))
best_test_pred = test_preds[best_test_name]
best_test_acc  = accuracy_score(y_test, best_test_pred)

print(f"\nBest Test: {best_test_name} → {best_test_acc:.1%}")
print("\nConfusion Matrix (test):")
print(pd.DataFrame(
    confusion_matrix(y_test, best_test_pred),
    index=le.classes_, columns=le.classes_,
))
print(classification_report(y_test, best_test_pred,
                             target_names=le.classes_, digits=3))

# ── Overfitting check ─────────────────────────────────────────────────────────
print(f"\n{'─' * 60}")
print("OVERFITTING CHECK")
print(f"{'─' * 60}")
gap  = best_cv_acc - best_test_acc
flag = (
    f"⚠  Gap > {analysis_cfg.OVERFIT_GAP_THRESHOLD:.0%} — consider more regularisation"
    if gap > analysis_cfg.OVERFIT_GAP_THRESHOLD
    else "✓  Gap is within acceptable range"
)
print(f"  CV accuracy  : {best_cv_acc:.1%}")
print(f"  Test accuracy: {best_test_acc:.1%}")
print(f"  Gap          : {gap:+.1%}  ({flag})")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 60}")
print("TOP 15 FEATURES (GBM, trained on full train set)")
print(f"{'─' * 60}")
fi = pd.Series(
    gbm_final.feature_importances_, index=feature_cols
).sort_values(ascending=False)
cum = 0
for i, (feat, val) in enumerate(fi.head(15).items()):
    cum += val
    print(f"  {i+1:>2d}. {feat:<45s} {val:.4f}  (cum {cum:.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# 10. SCORE ALL RESTAURANTS & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("SCORING ALL RESTAURANTS")
print(f"{'=' * 80}")

# Retrain on train + test combined for production model
X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])
sw_all = compute_sample_weight("balanced", y_all)

prod_model = GradientBoostingClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2], subsample=best_params[3],
    random_state=model_cfg.RANDOM_STATE, **model_cfg.GBM_FIXED,
)
prod_model.fit(X_all, y_all, sample_weight=sw_all)

pred_u  = prod_model.predict(X_unlab)
proba_u = prod_model.predict_proba(X_unlab)

m_idx  = list(le.classes_).index("Mainstream")
p_idx  = list(le.classes_).index("Premium")
sp_idx = list(le.classes_).index("Super Premium")

out = df.copy()
out["model_predicted_category"] = ""
out["prob_mainstream"]           = 0.0
out["prob_premium"]              = 0.0
out["prob_super_premium"]        = 0.0
out["confidence"]                = 0.0
out["prediction_source"]         = ""

for idx in labeled_raw.index:
    if idx in train_corrected.index:
        cat = train_corrected.at[idx, "Category"]
        src = "Human Label (train, corrected)"
    else:
        cat = test_corrected.at[idx, "Category"]
        src = "Human Label (test, corrected)"
    out.at[idx, "model_predicted_category"] = cat
    out.at[idx, "confidence"]               = 1.0
    out.at[idx, "prediction_source"]        = src

for i, idx in enumerate(unlabeled.index):
    out.at[idx, "model_predicted_category"] = le.inverse_transform([pred_u[i]])[0]
    out.at[idx, "prob_mainstream"]           = float(proba_u[i, m_idx])
    out.at[idx, "prob_premium"]              = float(proba_u[i, p_idx])
    out.at[idx, "prob_super_premium"]        = float(proba_u[i, sp_idx])
    out.at[idx, "confidence"]               = float(proba_u[i].max())
    out.at[idx, "prediction_source"]        = "Model V10-Clean"

out.to_csv(str(Paths.predictions_csv()), index=False)
print(f"\nSaved: {Paths.predictions_csv()}  ({len(out)} rows)")
print(f"\nFull distribution:")
print(out["model_predicted_category"].value_counts().to_string())

sp_model = out[
    (out["model_predicted_category"] == "Super Premium") &
    (out["prediction_source"] == "Model V10-Clean")
].sort_values("prob_super_premium", ascending=False)

print(f"\nSuper Premium (model-predicted): {len(sp_model)}")
print(f"\n  {'#':>3s}  {'Restaurant':<50s} {'SP Prob':>8s}  {'Conf':>6s}")
print(f"  {'─' * 72}")
for i, (_, r) in enumerate(sp_model.head(20).iterrows()):
    nm = str(r.get("name",""))[:48]
    print(f"  {i+1:>3d}  {nm:<50s} {r['prob_super_premium']:>8.3f}  {r['confidence']:>6.3f}")

print(f"\n{'=' * 100}")
print("DONE")
print(f"{'=' * 100}")
print(f"  {Paths.predictions_csv()}")
print(f"  {Paths.correction_log_csv()}")
print(f"  {Paths.correlation_csv()}")
print(f"  {Paths.correlation_plot()}")
print(f"  {Paths.distribution_csv()}")


# After all models are trained, predict on unlabeled data
unlabeled_preds = {}
for name, model in [
    ("GBM (tuned)", gbm_final),
    ("RF (tuned)", rf_final),
    ("Extra Trees", et_final),
    ("k-NN (tuned)", KNeighborsClassifier(**best_knn_params)),  # Use best params from grid
    ("Semi-Supervised", clf_ss),
    ("Stacking", lr_meta),  # Note: Stacking requires special handling
    ("Majority Vote", None),  # Handled separately
]:
    if name == "Majority Vote":
        preds = stats.mode(np.column_stack([unlabeled_preds[m] for m in unlabeled_preds]), axis=1)[0].ravel()
    elif name == "Stacking":
        # For stacking, you need to generate meta-features for unlabeled data
        # This is complex; see note below
        continue
    else:
        preds = model.predict(X_unlab)
    unlabeled_preds[name] = preds

# Save to CSV
pd.DataFrame(unlabeled_preds, index=unlabeled.index).to_csv(str(Paths.unlabeled_preds_csv()))
print(f"Saved unlabeled predictions: {Paths.unlabeled_preds_csv()}")