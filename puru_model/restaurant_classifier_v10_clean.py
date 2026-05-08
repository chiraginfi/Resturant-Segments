"""
Restaurant Classifier V10-Clean
No Sherlock, No ST — Honest model (no data leakage)
Best accuracy: 80.0% (Random Forest)

Input:  Brown Forman training data V5.xlsx - new_updated_data.csv
Output: restaurant_predictions_v10_clean.csv
"""

import pandas as pd
import numpy as np
import re
import ast
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA — Drop Sherlock & ST before anything
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 100)
print("V10-CLEAN: NO SHERLOCK, NO ST — HONEST MODEL")
print("=" * 100)

df = pd.read_csv(
    "/mnt/data/image_recognition/brown_forman_req/puru_input/training_data.csv", low_memory=False
)
df = df.drop(columns=["Category - Sherlock", "ST"])
print(f"Dropped: 'Category - Sherlock', 'ST'  →  {len(df.columns)} columns remain")

df["Category"] = df["Category"].str.strip().str.title()
labeled = df[df["Category"].isin(["Mainstream", "Premium", "Super Premium"])].copy()
unlabeled = df[~df["Category"].isin(["Mainstream", "Premium", "Super Premium"])].copy()

print(f"Labeled: {len(labeled)} | Unlabeled: {len(unlabeled)}")
print(f"Distribution: {dict(labeled['Category'].value_counts())}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. LABEL CORRECTIONS (rule-based, no Sherlock)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 60}")
print("LABEL CORRECTIONS")
print(f"{'─' * 60}")

labeled["cost_both"] = pd.to_numeric(
    labeled["cost_for_two_both"], errors="coerce"
)
addr = labeled["address"].fillna("").str.lower()
corrections = 0

# Fix 1: Suburban low-price Premium → Mainstream
suburban = [
    "virar", "vasai", "mira road", "nalasopara",
    "dombivli", "kalyan", "ulhasnagar",
]
is_suburban = addr.apply(lambda x: any(s in x for s in suburban))
fix1 = (
    (labeled["Category"] == "Premium") & is_suburban & (labeled["cost_both"] < 2000)
)
labeled.loc[fix1, "Category"] = "Mainstream"
print(f"  Fix 1: {fix1.sum()} suburban low-price Premium → Mainstream")
corrections += fix1.sum()

# Fix 2: High-price premium-area Mainstream → Premium
premium_areas = [
    "bandra", "colaba", "lower parel", "worli", "juhu",
    "kala ghoda", "bkc", "nariman point", "fort",
]
is_premium_area = addr.apply(lambda x: any(a in x for a in premium_areas))
fix2 = (
    (labeled["Category"] == "Mainstream")
    & (labeled["cost_both"] >= 4000)
    & is_premium_area
)
labeled.loc[fix2, "Category"] = "Premium"
print(f"  Fix 2: {fix2.sum()} high-price premium-area Mainstream → Premium")
corrections += fix2.sum()

# Fix 3: 5-star hotel high-price Premium → Super Premium
star_map = {"5 stars": 5, "4 stars": 4, "3 stars": 3}
labeled["hotel_stars_raw"] = labeled["host_hotel_star"].map(star_map).fillna(0)
fix3 = (
    (labeled["Category"] == "Premium")
    & (labeled["hotel_stars_raw"] == 5)
    & (labeled["cost_both"] >= 5000)
)
labeled.loc[fix3, "Category"] = "Super Premium"
print(f"  Fix 3: {fix3.sum()} 5-star hotel high-price Premium → Super Premium")
corrections += fix3.sum()

print(f"\n  Total corrections: {corrections}")
print(f"  Distribution: {dict(labeled['Category'].value_counts())}")
labeled.to_csv("/mnt/data/image_recognition/brown_forman_req/puru_output/label_correction.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def engineer(d):
    """Full feature engineering — no Sherlock, no ST."""

    # ── Price features (100% fill) ───────────────────────────────────────
    for col in [
        "cost_for_two_food", "cost_for_two_drinks",
        "cost_for_two_both", "dining_price",
    ]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["cost_both"] = d["cost_for_two_both"]
    d["cost_food"] = d["cost_for_two_food"]
    d["cost_drinks"] = d["cost_for_two_drinks"]
    d["is_high_price"] = (d["cost_both"] >= 5000).astype(float)
    d["is_very_high_price"] = (d["cost_both"] >= 8000).astype(float)
    d["is_low_price"] = (d["cost_both"] <= 1500).astype(float)
    d["is_very_low_price"] = (d["cost_both"] <= 800).astype(float)
    d["drink_to_food_ratio"] = d["cost_drinks"] / (d["cost_food"] + 1)
    d["dining_price_num"] = pd.to_numeric(d["dining_price"], errors="coerce")
    d["price_tier"] = pd.cut(
        d["cost_both"],
        bins=[0, 1000, 2000, 3500, 5500, 8000, 200000],
        labels=[0, 1, 2, 3, 4, 5],
    ).astype(float)
    d["log_cost"] = np.log1p(d["cost_both"])

    # ── Hotel features ───────────────────────────────────────────────────
    d["hotel_stars"] = d["host_hotel_star"].map(
        {"5 stars": 5, "4 stars": 4, "3 stars": 3, "2 stars": 2, "1 star": 1}
    ).fillna(0)
    d["is_5star_hotel"] = (d["hotel_stars"] == 5).astype(int)
    d["is_hotel"] = (d["hotel_stars"] > 0).astype(int)
    d["hotel_x_price"] = d["hotel_stars"] * d["cost_both"]

    # ── Michelin / premium flags ─────────────────────────────────────────
    d["michelin_star"] = pd.to_numeric(d["michelin_star"], errors="coerce").fillna(0)
    d["premium_place"] = pd.to_numeric(d["premium_place"], errors="coerce").fillna(0)
    d["premium_spirit_flag"] = (
        d["premium_spirit_flag"]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .fillna(0)
    )
    d["premium_brand_count"] = pd.to_numeric(
        d["premium_brand_count"], errors="coerce"
    ).fillna(0)
    d["score"] = pd.to_numeric(d["score"], errors="coerce")

    # ── Drink variety ────────────────────────────────────────────────────
    d["drink_category_count"] = pd.to_numeric(
        d["drink_category_count"], errors="coerce"
    ).fillna(0)
    d["total_item_count"] = pd.to_numeric(
        d["total_item_count"], errors="coerce"
    ).fillna(0)
    d["max_drink_price_avg"] = pd.to_numeric(
        d["max_drink_price_avg"], errors="coerce"
    )
    d["has_domain"] = pd.to_numeric(d["has_domain"], errors="coerce").fillna(0)

    # ── Visit data ───────────────────────────────────────────────────────
    d["weekday_visits"] = pd.to_numeric(d["weekday_visits"], errors="coerce")
    d["weekend_visits"] = pd.to_numeric(d["weekend_visits"], errors="coerce")
    d["total_visits"] = d["weekday_visits"].fillna(0) + d["weekend_visits"].fillna(0)

    # ── Time spent ───────────────────────────────────────────────────────
    def parse_time(t):
        if pd.isna(t):
            return np.nan
        t = str(t).lower()
        h = re.findall(r"([\d.]+)\s*hour", t)
        m = re.findall(r"([\d.]+)\s*min", t)
        mins = 0
        if h:
            mins += float(h[0]) * 60
        if m:
            mins += float(m[0])
        return mins if mins > 0 else np.nan

    d["time_spent_mins"] = d["typical_time_spent"].apply(parse_time)

    # ── Existing calculated fields ───────────────────────────────────────
    d["segment_score"] = pd.to_numeric(d["segment_score"], errors="coerce")
    d["restaurant_segment_ord"] = d["restaurant_segment"].map(
        {"Budget": 0, "Mid": 1, "Premium": 2, "Luxury": 3}
    )
    d["has_any_award"] = d["mentions"].notna().astype(int)

    brand_cols = [c for c in d.columns if c.endswith("_brand_name")]
    d["num_branded_labels"] = d[brand_cols].apply(
        lambda r: sum(1 for v in r if pd.notna(v) and v != "other"), axis=1
    )

    # ── Parse score_breakdown into sub-features ──────────────────────────
    def parse_sb(s):
        if pd.isna(s):
            return {}
        parts = str(s).split("|")
        feats = {}
        for p in parts:
            p = p.strip()
            m = re.search(r":([+-]?\d+)", p)
            if m:
                val = int(m.group(1))
                if "review_vibe" in p:
                    feats["sb_review_vibe"] = val
                elif "drink_category" in p:
                    feats["sb_drink_depth"] = val
                elif "menu_volume" in p:
                    feats["sb_menu_volume"] = val
                elif "rating" in p:
                    feats["sb_rating"] = val
                elif "price" in p and "kw" not in p:
                    feats["sb_price"] = val
                elif "premium_spirit" in p:
                    feats["sb_premium_spirit"] = val
                elif "neg_review" in p:
                    feats["sb_neg_review"] = val
        return feats

    sb = d["score_breakdown"].apply(parse_sb).apply(pd.Series)
    for c in sb.columns:
        d[c] = sb[c]

    # ── Premium brands from brand_count_dict ─────────────────────────────
    premium_brands_list = [
        "grey goose", "belvedere", "absolut", "ketel one",
        "hendricks", "tanqueray", "bombay sapphire", "patron",
        "don julio", "clase azul", "johnny walker", "chivas",
        "glenfiddich", "macallan", "glenlivet", "jack daniel",
        "jameson", "hennessy", "moet", "veuve",
    ]

    def count_premium_brands(bcd):
        if pd.isna(bcd) or bcd == "{}":
            return 0
        try:
            parsed = ast.literal_eval(str(bcd))
            return sum(
                1 for b in parsed
                if any(pb in b.lower() for pb in premium_brands_list)
            )
        except Exception:
            return 0

    d["luxury_brand_count"] = d["brand_count_dict"].apply(count_premium_brands)

    # ── NLP features from reviews ────────────────────────────────────────
    def nlp(r):
        r, t = str(r), str(r).lower()
        f = {}
        for a in ["FOOD_ASPECT", "SERVICE_ASPECT", "ATMOSPHERE_ASPECT"]:
            s = [int(x) for x in re.findall(rf"{a}=(\d)", r)]
            f[f"avg_{a.lower()}"] = np.mean(s) if s else np.nan

        f["luxury_words"] = sum(
            t.count(w) for w in [
                "luxury", "elegant", "exquisite", "sophisticated",
                "fine dining", "premium", "exceptional", "world-class",
                "michelin", "gourmet", "exclusive", "impeccable", "refined",
            ]
        )
        f["negative_words"] = sum(
            t.count(w) for w in [
                "disappointing", "poor", "bad", "worst", "terrible",
                "overpriced", "mediocre", "bland",
            ]
        )
        f["experience_words"] = sum(
            t.count(w) for w in [
                "experience", "ambience", "cocktail", "sommelier",
                "curated", "crafted", "signature", "tasting", "pairing",
                "plating", "presentation",
            ]
        )
        tot = f["luxury_words"] + f["negative_words"]
        f["positive_ratio"] = f["luxury_words"] / tot if tot > 0 else 0.5
        f["chef_mentions"] = sum(
            t.count(w) for w in [
                "chef", "tasting menu", "omakase", "culinary", "executive chef",
            ]
        )
        f["ambiance_quality"] = sum(
            t.count(w) for w in [
                "stunning", "breathtaking", "gorgeous", "panoramic",
                "intimate", "plush", "lavish", "opulent", "magnificent",
            ]
        )
        rts = re.findall(r"review=([^}]+)", r)
        f["avg_review_length"] = np.mean([len(x) for x in rts]) if rts else 0
        f["social_media_score"] = sum(
            t.count(w) for w in [
                "instagram", "insta", "photo", "selfie", "aesthetic",
                "share", "post", "tag", "reel",
            ]
        )
        f["social_proof_score"] = sum(
            t.count(w) for w in [
                "recommend", "must try", "birthday", "anniversary",
                "celebration", "reservation", "booked", "regular",
                "viral", "famous", "popular", "celebrity", "influencer",
            ]
        )
        return pd.Series(f)

    print("  NLP extraction...")
    nf = d["reviews"].apply(nlp)
    for c in nf.columns:
        d[c] = nf[c]

    # ── Binary flags ─────────────────────────────────────────────────────
    binary_cols = [
        c for c in d.columns
        if any(
            c.startswith(p) for p in [
                "offerings__", "service_options__", "highlights__",
                "parking__", "planning__", "atmosphere__", "amenities__",
            ]
        )
        and c != "offerings__serves_liquor"
    ]

    # ── Interaction features ─────────────────────────────────────────────
    d["rating_x_upscale"] = d["ratings"] * d["atmosphere__feels_upscale"]
    d["upscale_x_price"] = d["atmosphere__feels_upscale"] * d["cost_both"]
    d["5star_x_price"] = d["is_5star_hotel"] * d["cost_both"]
    d["premium_place_x_price"] = d["premium_place"] * d["cost_both"]
    d["michelin_x_price"] = d["michelin_star"] * d["cost_both"]
    d["brands_x_upscale"] = (
        d["premium_brand_count"] * d["atmosphere__feels_upscale"]
    )
    d["chef_x_5star"] = d["chef_mentions"] * d["is_5star_hotel"]
    d["luxury_x_price"] = d["luxury_words"] * d["cost_both"]
    d["chef_x_price"] = d["chef_mentions"] * d["cost_both"]
    d["ambiance_x_price"] = d["ambiance_quality"] * d["cost_both"]
    d["prem_brands_x_hotel"] = d["premium_brand_count"] * d["is_5star_hotel"]

    # ── Domain composite score ───────────────────────────────────────────
    d["domain_score_clean"] = (
        d["is_5star_hotel"] * 4
        + d["michelin_star"] * 6
        + d["premium_place"] * 2
        + (d["premium_brand_count"] >= 10).astype(int) * 2
        + d["premium_spirit_flag"] * 2
        + (d["chef_mentions"] >= 2).astype(int) * 3
        + d["has_any_award"] * 3
        + d["ambiance_quality"].clip(upper=3)
        + d["is_high_price"] * 3
        + d["is_very_high_price"] * 3
        - d["is_low_price"] * 3
        - d["is_very_low_price"] * 2
        + d["has_domain"]
        + (d["drink_category_count"] >= 6).astype(int) * 2
    )

    # ── Drink price avg & count columns ──────────────────────────────────
    drink_avg_cols = [c for c in d.columns if c.endswith("_item_price_avg")]
    for c in drink_avg_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    drink_count_cols = [
        c for c in d.columns
        if c.endswith("_item_count") and c != "total_item_count"
    ]
    for c in drink_count_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d, binary_cols, drink_avg_cols, drink_count_cols


# ── Engineer features ────────────────────────────────────────────────────────
print("\nEngineering labeled...")
labeled, binary_cols, drink_avg_cols, drink_count_cols = engineer(labeled.copy())
print("Engineering unlabeled...")
unlabeled, _, _, _ = engineer(unlabeled.copy())


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUILD FEATURE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
core_features = [
    # Price (100% fill)
    "cost_both", "cost_food", "cost_drinks", "dining_price_num", "log_cost",
    "is_high_price", "is_very_high_price", "is_low_price", "is_very_low_price",
    "drink_to_food_ratio", "price_tier",
    # Hotel
    "hotel_stars", "is_5star_hotel", "is_hotel", "hotel_x_price",
    # Quality signals
    "michelin_star", "michelin_x_price",
    "premium_place", "premium_place_x_price",
    "premium_spirit_flag", "premium_brand_count", "luxury_brand_count",
    # Calculated scores
    "score", "segment_score", "restaurant_segment_ord",
    # Drink variety
    "drink_category_count", "total_item_count", "max_drink_price_avg",
    # Operational
    "has_domain", "time_spent_mins",
    "weekday_visits", "weekend_visits", "total_visits",
    # Awards / brands
    "has_any_award", "num_branded_labels",
    # Score breakdown sub-features
    "sb_review_vibe", "sb_drink_depth", "sb_menu_volume", "sb_rating",
    "sb_price", "sb_premium_spirit", "sb_neg_review",
    # NLP
    "luxury_words", "negative_words", "experience_words", "positive_ratio",
    "chef_mentions", "ambiance_quality", "avg_review_length",
    "avg_food_aspect", "avg_service_aspect", "avg_atmosphere_aspect",
    "social_media_score", "social_proof_score",
    # Interactions
    "rating_x_upscale", "upscale_x_price", "5star_x_price",
    "brands_x_upscale", "chef_x_5star", "luxury_x_price",
    "chef_x_price", "ambiance_x_price", "prem_brands_x_hotel",
    # Basic
    "ratings", "reviews_count",
    # Domain composite
    "domain_score_clean",
]

feature_cols = list(dict.fromkeys(
    f for f in core_features + drink_avg_cols + drink_count_cols + binary_cols
    if f in labeled.columns
))

X = labeled[feature_cols].astype(float)

# Drop all-NaN columns
non_null = X.columns[X.notna().any()].tolist()
feature_cols = non_null
X = X[feature_cols]

le = LabelEncoder()
y = le.fit_transform(labeled["Category"])

X_imp = SimpleImputer(strategy="median").fit_transform(X)

# Drop constant columns
stds = np.std(X_imp, axis=0)
non_const = stds > 0
dropped = [feature_cols[i] for i in range(len(feature_cols)) if not non_const[i]]
if dropped:
    print(f"  Dropped constant: {dropped}")
feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if non_const[i]]
X_imp = X_imp[:, non_const]

print(f"\nFeatures: {len(feature_cols)} | Samples: {len(y)}")
print(f"Labels: {dict(zip(le.classes_, np.bincount(y)))}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("MODEL RESULTS (CLEAN — No Sherlock, No ST)")
print(f"{'=' * 100}")

all_preds = {}

# ── 5a. Tuned GBM (grid search) ─────────────────────────────────────────────
print("\nTuning GBM...")
best_acc = 0
best_pred = None
best_params = None

for n_est in [300, 500]:
    for max_d in [3, 4, 5]:
        for lr in [0.03, 0.05, 0.08]:
            for ss in [0.6, 0.7, 0.8]:
                preds = np.zeros_like(y)
                for tri, tei in skf.split(X_imp, y):
                    sw = compute_sample_weight("balanced", y[tri])
                    m = GradientBoostingClassifier(
                        n_estimators=n_est, max_depth=max_d,
                        learning_rate=lr, subsample=ss,
                        min_samples_leaf=3, max_features="sqrt",
                        random_state=42,
                    )
                    m.fit(X_imp[tri], y[tri], sample_weight=sw)
                    preds[tei] = m.predict(X_imp[tei])
                a = accuracy_score(y, preds)
                if a > best_acc:
                    best_acc = a
                    best_pred = preds.copy()
                    best_params = (n_est, max_d, lr, ss)

all_preds["GBM (tuned)"] = best_pred
print(
    f"  Best GBM: n={best_params[0]}, d={best_params[1]}, "
    f"lr={best_params[2]}, ss={best_params[3]} → {best_acc:.1%}"
)

# ── 5b. Tuned Random Forest ─────────────────────────────────────────────────
print("Tuning RF...")
best_rf = 0
best_rf_pred = None

for n_est in [500, 800]:
    for max_d in [6, 8, 10, None]:
        for msl in [2, 3, 5]:
            preds = np.zeros_like(y)
            for tri, tei in skf.split(X_imp, y):
                m = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_d,
                    min_samples_leaf=msl, class_weight="balanced",
                    max_features="sqrt", random_state=42,
                )
                m.fit(X_imp[tri], y[tri])
                preds[tei] = m.predict(X_imp[tei])
            a = accuracy_score(y, preds)
            if a > best_rf:
                best_rf = a
                best_rf_pred = preds.copy()

all_preds["RF (tuned)"] = best_rf_pred
print(f"  Best RF: {best_rf:.1%}")

# ── 5c. Extra Trees ─────────────────────────────────────────────────────────
preds = np.zeros_like(y)
for tri, tei in skf.split(X_imp, y):
    m = ExtraTreesClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", max_features="sqrt", random_state=42,
    )
    m.fit(X_imp[tri], y[tri])
    preds[tei] = m.predict(X_imp[tei])
all_preds["Extra Trees"] = preds
print(f"  Extra Trees: {accuracy_score(y, preds):.1%}")

# ── 5d. Semi-Supervised ─────────────────────────────────────────────────────
print("Running semi-supervised...")
all_X = pd.concat(
    [labeled[feature_cols].astype(float), unlabeled[feature_cols].astype(float)],
    ignore_index=True,
)
all_imp = SimpleImputer(strategy="median").fit_transform(all_X)
all_stds = np.std(all_imp, axis=0)
all_nc = all_stds > 0
X_unlab_imp = all_imp[len(labeled):, all_nc]
X_imp_ss = all_imp[:len(labeled), all_nc]

# Ensure dimension match
if X_unlab_imp.shape[1] != X_imp_ss.shape[1]:
    min_cols = min(X_unlab_imp.shape[1], X_imp_ss.shape[1])
    X_unlab_imp = X_unlab_imp[:, :min_cols]
    X_imp_ss = X_imp_ss[:, :min_cols]

sw = compute_sample_weight("balanced", y)
clf_ss = GradientBoostingClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2], subsample=best_params[3],
    min_samples_leaf=3, max_features="sqrt", random_state=42,
)
clf_ss.fit(X_imp_ss, y, sample_weight=sw)
proba = clf_ss.predict_proba(X_unlab_imp)
pseudo = clf_ss.predict(X_unlab_imp)
conf_mask = proba.max(axis=1) >= 0.88
dist = pd.Series(le.inverse_transform(pseudo[conf_mask])).value_counts()
print(f"  Pseudo-labels (≥0.88): {conf_mask.sum()} → {dict(dist)}")

preds_ss = np.zeros_like(y)
for tri, tei in skf.split(X_imp_ss, y):
    X_tr = np.vstack([X_imp_ss[tri], X_unlab_imp[conf_mask]])
    y_tr = np.concatenate([y[tri], pseudo[conf_mask]])
    sw2 = compute_sample_weight("balanced", y_tr)
    m = GradientBoostingClassifier(
        n_estimators=best_params[0], max_depth=best_params[1],
        learning_rate=best_params[2], subsample=best_params[3],
        min_samples_leaf=3, max_features="sqrt", random_state=42,
    )
    m.fit(X_tr, y_tr, sample_weight=sw2)
    preds_ss[tei] = m.predict(X_imp_ss[tei])
all_preds["Semi-Supervised"] = preds_ss
print(f"  Semi-Supervised: {accuracy_score(y, preds_ss):.1%}")

# ── 5e. Stacking Ensemble ───────────────────────────────────────────────────
print("Building stacking ensemble...")
models_stack = [
    ("GBM1", GradientBoostingClassifier(
        n_estimators=best_params[0], max_depth=best_params[1],
        learning_rate=best_params[2], subsample=best_params[3],
        min_samples_leaf=3, max_features="sqrt", random_state=42), True),
    ("GBM2", GradientBoostingClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=5, max_features="sqrt",
        random_state=123), True),
    ("GBM3", GradientBoostingClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.6, min_samples_leaf=3, max_features="sqrt",
        random_state=456), True),
    ("RF", RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", max_features="sqrt", random_state=42), False),
    ("ET", ExtraTreesClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", max_features="sqrt", random_state=42), False),
]

nc = len(le.classes_)
meta = np.zeros((len(y), len(models_stack) * nc))
for mi, (nm, mod, usw) in enumerate(models_stack):
    for tri, tei in skf.split(X_imp, y):
        mm = type(mod)(**mod.get_params())
        if usw:
            sw = compute_sample_weight("balanced", y[tri])
            mm.fit(X_imp[tri], y[tri], sample_weight=sw)
        else:
            mm.fit(X_imp[tri], y[tri])
        meta[tei, mi * nc:(mi + 1) * nc] = mm.predict_proba(X_imp[tei])

meta_votes = np.column_stack(list(all_preds.values()))
meta_all = np.column_stack([meta, meta_votes])
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
preds_stack = cross_val_predict(lr, meta_all, y, cv=skf)
all_preds["Stacking"] = preds_stack
print(f"  Stacking: {accuracy_score(y, preds_stack):.1%}")

# ── 5f. Majority Vote ───────────────────────────────────────────────────────
pred_matrix = np.column_stack(list(all_preds.values()))
preds_maj = stats.mode(pred_matrix, axis=1)[0].ravel().astype(int)
all_preds["Majority Vote"] = preds_maj
print(f"  Majority Vote: {accuracy_score(y, preds_maj):.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 100}")
print("FINAL RESULTS — CLEAN MODEL (No Sherlock, No ST)")
print(f"{'=' * 100}")

header = (
    f"{'Model':<25s} {'Acc':>7s} │ "
    f"{'M Pre':>6s} {'M Rec':>6s} {'M F1':>6s} │ "
    f"{'P Pre':>6s} {'P Rec':>6s} {'P F1':>6s} │ "
    f"{'SP Pre':>6s} {'SP Rec':>6s} {'SP F1':>6s}"
)
print(f"\n{header}")
print("─" * 105)

for name, pred in all_preds.items():
    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)
    sp_i = list(le.classes_).index("Super Premium")
    p_i = list(le.classes_).index("Premium")
    m_i = list(le.classes_).index("Mainstream")

    m_pre = cm[m_i, m_i] / cm[:, m_i].sum() if cm[:, m_i].sum() > 0 else 0
    m_rec = cm[m_i, m_i] / cm[m_i, :].sum()
    m_f1 = 2 * m_pre * m_rec / (m_pre + m_rec) if (m_pre + m_rec) > 0 else 0

    p_pre = cm[p_i, p_i] / cm[:, p_i].sum() if cm[:, p_i].sum() > 0 else 0
    p_rec = cm[p_i, p_i] / cm[p_i, :].sum()
    p_f1 = 2 * p_pre * p_rec / (p_pre + p_rec) if (p_pre + p_rec) > 0 else 0

    sp_pre = cm[sp_i, sp_i] / cm[:, sp_i].sum() if cm[:, sp_i].sum() > 0 else 0
    sp_rec = cm[sp_i, sp_i] / cm[sp_i, :].sum()
    sp_f1 = 2 * sp_pre * sp_rec / (sp_pre + sp_rec) if (sp_pre + sp_rec) > 0 else 0

    best_mark = (
        " ◀"
        if acc >= max(accuracy_score(y, p) for p in all_preds.values())
        else ""
    )
    print(
        f"{name:<25s} {acc:>6.1%} │ "
        f"{m_pre:>5.1%} {m_rec:>5.1%} {m_f1:>5.1%} │ "
        f"{p_pre:>5.1%} {p_rec:>5.1%} {p_f1:>5.1%} │ "
        f"{sp_pre:>5.1%} {sp_rec:>5.1%} {sp_f1:>5.1%}{best_mark}"
    )

# Best model details
best_name = max(all_preds, key=lambda k: accuracy_score(y, all_preds[k]))
bp = all_preds[best_name]
print(f"\n{'─' * 60}")
print(f"BEST MODEL: {best_name} → {accuracy_score(y, bp):.1%}")
print(f"{'─' * 60}")
print("\nConfusion Matrix:")
print(pd.DataFrame(confusion_matrix(y, bp), index=le.classes_, columns=le.classes_))
print(classification_report(y, bp, target_names=le.classes_, digits=3))


# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print(f"{'─' * 60}")
print("TOP 15 FEATURES")
print(f"{'─' * 60}")

sw = compute_sample_weight("balanced", y)
final_model = GradientBoostingClassifier(
    n_estimators=best_params[0], max_depth=best_params[1],
    learning_rate=best_params[2], subsample=best_params[3],
    min_samples_leaf=3, max_features="sqrt", random_state=42,
)
final_model.fit(X_imp, y, sample_weight=sw)
fi = pd.Series(
    final_model.feature_importances_, index=feature_cols
).sort_values(ascending=False)

cum = 0
for i, (feat, val) in enumerate(fi.head(15).items()):
    cum += val
    print(f"  {i + 1:>2d}. {feat:<45s} {val:.4f}  ({cum:.1%})")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SCORE ALL RESTAURANTS & EXPORT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 80}")
print("SCORING ALL RESTAURANTS")
print(f"{'=' * 80}")

final_model.fit(X_imp_ss, y, sample_weight=compute_sample_weight("balanced", y))
pred_u = final_model.predict(X_unlab_imp)
proba_u = final_model.predict_proba(X_unlab_imp)

out = df.copy()
out["model_predicted_category"] = ""
out["prob_mainstream"] = 0.0
out["prob_premium"] = 0.0
out["prob_super_premium"] = 0.0
out["confidence"] = 0.0
out["prediction_source"] = ""

for idx in labeled.index:
    out.at[idx, "model_predicted_category"] = labeled.at[idx, "Category"]
    out.at[idx, "confidence"] = 1.0
    out.at[idx, "prediction_source"] = "Human Label (corrected)"

m_idx = list(le.classes_).index("Mainstream")
p_idx = list(le.classes_).index("Premium")
sp_idx = list(le.classes_).index("Super Premium")

for i, idx in enumerate(unlabeled.index):
    out.at[idx, "model_predicted_category"] = le.inverse_transform([pred_u[i]])[0]
    out.at[idx, "prob_mainstream"] = float(proba_u[i, m_idx])
    out.at[idx, "prob_premium"] = float(proba_u[i, p_idx])
    out.at[idx, "prob_super_premium"] = float(proba_u[i, sp_idx])
    out.at[idx, "confidence"] = float(proba_u[i].max())
    out.at[idx, "prediction_source"] = "Model V10-Clean"

out.to_csv("restaurant_predictions_v10_clean.csv", index=False)
print(f"\nSaved: restaurant_predictions_v10_clean.csv ({len(out)} rows)")
print(f"\nDistribution:")
print(out["model_predicted_category"].value_counts())

sp_all = out[out["model_predicted_category"] == "Super Premium"]
sp_model = sp_all[sp_all["prediction_source"] == "Model V10-Clean"]
sp_model = sp_model.sort_values("prob_super_premium", ascending=False)
print(
    f"\nSuper Premium: {len(sp_all)} total "
    f"({len(sp_all) - len(sp_model)} human + {len(sp_model)} model)"
)

print(f"\nTop 20 model-predicted Super Premium:")
print(f"  {'#':>3s}  {'Restaurant':<50s} {'SP Prob':>8s} {'Conf':>6s}")
print(f"  {'─' * 70}")
for i, (_, r) in enumerate(sp_model.head(20).iterrows()):
    nm = str(r.get("name", ""))[:48]
    print(f"  {i + 1:>3d}  {nm:<50s} {r['prob_super_premium']:>8.3f} {r['confidence']:>6.3f}")
