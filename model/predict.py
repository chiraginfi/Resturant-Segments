import json
import pickle
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import DMatrix

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "xgb_model.pkl"

# ── columns ─────────────────────────────────────────

PRICE_AVG_COLS = [
    "absinthe_item_price_avg","beer_item_price_avg","brandy_item_price_avg",
    "gin_item_price_avg","martini_item_price_avg","mezcal_item_price_avg",
    "mojito_item_price_avg","old_fashioned_item_price_avg","other_item_price_avg",
    "other_cocktails_item_price_avg","picante_item_price_avg","rum_item_price_avg",
    "soju_item_price_avg","spirits_item_price_avg","tequila_item_price_avg",
    "vodka_item_price_avg","whisky_item_price_avg","wine_item_price_avg",
]

ITEM_COUNT_COLS = [c.replace("_item_price_avg", "_item_count") for c in PRICE_AVG_COLS]
BRAND_COLS      = [c.replace("_item_price_avg", "_brand_name") for c in PRICE_AVG_COLS]

BOOL_COLS = [
    "atmosphere__feels_upscale","atmosphere__feels_romantic","atmosphere__feels_quiet",
    "atmosphere__feels_cozy","atmosphere__feels_hip","atmosphere__feels_casual",
    "atmosphere__is_recently_popular",
    "offerings__serves_wine","offerings__serves_happy_hour_drinks",
    "offerings__serves_happy_hour_food","offerings__serves_late_night_food",
    "offerings__has_dancing","offerings__has_private_dining_room",
    "offerings__serves_liquor",
    "highlights__has_live_music","highlights__has_seating_rooftop",
    "highlights__has_bar_games","highlights__has_karaoke_nights",
    "highlights__has_fast_service",
    "parking__has_parking_valet",
    "planning__requires_reservations","planning__accepts_reservations",
    "planning__recommends_reservations_dinner",
    "planning__recommends_reservations_lunch",
    "planning__recommends_reservations_brunch",
    "amenities__has_bar_onsite",
    "premium_spirit_flag",
]

MENTION_COLS = ["30 Best Bars","Asia's Best 50","CN traveler"]

# ── brand regex ─────────────────────────────────────

PREMIUM_BRANDS = {"grey goose","ketel one","ciroc","tanqueray","hendrick","bombay sapphire","chivas regal","blue label","black label","glenlivet","glenfiddich","woodford","jack daniels","indri","godawan","fratelli","heineken","corona"}
STANDARD_BRANDS = {"bacardi","old monk","kingfisher","tuborg","budweiser","smirnoff","absolut","sula"}

_PREMIUM_RE  = re.compile("|".join(re.escape(b) for b in PREMIUM_BRANDS), re.I)
_STANDARD_RE = re.compile("|".join(re.escape(b) for b in STANDARD_BRANDS), re.I)

def _brand_tier(series):
    s = series.fillna("").astype(str).str.lower()
    premium  = s.str.contains(_PREMIUM_RE)
    standard = (~premium) & s.str.contains(_STANDARD_RE)
    result = pd.Series(0, index=series.index)
    result[premium] = 2
    result[standard] = 1
    return result

# ── feature engineering ─────────────────────────────

def _engineer_features(df, feature_names):

    feat = pd.DataFrame(index=df.index)

    # -------------------- NUMERIC --------------------
    feat["ratings"] = pd.to_numeric(df["ratings"], errors="coerce").fillna(3)

    reviews = pd.to_numeric(df.get("reviews_count"), errors="coerce")
    min_reviews = reviews.min()
    if np.isnan(min_reviews):
        min_reviews = 0
    feat["reviews_count"] = reviews.fillna(min_reviews)

    cost_both = pd.to_numeric(df.get("cost_for_two_both"), errors="coerce")
    dining = pd.to_numeric(df.get("dining_price"), errors="coerce")
    feat["cost_for_two_numeric"] = cost_both.fillna(dining).fillna(0)

    # -------------------- BASE FEATURES --------------------
    feat["max_drink_price_avg"] = pd.to_numeric(df["max_drink_price_avg"], errors="coerce").fillna(0)
    feat["total_item_count"] = pd.to_numeric(df["total_item_count"], errors="coerce").fillna(0)
    feat["drink_category_count"] = pd.to_numeric(df["drink_category_count"], errors="coerce").fillna(0)

    # -------------------- IMPORTANT SIGNALS --------------------
    feat["premium_place"] = pd.to_numeric(df.get("premium_place", 0), errors="coerce").fillna(0)
    feat["segment_score"] = pd.to_numeric(df.get("segment_score", 0), errors="coerce").fillna(0)
    feat["score"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)

    wbi = pd.to_numeric(df.get("wbi"), errors="coerce")
    feat["wbi"] = wbi.fillna(wbi.median())

    feat["premium_brand_count_input"] = pd.to_numeric(
        df.get("premium_brand_count", 0), errors="coerce"
    ).fillna(0)

    # -------------------- NEW FEATURES --------------------

    # --- typical_time_spent → minutes ---
    def parse_time_to_minutes(x):
        if pd.isna(x):
            return np.nan
        
        x = str(x).lower()
        hours = re.search(r"(\d+\.?\d*)\s*hour", x)
        mins  = re.search(r"(\d+)\s*min", x)
        
        total = 0
        if hours:
            total += float(hours.group(1)) * 60
        if mins:
            total += float(mins.group(1))
        
        return total if total > 0 else np.nan

    time_spent = df.get("typical_time_spent").apply(parse_time_to_minutes)
    feat["typical_time_spent"] = time_spent.fillna(time_spent.median())

    # --- host_hotel_star ---
    def parse_star(x):
        if pd.isna(x):
            return 0
        x = str(x).lower()
        match = re.search(r"(\d)", x)
        if match:
            return int(match.group(1))
        return 0

    feat["host_hotel_star"] = df.get("host_hotel_star").apply(parse_star)

    # derived feature (VERY IMPORTANT)
    feat["is_in_luxury_hotel"] = (feat["host_hotel_star"] >= 5).astype(int)

    # --- has_domain & michelin ---
    feat["has_domain"] = pd.to_numeric(df.get("has_domain", 0), errors="coerce").fillna(0)
    feat["michelin_star"] = pd.to_numeric(df.get("michelin_star", 0), errors="coerce").fillna(0)

    # location premium boost
    feat["location_premium_score"] = (
        feat["is_in_luxury_hotel"] * 2 +
        feat["michelin_star"] * 3
    )

    # -------------------- PRICE FEATURES --------------------
    for col in PRICE_AVG_COLS:
        feat[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    price_matrix = feat[PRICE_AVG_COLS].replace(0, np.nan)
    feat["avg_price_across_categories"] = price_matrix.mean(axis=1).fillna(0)
    feat["max_price_across_categories"] = price_matrix.max(axis=1).fillna(0)

    # -------------------- ITEM COUNTS --------------------
    for col in ITEM_COUNT_COLS:
        feat[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # -------------------- BOOL --------------------
    for col in BOOL_COLS:
        feat[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    # -------------------- BRAND SCORING --------------------
    premium_score = pd.Series(0, index=df.index)
    for col in BRAND_COLS:
        if col in df.columns:
            premium_score += _brand_tier(df[col])

    feat["premium_brand_score"] = premium_score

    # -------------------- MENTION FEATURES --------------------
    for col in MENTION_COLS:
        feat[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

    feat["mention_score"] = (
        feat["Asia's Best 50"] * 3 +
        feat["CN traveler"] * 2 +
        feat["30 Best Bars"] * 2
    )

    # -------------------- INTERACTIONS --------------------
    feat["premium_density"] = feat["premium_brand_score"] / (feat["drink_category_count"] + 1)
    feat["menu_richness"] = feat["total_item_count"] * feat["drink_category_count"]

    return feat[feature_names]

# ── load model ──────────────────────────────────────

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ── prediction ──────────────────────────────────────

def predict_dataframe(df, bundle):

    feature_names = bundle["feature_names"]
    model = bundle["model"]
    le = bundle["label_encoder"]

    X = _engineer_features(df.copy(), feature_names)

    preds = model.predict(X)
    proba = model.predict_proba(X)

    # ── MENTION MASK (override) ──────────────────────
    super_idx = list(le.classes_).index("Super Premium")

    mention_mask = X["mention_score"] > 0

    preds = preds.copy()
    preds[mention_mask] = super_idx

    # ── SHAP ─────────────────────────────────────────
    from xgboost import DMatrix

    dtest = DMatrix(X, feature_names=feature_names)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)

    n_classes = len(le.classes_)
    n_feat = len(feature_names)

    contrib_arr = (contribs[:n_feat].reshape(1, -1) if contribs.ndim == 1 
               else contribs[:, :n_feat] if contribs.ndim == 2 
               else contribs.reshape(len(X), n_classes, n_feat + 1)[:, 1, :n_feat])
    abs_contrib = np.abs(contrib_arr)

    top_idx = np.argsort(abs_contrib, axis=1)

    # ── OUTPUT ───────────────────────────────────────
    out = df.copy()

    out["predicted_label"] = le.inverse_transform(preds)

    out["prediction_source"] = np.where(
        mention_mask,
        "mention_override",
        "model"
    )

    out["predicted_proba"] = [
        ", ".join(f"{c}={p:.3f}" for c, p in zip(le.classes_, row))
        for row in proba
    ]

    # ── TOP FEATURES ─────────────────────────────────
    for i in range(5):
        idx = top_idx[:, -(i+1)]
        out[f"top{i+1}_feature"] = [feature_names[j] for j in idx]
        out[f"top{i+1}_shap"] = abs_contrib[np.arange(len(X)), idx].round(4)

    return out

# ── CLI ─────────────────────────────────────────────

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    bundle = load_model()

    df = pd.read_csv(args.input)
    result = predict_dataframe(df, bundle)

    out_path = args.output or args.input.replace(".csv", "_predictions.csv")
    result.to_csv(out_path, index=False)

    print("Saved →", out_path)

if __name__ == "__main__":
    main()