import json
import pickle
import warnings
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, DMatrix

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
INPUT_CSV   = BASE_DIR.parent / "model_input" / "training_data_v7.csv"
META_DIR    = BASE_DIR.parent / "model_metadata"
MODEL_PATH  = BASE_DIR / "xgb_model.pkl"

# ── columns ───────────────────────────────────────────────────
PRICE_AVG_COLS = [
    "absinthe_item_price_avg","beer_item_price_avg","brandy_item_price_avg",
    "gin_item_price_avg","martini_item_price_avg","mezcal_item_price_avg",
    "mojito_item_price_avg","old_fashioned_item_price_avg","other_item_price_avg",
    "other_cocktails_item_price_avg","picante_item_price_avg","rum_item_price_avg",
    "soju_item_price_avg","spirits_item_price_avg","tequila_item_price_avg",
    "vodka_item_price_avg","whisky_item_price_avg","wine_item_price_avg",
]

ITEM_COUNT_COLS = [c.replace("_item_price_avg", "_item_count") for c in PRICE_AVG_COLS]
BRAND_COLS = [c.replace("_item_price_avg", "_brand_name") for c in PRICE_AVG_COLS]

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

# ── mention columns (exploded) ────────────────────────────────
MENTION_COLS = [
    "30 Best Bars",
    "Asia's Best 50",
    "CN traveler"
]

# ── brand regex ───────────────────────────────────────────────
import re as _re

PREMIUM_BRANDS = {"grey goose","ketel one","ciroc","tanqueray","hendrick","bombay sapphire","chivas regal","blue label","black label","glenlivet","glenfiddich","woodford","jack daniels","indri","godawan","fratelli","heineken","corona"}

STANDARD_BRANDS = {"bacardi","old monk","kingfisher","tuborg","budweiser","smirnoff","absolut","sula"}

_PREMIUM_RE  = _re.compile("|".join(_re.escape(b) for b in PREMIUM_BRANDS), _re.I)
_STANDARD_RE = _re.compile("|".join(_re.escape(b) for b in STANDARD_BRANDS), _re.I)


def _brand_tier(series):
    s = series.fillna("").astype(str).str.lower()
    premium  = s.str.contains(_PREMIUM_RE)
    standard = (~premium) & s.str.contains(_STANDARD_RE)
    result = pd.Series(0, index=series.index)
    result[premium] = 2
    result[standard] = 1
    return result


# ── feature engineering ───────────────────────────────────────
def engineer_features(df):

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
    # feat["segment_score"] = pd.to_numeric(df.get("segment_score", 0), errors="coerce").fillna(0)
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

    return feat


# ── train ─────────────────────────────────────────────────────
def train():

    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)

    # ── LABEL (Category) ─────────────────────────────
    LABEL_MAP = {
        "mainstream": "Mainstream",
        "premium": "Premium",
        "super premium": "Super Premium",
        "Super Premium":'Super Premium'
    }

    df["label"] = (
        df["Category"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LABEL_MAP)
    )

    df = df[df["label"].notna()].copy()

    print("\nLabel distribution:")
    print(df["label"].value_counts(), "\n")

    # ── FEATURES ─────────────────────────────────────
    X = engineer_features(df)
    feature_names = list(X.columns)

    print(f"Feature matrix: {X.shape}")

    # ── ENCODE ───────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    print("Classes:", list(le.classes_), "\n")

    # ── SPLIT ────────────────────────────────────────
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    # ── MODEL ────────────────────────────────────────
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # ── PREDICTION ───────────────────────────────────
    y_pred = model.predict(X_test)

    # ── OVERRIDES ────────────────────────────────────
    super_premium_idx = le.transform(["Premium"])[0]

    mention_mask = X_test["mention_score"] > 0

    y_pred[mention_mask] = super_premium_idx

    print(f"Overrides → mentions: {mention_mask.sum()}\n")

    # ── CLASSIFICATION REPORT ────────────────────────
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── CONFUSION MATRIX ─────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print("\n=== Confusion Matrix ===")
    print(cm_df, "\n")

    # ── FEATURE IMPORTANCE ───────────────────────────
    importance = pd.Series(model.feature_importances_, index=feature_names)

    importance_df = (
        importance
        .sort_values(ascending=False)
        .reset_index()
    )

    importance_df.columns = ["feature", "importance"]

    print("=== Top 20 Features ===")
    print(importance_df.head(20).to_string(index=False), "\n")

    # ── SHAP (TOP FEATURES PER ROW) ──────────────────
    dtest = DMatrix(X_test, feature_names=feature_names)
    contribs = model.get_booster().predict(dtest, pred_contribs=True)

    n_classes = len(le.classes_)
    n_feat = X_test.shape[1]


    contrib_arr = (contribs[:n_feat].reshape(1, -1) if contribs.ndim == 1 
            else contribs[:, :n_feat] if contribs.ndim == 2 
            else contribs.reshape(len(X_test), len(le.classes_), n_feat + 1)[:, 1, :n_feat])
    abs_contrib = np.abs(contrib_arr)
    top_idx = np.argsort(abs_contrib, axis=1)[:, -5:]

    # ── BUILD OUTPUT ─────────────────────────────────
    test_meta = df.loc[idx_test, ["poi_code", "name", "Category"]].copy()

    test_meta["predicted"] = le.inverse_transform(y_pred)
    test_meta["correct"] = test_meta["Category"] == test_meta["predicted"]

    # top 5 features
    for i in range(5):
        idx = top_idx[:, -(i+1)]
        test_meta[f"top{i+1}_feature"] = [feature_names[j] for j in idx]
        test_meta[f"top{i+1}_shap"] = abs_contrib[np.arange(len(X_test)), idx].round(4)

    # ── SAVE OUTPUTS ─────────────────────────────────
    META_DIR.mkdir(parents=True, exist_ok=True)

    # feature importance
    importance_path = META_DIR / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    # predictions with top features
    test_pred_path = META_DIR / "test_predictions_with_top_features.csv"
    test_meta.to_csv(test_pred_path, index=False)

    # model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": model,
            "label_encoder": le,
            "feature_names": feature_names
        }, f)

    print(f"Saved:")
    print(f"- Feature importance → {importance_path}")
    print(f"- Predictions (with top features) → {test_pred_path}")
    print(f"- Model → {MODEL_PATH}")


if __name__ == "__main__":
    train()