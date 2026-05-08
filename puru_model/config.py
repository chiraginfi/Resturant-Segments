"""
config.py
────────────────────────────────────────────────────────────────────────────────
Central configuration for the Restaurant Classifier pipeline.
All filepaths, hyperparameters, thresholds, and rule parameters live here.
Edit this file only — nothing else needs to change for a new run or environment.
────────────────────────────────────────────────────────────────────────────────
"""

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

class Paths:
    # ── Input ──────────────────────────────────────────────────────────────
    INPUT_CSV = Path(
        "/mnt/data/image_recognition/brown_forman_req/puru_input/training_data_bf_22.csv"
    )

    # ── Output directory ───────────────────────────────────────────────────
    OUTPUT_DIR = Path(
        "/mnt/data/image_recognition/brown_forman_req/puru_output"
    )

    # ── Output files (auto-derived from OUTPUT_DIR) ────────────────────────
    @classmethod
    def predictions_csv(cls):
        return cls.OUTPUT_DIR / "0327_predictions.csv"

    @classmethod
    def correction_log_csv(cls):
        return cls.OUTPUT_DIR / "0327_label_corrected.csv"

    @classmethod
    def correlation_csv(cls):
        return cls.OUTPUT_DIR / "correlation_analysis.csv"

    @classmethod
    def distribution_csv(cls):
        return cls.OUTPUT_DIR / "distribution_analysis.csv"

    @classmethod
    def correlation_plot(cls):
        return cls.OUTPUT_DIR / "correlation_plot.png"

    @classmethod
    def ensure_output_dir(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA & SPLIT
# ══════════════════════════════════════════════════════════════════════════════

class DataConfig:
    VALID_CATEGORIES  = ["Mainstream", "Premium", "Super Premium"]
    LEAKAGE_COLUMNS   = ["Category - Sherlock", "ST"]

    # Ordinal mapping used for Spearman correlation analysis
    CATEGORY_ORDINAL  = {
        "Mainstream"   : 0,
        "Premium"      : 1,
        "Super Premium": 2,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

class SplitConfig:
    TEST_SIZE    = 0.20   # fraction of labeled data held out as blind test
    RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

class CVConfig:
    N_SPLITS     = 5
    SHUFFLE      = True
    RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# LABEL CORRECTION RULES
# ══════════════════════════════════════════════════════════════════════════════

class LabelCorrectionConfig:
    SUBURBAN_AREAS = [
        "virar", "vasai", "mira road", "nalasopara",
        "dombivli", "kalyan", "ulhasnagar",
    ]

    PREMIUM_AREAS = [
        "bandra", "colaba", "lower parel", "worli", "juhu",
        "kala ghoda", "bkc", "nariman point", "fort",
    ]

    HOTEL_STAR_MAP = {
        "5 stars": 5, "4 stars": 4, "3 stars": 3,
        "2 stars": 2, "1 star" : 1,
    }

    # Fix 1: suburban Premium below this price → Mainstream
    FIX1_MAX_PRICE      = 2000

    # Fix 2: premium-area Mainstream above this price → Premium
    FIX2_MIN_PRICE      = 4000

    # Fix 3: N-star hotel Premium above this price → Super Premium
    FIX3_MIN_PRICE      = 5000
    FIX3_MIN_STARS      = 5


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

class FeatureConfig:
    # Price bin edges for price_tier feature
    PRICE_BINS   = [0, 1000, 2000, 3500, 5500, 8000, 200_000]
    PRICE_LABELS = [0, 1, 2, 3, 4, 5]

    # Thresholds used in binary flag features
    HIGH_PRICE_THRESHOLD      = 5000
    VERY_HIGH_PRICE_THRESHOLD = 8000
    LOW_PRICE_THRESHOLD       = 1500
    VERY_LOW_PRICE_THRESHOLD  = 800

    # domain_score_clean thresholds
    DOMAIN_MIN_PREMIUM_BRANDS    = 10
    DOMAIN_MIN_CHEF_MENTIONS     = 2
    DOMAIN_MIN_DRINK_CATEGORIES  = 6
    DOMAIN_AMBIANCE_CLIP         = 3

    # Premium brand list for luxury_brand_count
    PREMIUM_BRANDS = [
        "grey goose", "belvedere", "absolut", "ketel one",
        "hendricks", "tanqueray", "bombay sapphire", "patron",
        "don julio", "clase azul", "johnny walker", "chivas",
        "glenfiddich", "macallan", "glenlivet", "jack daniel",
        "jameson", "hennessy", "moet", "veuve",
    ]

    # domain_score_clean weights
    DOMAIN_WEIGHTS = {
        "is_5star_hotel"       : +4,
        "michelin_star"        : +6,
        "premium_place"        : +2,
        "premium_brand_gte10"  : +2,
        "premium_spirit_flag"  : +2,
        "chef_mentions_gte2"   : +3,
        "has_any_award"        : +3,
        "is_high_price"        : +3,
        "is_very_high_price"   : +3,
        "is_low_price"         : -3,
        "is_very_low_price"    : -2,
        "has_domain"           : +1,
        "drink_cat_gte6"       : +2,
    }

    USER_IMPORTANT_COLUMNS = ["wbi"]
    USER_REJECT_COLUMNS = ["segment_score"]

# ══════════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

class ModelConfig:
    RANDOM_STATE = 42

    # ── GBM grid search space ──────────────────────────────────────────────
    GBM_GRID = {
        "n_estimators" : [300, 500],
        "max_depth"    : [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample"    : [0.6, 0.7, 0.8],
    }

    # Fixed GBM params (not searched)
    GBM_FIXED = {
        "min_samples_leaf": 3,
        "max_features"    : "sqrt",
    }

    # ── RF grid search space ───────────────────────────────────────────────
    RF_GRID = {
        "n_estimators"    : [500, 800],
        "max_depth"       : [6, 8, 10, None],
        "min_samples_leaf": [2, 3, 5],
    }

    # Fixed RF params
    RF_FIXED = {
        "class_weight": "balanced",
        "max_features": "sqrt",
    }

    # ── Extra Trees (fixed, not searched) ─────────────────────────────────
    ET_PARAMS = {
        "n_estimators"    : 500,
        "max_depth"       : 8,
        "min_samples_leaf": 3,
        "class_weight"    : "balanced",
        "max_features"    : "sqrt",
    }

    # ── Stacking ensemble base models ──────────────────────────────────────
    STACK_GBM2 = {
        "n_estimators"    : 400,
        "max_depth"       : 3,
        "learning_rate"   : 0.08,
        "subsample"       : 0.8,
        "min_samples_leaf": 5,
        "max_features"    : "sqrt",
        "random_state"    : 123,
    }

    STACK_GBM3 = {
        "n_estimators"    : 500,
        "max_depth"       : 5,
        "learning_rate"   : 0.03,
        "subsample"       : 0.6,
        "min_samples_leaf": 3,
        "max_features"    : "sqrt",
        "random_state"    : 456,
    }

    # ── Meta-learner (stacking) ────────────────────────────────────────────
    META_LR = {
        "max_iter"    : 2000,
        "C"           : 1.0,
    }


    KNN_GRID = {
        "n_neighbors": [3, 5, 7, 9],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }
    KNN_FIXED = {
        "algorithm": "auto",
    }

# ══════════════════════════════════════════════════════════════════════════════
# SEMI-SUPERVISED
# ══════════════════════════════════════════════════════════════════════════════

class SemiSupervisedConfig:
    # Minimum prediction confidence to accept a pseudo-label
    CONFIDENCE_THRESHOLD = 0.88


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

class AnalysisConfig:
    # Correlation: minimum |ρ| to flag as reliable given n~336 train samples
    MIN_RELIABLE_RHO     = 0.20

    # Distribution: KS statistic above this → flag as covariate shift
    KS_SHIFT_THRESHOLD   = 0.20

    # Top N features to show in correlation table / plot
    CORR_TOP_N           = 30

    # Overfitting: CV-test gap above this → warn
    OVERFIT_GAP_THRESHOLD = 0.06


# ══════════════════════════════════════════════════════════════════════════════
# PLOT SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

class PlotConfig:
    FIGURE_DPI    = 150
    FIGURE_SIZE   = (14, 10)   # inches (correlation plot)
    STYLE         = "seaborn-v0_8-whitegrid"

    # Bar colours: positive / negative correlation
    COLOR_POSITIVE = "#2563EB"   # blue
    COLOR_NEGATIVE = "#DC2626"   # red
    COLOR_RELIABLE = "#16A34A"   # green tick
    COLOR_WEAK     = "#9CA3AF"   # grey tick

    FONT_FAMILY    = "monospace"


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: flat access to all configs
# ══════════════════════════════════════════════════════════════════════════════

paths       = Paths()
data_cfg    = DataConfig()
split_cfg   = SplitConfig()
cv_cfg      = CVConfig()
lc_cfg      = LabelCorrectionConfig()
feat_cfg    = FeatureConfig()
model_cfg   = ModelConfig()
ss_cfg      = SemiSupervisedConfig()
analysis_cfg= AnalysisConfig()
plot_cfg    = PlotConfig()