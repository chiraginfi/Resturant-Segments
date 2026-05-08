"""
feature_engineering.py
────────────────────────────────────────────────────────────────────────────────
Feature engineering + diagnostic analyses for restaurant classifier.
All thresholds and parameters are read from config.py.

Exports
-------
engineer(df)
    Full feature engineering pipeline.

build_feature_matrix(labeled, feature_cols)
    Impute, drop constants → (X_imp, feature_cols, le, y, imputer).

run_correlation_analysis(...)
    Spearman ρ vs. ordinal target. Saves CSV + PNG plot.

run_distribution_analysis(...)
    KS-test labeled vs. unlabeled. Saves CSV.
────────────────────────────────────────────────────────────────────────────────
"""

import re
import ast
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for all envs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from config import feat_cfg, analysis_cfg, plot_cfg, data_cfg, Paths


# ══════════════════════════════════════════════════════════════════════════════
# 1. FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _parse_time(t):
    if pd.isna(t):
        return np.nan
    t = str(t).lower()
    h = re.findall(r"([\d.]+)\s*hour", t)
    m = re.findall(r"([\d.]+)\s*min", t)
    mins = 0
    if h: mins += float(h[0]) * 60
    if m: mins += float(m[0])
    return mins if mins > 0 else np.nan


def _count_premium_brands(bcd):
    if pd.isna(bcd) or bcd == "{}":
        return 0
    try:
        parsed = ast.literal_eval(str(bcd))
        return sum(
            1 for b in parsed
            if any(pb in b.lower() for pb in feat_cfg.PREMIUM_BRANDS)
        )
    except Exception:
        return 0


def _parse_score_breakdown(s):
    if pd.isna(s):
        return {}
    parts, feats = str(s).split("|"), {}
    for p in parts:
        p = p.strip()
        m = re.search(r":([+-]?\d+)", p)
        if m:
            val = int(m.group(1))
            if   "review_vibe"    in p: feats["sb_review_vibe"]    = val
            elif "drink_category" in p: feats["sb_drink_depth"]    = val
            elif "menu_volume"    in p: feats["sb_menu_volume"]    = val
            elif "rating"         in p: feats["sb_rating"]         = val
            elif "price" in p and "kw" not in p: feats["sb_price"] = val
            elif "premium_spirit" in p: feats["sb_premium_spirit"] = val
            elif "neg_review"     in p: feats["sb_neg_review"]     = val
    return feats


def _nlp_features(r):
    r_str, t = str(r), str(r).lower()
    f = {}
    for aspect in ["FOOD_ASPECT", "SERVICE_ASPECT", "ATMOSPHERE_ASPECT"]:
        scores = [int(x) for x in re.findall(rf"{aspect}=(\d)", r_str)]
        f[f"avg_{aspect.lower()}"] = np.mean(scores) if scores else np.nan

    f["luxury_words"]     = sum(t.count(w) for w in [
        "luxury","elegant","exquisite","sophisticated","fine dining",
        "premium","exceptional","world-class","michelin","gourmet",
        "exclusive","impeccable","refined",
    ])
    f["negative_words"]   = sum(t.count(w) for w in [
        "disappointing","poor","bad","worst","terrible",
        "overpriced","mediocre","bland",
    ])
    f["experience_words"] = sum(t.count(w) for w in [
        "experience","ambience","cocktail","sommelier","curated",
        "crafted","signature","tasting","pairing","plating","presentation",
    ])
    tot = f["luxury_words"] + f["negative_words"]
    f["positive_ratio"]   = f["luxury_words"] / tot if tot > 0 else 0.5
    f["chef_mentions"]    = sum(t.count(w) for w in [
        "chef","tasting menu","omakase","culinary","executive chef",
    ])
    f["ambiance_quality"] = sum(t.count(w) for w in [
        "stunning","breathtaking","gorgeous","panoramic","intimate",
        "plush","lavish","opulent","magnificent",
    ])
    rts = re.findall(r"review=([^}]+)", r_str)
    f["avg_review_length"]  = np.mean([len(x) for x in rts]) if rts else 0
    f["social_media_score"] = sum(t.count(w) for w in [
        "instagram","insta","photo","selfie","aesthetic",
        "share","post","tag","reel",
    ])
    f["social_proof_score"] = sum(t.count(w) for w in [
        "recommend","must try","birthday","anniversary","celebration",
        "reservation","booked","regular","viral","famous","popular",
        "celebrity","influencer",
    ])
    return pd.Series(f)


# ══════════════════════════════════════════════════════════════════════════════
# 2. MAIN ENGINEER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def engineer(d: pd.DataFrame):
    """
    Full feature engineering pipeline. All thresholds from config.feat_cfg.
    Returns (d, binary_cols, drink_avg_cols, drink_count_cols).
    """
    # ── Price ─────────────────────────────────────────────────────────────
    for col in ["cost_for_two_food","cost_for_two_drinks",
                "cost_for_two_both","dining_price"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    d["cost_both"]           = d["cost_for_two_both"]
    d["cost_food"]           = d["cost_for_two_food"]
    d["cost_drinks"]         = d["cost_for_two_drinks"]
    d["is_high_price"]       = (d["cost_both"] >= feat_cfg.HIGH_PRICE_THRESHOLD).astype(float)
    d["is_very_high_price"]  = (d["cost_both"] >= feat_cfg.VERY_HIGH_PRICE_THRESHOLD).astype(float)
    d["is_low_price"]        = (d["cost_both"] <= feat_cfg.LOW_PRICE_THRESHOLD).astype(float)
    d["is_very_low_price"]   = (d["cost_both"] <= feat_cfg.VERY_LOW_PRICE_THRESHOLD).astype(float)
    d["drink_to_food_ratio"] = d["cost_drinks"] / (d["cost_food"] + 1)
    d["dining_price_num"]    = pd.to_numeric(d["dining_price"], errors="coerce")
    d["price_tier"]          = pd.cut(
        d["cost_both"],
        bins=feat_cfg.PRICE_BINS,
        labels=feat_cfg.PRICE_LABELS,
    ).astype(float)
    d["log_cost"] = np.log1p(d["cost_both"])

    # ── Hotel ─────────────────────────────────────────────────────────────
    d["hotel_stars"]    = d["host_hotel_star"].map(
        {"5 stars":5,"4 stars":4,"3 stars":3,"2 stars":2,"1 star":1}
    ).fillna(0)
    d["is_5star_hotel"] = (d["hotel_stars"] == 5).astype(int)
    d["is_hotel"]       = (d["hotel_stars"] > 0).astype(int)
    d["hotel_x_price"]  = d["hotel_stars"] * d["cost_both"]

    # ── Michelin / premium flags ──────────────────────────────────────────
    d["michelin_star"]       = pd.to_numeric(d["michelin_star"], errors="coerce").fillna(0)
    d["premium_place"]       = pd.to_numeric(d["premium_place"], errors="coerce").fillna(0)
    d["premium_spirit_flag"] = (
        d["premium_spirit_flag"]
        .map({True:1,False:0,"True":1,"False":0})
        .fillna(0)
    )
    d["premium_brand_count"] = pd.to_numeric(d["premium_brand_count"], errors="coerce").fillna(0)
    d["score"]               = pd.to_numeric(d["score"], errors="coerce")

    # ── Drink variety ─────────────────────────────────────────────────────
    d["drink_category_count"] = pd.to_numeric(d["drink_category_count"], errors="coerce").fillna(0)
    d["total_item_count"]     = pd.to_numeric(d["total_item_count"], errors="coerce").fillna(0)
    d["max_drink_price_avg"]  = pd.to_numeric(d["max_drink_price_avg"], errors="coerce")
    d["has_domain"]           = pd.to_numeric(d["has_domain"], errors="coerce").fillna(0)

    # ── Visit data ────────────────────────────────────────────────────────
    d["weekday_visits"] = pd.to_numeric(d["weekday_visits"], errors="coerce")
    d["weekend_visits"] = pd.to_numeric(d["weekend_visits"], errors="coerce")
    d["total_visits"]   = d["weekday_visits"].fillna(0) + d["weekend_visits"].fillna(0)

    # ── Time spent ────────────────────────────────────────────────────────
    d["time_spent_mins"] = d["typical_time_spent"].apply(_parse_time)

    # ── Existing calculated fields ────────────────────────────────────────
    if "segment_score" in d.columns:
        d["segment_score"]          = pd.to_numeric(d["segment_score"], errors="coerce")
    d["restaurant_segment_ord"] = d["restaurant_segment"].map(
        {"Budget":0,"Mid":1,"Premium":2,"Luxury":3}
    )
    d["has_any_award"]     = d["mentions"].notna().astype(int)
    brand_cols = [c for c in d.columns if c.endswith("_brand_name")]
    d["num_branded_labels"] = d[brand_cols].apply(
        lambda r: sum(1 for v in r if pd.notna(v) and v != "other"), axis=1
    )

    # ── Score breakdown ───────────────────────────────────────────────────
    sb = d["score_breakdown"].apply(_parse_score_breakdown).apply(pd.Series)
    for c in sb.columns:
        d[c] = sb[c]

    # ── Luxury brand count ────────────────────────────────────────────────
    d["luxury_brand_count"] = d["brand_count_dict"].apply(_count_premium_brands)

    # ── NLP ───────────────────────────────────────────────────────────────
    print("  NLP extraction...")
    nf = d["reviews"].apply(_nlp_features)
    for c in nf.columns:
        d[c] = nf[c]

    # ── Binary flag columns ───────────────────────────────────────────────
    binary_cols = [
        c for c in d.columns
        if any(c.startswith(p) for p in [
            "offerings__","service_options__","highlights__",
            "parking__","planning__","atmosphere__","amenities__",
        ])
        and c != "offerings__serves_liquor"
    ]

    # ── Interaction features ──────────────────────────────────────────────
    d["rating_x_upscale"]      = d["ratings"] * d["atmosphere__feels_upscale"]
    d["upscale_x_price"]       = d["atmosphere__feels_upscale"] * d["cost_both"]
    d["5star_x_price"]         = d["is_5star_hotel"] * d["cost_both"]
    d["premium_place_x_price"] = d["premium_place"] * d["cost_both"]
    d["michelin_x_price"]      = d["michelin_star"] * d["cost_both"]
    d["brands_x_upscale"]      = d["premium_brand_count"] * d["atmosphere__feels_upscale"]
    d["chef_x_5star"]          = d["chef_mentions"] * d["is_5star_hotel"]
    d["luxury_x_price"]        = d["luxury_words"] * d["cost_both"]
    d["chef_x_price"]          = d["chef_mentions"] * d["cost_both"]
    d["ambiance_x_price"]      = d["ambiance_quality"] * d["cost_both"]
    d["prem_brands_x_hotel"]   = d["premium_brand_count"] * d["is_5star_hotel"]

    # ── Domain composite score (weights from config) ──────────────────────
    w = feat_cfg.DOMAIN_WEIGHTS
    d["domain_score_clean"] = (
          d["is_5star_hotel"]  * w["is_5star_hotel"]
        + d["michelin_star"]   * w["michelin_star"]
        + d["premium_place"]   * w["premium_place"]
        + (d["premium_brand_count"] >= feat_cfg.DOMAIN_MIN_PREMIUM_BRANDS).astype(int) * w["premium_brand_gte10"]
        + d["premium_spirit_flag"]  * w["premium_spirit_flag"]
        + (d["chef_mentions"] >= feat_cfg.DOMAIN_MIN_CHEF_MENTIONS).astype(int)        * w["chef_mentions_gte2"]
        + d["has_any_award"]   * w["has_any_award"]
        + d["ambiance_quality"].clip(upper=feat_cfg.DOMAIN_AMBIANCE_CLIP)
        + d["is_high_price"]   * w["is_high_price"]
        + d["is_very_high_price"] * w["is_very_high_price"]
        - d["is_low_price"]    * abs(w["is_low_price"])
        - d["is_very_low_price"] * abs(w["is_very_low_price"])
        + d["has_domain"]      * w["has_domain"]
        + (d["drink_category_count"] >= feat_cfg.DOMAIN_MIN_DRINK_CATEGORIES).astype(int) * w["drink_cat_gte6"]
    )

    # ── Drink price / count columns ───────────────────────────────────────
    drink_avg_cols   = [c for c in d.columns if c.endswith("_item_price_avg")]
    drink_count_cols = [c for c in d.columns
                        if c.endswith("_item_count") and c != "total_item_count"]
    for c in drink_avg_cols + drink_count_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    return d, binary_cols, drink_avg_cols, drink_count_cols


# ── Core feature list ─────────────────────────────────────────────────────────
CORE_FEATURES = [
    "cost_both","cost_food","cost_drinks","dining_price_num","log_cost",
    "is_high_price","is_very_high_price","is_low_price","is_very_low_price",
    "drink_to_food_ratio","price_tier",
    "hotel_stars","is_5star_hotel","is_hotel","hotel_x_price",
    "michelin_star","michelin_x_price",
    "premium_place","premium_place_x_price",
    "premium_spirit_flag","premium_brand_count","luxury_brand_count",
    "score","segment_score","restaurant_segment_ord",
    "drink_category_count","total_item_count","max_drink_price_avg",
    "has_domain","time_spent_mins",
    "weekday_visits","weekend_visits","total_visits",
    "has_any_award","num_branded_labels",
    "sb_review_vibe","sb_drink_depth","sb_menu_volume","sb_rating",
    "sb_price","sb_premium_spirit","sb_neg_review",
    "luxury_words","negative_words","experience_words","positive_ratio",
    "chef_mentions","ambiance_quality","avg_review_length",
    "avg_food_aspect","avg_service_aspect","avg_atmosphere_aspect",
    "social_media_score","social_proof_score",
    "rating_x_upscale","upscale_x_price","5star_x_price",
    "brands_x_upscale","chef_x_5star","luxury_x_price",
    "chef_x_price","ambiance_x_price","prem_brands_x_hotel",
    "ratings","reviews_count",
    "domain_score_clean",
]

# def select_features(train_eng, feature_cols, corr_threshold=0.9):

#     corr_matrix = train_eng[feature_cols].corr().abs()
#     upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     to_drop = [
#         column for column in upper.columns
#         if any(upper[column] > corr_threshold) and column not in feat_cfg.USER_IMPORTANT_COLUMNS
#     ]
#     feature_cols_reduced = [
#         f for f in feature_cols
#         if f not in to_drop or f in feat_cfg.USER_IMPORTANT_COLUMNS
#     ]
#     return feature_cols_reduced
def select_features(train_eng, feature_cols, corr_threshold=0.9):
    corr_matrix = train_eng[feature_cols].corr().abs()

    # Save correlation matrix as a heatmap
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plot_path = Paths.OUTPUT_DIR / "feature_correlation_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation matrix plot: {plot_path}")

    # Feature selection logic
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [
        column for column in upper.columns
        if any(upper[column] > corr_threshold) and column not in feat_cfg.USER_IMPORTANT_COLUMNS
    ]
    feature_cols_reduced = [
        f for f in feature_cols
        if f not in to_drop or f in feat_cfg.USER_IMPORTANT_COLUMNS
    ]
    return feature_cols_reduced


def build_feature_matrix(labeled: pd.DataFrame, feature_cols: list):
    X            = labeled[feature_cols].astype(float)
    non_null     = X.columns[X.notna().any()].tolist()
    feature_cols = non_null
    X            = X[feature_cols]

    le = LabelEncoder()
    y  = le.fit_transform(labeled["Category"])

    # First pass: identify constant columns
    imputer_tmp = SimpleImputer(strategy="median")
    X_imp_tmp   = imputer_tmp.fit_transform(X)

    stds      = np.std(X_imp_tmp, axis=0)
    non_const = stds > 0
    dropped   = [feature_cols[i] for i in range(len(feature_cols)) if not non_const[i]]
    if dropped:
        print(f"  Dropped constant columns: {dropped}")

    # Drop constants from feature list and refit imputer on clean columns only
    feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if non_const[i]]
    X_clean      = labeled[feature_cols].astype(float)

    imputer = SimpleImputer(strategy="median")
    X_imp   = imputer.fit_transform(X_clean)

    return X_imp, feature_cols, le, y, imputer


# ══════════════════════════════════════════════════════════════════════════════
# 3. CORRELATION ANALYSIS + PLOT
# ══════════════════════════════════════════════════════════════════════════════

def run_correlation_analysis(
    labeled: pd.DataFrame,
    feature_cols: list,
    y: np.ndarray,
    le: LabelEncoder,
    X_imp: np.ndarray,
    top_n: int = None,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    Spearman ρ of every feature vs. ordinal target.
    Saves correlation_analysis.csv and correlation_plot.png.
    Returns DataFrame sorted by |ρ| descending.
    """
    top_n      = top_n      or analysis_cfg.CORR_TOP_N
    output_dir = output_dir or str(Paths.OUTPUT_DIR)

    print(f"\n{'═' * 70}")
    print("CORRELATION ANALYSIS — Spearman ρ vs. Target (ordinal)")
    print(f"{'═' * 70}")
    print(f"  n={len(y)} train samples  |  reliable threshold: |ρ| ≥ {analysis_cfg.MIN_RELIABLE_RHO}\n")

    # Build ordinal target (Mainstream=0, Premium=1, Super Premium=2)
    ordinal_map = {}
    for cat, ord_val in data_cfg.CATEGORY_ORDINAL.items():
        enc = le.transform([cat])[0]
        ordinal_map[enc] = ord_val
    y_ord = np.array([ordinal_map[v] for v in y])

    rows = []
    for i, feat in enumerate(feature_cols):
        x = X_imp[:, i]
        if np.std(x) == 0:
            continue
        rho, pval = stats.spearmanr(x, y_ord)
        rows.append({
            "feature"     : feat,
            "spearman_rho": round(rho, 4),
            "abs_rho"     : round(abs(rho), 4),
            "p_value"     : round(pval, 4),
            "significant" : pval < 0.05,
            "reliable"    : abs(rho) >= analysis_cfg.MIN_RELIABLE_RHO,
        })

    corr_df = pd.DataFrame(rows).sort_values("abs_rho", ascending=False)

    # ── Print table ───────────────────────────────────────────────────────
    print(f"  {'#':>3s}  {'Feature':<45s} {'ρ':>7s}  {'p':>7s}  Reliable")
    print(f"  {'─' * 72}")
    for rank, (_, row) in enumerate(corr_df.head(top_n).iterrows(), 1):
        rel = "✓" if row["reliable"] else "·"
        sig = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "  ")
        print(
            f"  {rank:>3d}  {row['feature']:<45s} "
            f"{row['spearman_rho']:>+7.4f}  {row['p_value']:>7.4f}  {rel} {sig}"
        )

    reliable_count = corr_df["reliable"].sum()
    print(f"\n  Reliable features (|ρ| ≥ {analysis_cfg.MIN_RELIABLE_RHO}): "
          f"{reliable_count} / {len(corr_df)}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = f"{output_dir}/correlation_analysis.csv"
    corr_df.to_csv(csv_path, index=False)
    print(f"  CSV saved  → {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_path = _plot_correlation(corr_df, top_n, output_dir)
    print(f"  Plot saved → {plot_path}")

    return corr_df


def _plot_correlation(corr_df: pd.DataFrame, top_n: int, output_dir: str) -> str:
    """
    Horizontal bar chart: top-N features by |Spearman ρ|.
      Blue  = positive correlation (higher value → higher tier)
      Red   = negative correlation
      Green border = reliable  |ρ| ≥ threshold
      Grey border  = weak / unreliable
    Dashed vertical lines mark the ± reliability threshold.
    """
    plot_data = corr_df.head(top_n).sort_values("spearman_rho", ascending=True)
    n         = len(plot_data)
    fig_h     = max(8, n * 0.38)

    fig, ax = plt.subplots(figsize=(plot_cfg.FIGURE_SIZE[0], fig_h),
                           dpi=plot_cfg.FIGURE_DPI)
    try:
        plt.style.use(plot_cfg.STYLE)
    except Exception:
        pass

    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    colours = [
        plot_cfg.COLOR_POSITIVE if v >= 0 else plot_cfg.COLOR_NEGATIVE
        for v in plot_data["spearman_rho"]
    ]

    bars = ax.barh(range(n), plot_data["spearman_rho"],
                   color=colours, height=0.65, zorder=3)

    # Outline colour encodes reliability
    for i, (_, row) in enumerate(plot_data.iterrows()):
        bars[i].set_edgecolor(
            plot_cfg.COLOR_RELIABLE if row["reliable"] else plot_cfg.COLOR_WEAK
        )
        bars[i].set_linewidth(1.8 if row["reliable"] else 0.6)

    # Inline value labels
    for i, (_, row) in enumerate(plot_data.iterrows()):
        rho  = row["spearman_rho"]
        xpos = rho + (0.005 if rho >= 0 else -0.005)
        ha   = "left" if rho >= 0 else "right"
        sig  = "**" if row["p_value"] < 0.01 else ("*" if row["p_value"] < 0.05 else "")
        ax.text(xpos, i, f"{rho:+.3f}{sig}",
                va="center", ha=ha, fontsize=8,
                fontfamily=plot_cfg.FONT_FAMILY, color="#1E293B")

    ax.set_yticks(range(n))
    ax.set_yticklabels(plot_data["feature"], fontsize=9,
                       fontfamily=plot_cfg.FONT_FAMILY)

    x_max = max(abs(plot_data["spearman_rho"].max()),
                abs(plot_data["spearman_rho"].min())) + 0.12
    ax.set_xlim(-x_max, x_max)
    ax.axvline(0, color="#64748B", linewidth=0.8, zorder=2)
    ax.axvline( analysis_cfg.MIN_RELIABLE_RHO, color="#16A34A",
                linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)
    ax.axvline(-analysis_cfg.MIN_RELIABLE_RHO, color="#DC2626",
                linewidth=1.0, linestyle="--", alpha=0.7, zorder=2)

    ax.set_xlabel(
        "Spearman ρ  (Mainstream=0 · Premium=1 · Super Premium=2)",
        fontsize=10, fontfamily=plot_cfg.FONT_FAMILY, color="#334155",
    )
    ax.set_title(
        f"Feature Correlation vs. Restaurant Tier  |  Top {top_n} by |ρ|",
        fontsize=13, fontweight="bold",
        fontfamily=plot_cfg.FONT_FAMILY, color="#0F172A", pad=14,
    )

    ax.grid(axis="x", color="#CBD5E1", linewidth=0.5, zorder=0)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="x", colors="#475569", labelsize=9)
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        mpatches.Patch(facecolor=plot_cfg.COLOR_POSITIVE,
                       label="Positive  (higher value → higher tier)"),
        mpatches.Patch(facecolor=plot_cfg.COLOR_NEGATIVE,
                       label="Negative  (higher value → lower tier)"),
        mpatches.Patch(facecolor="white", edgecolor=plot_cfg.COLOR_RELIABLE,
                       linewidth=1.8,
                       label=f"Reliable  |ρ| ≥ {analysis_cfg.MIN_RELIABLE_RHO}"),
        mpatches.Patch(facecolor="white", edgecolor=plot_cfg.COLOR_WEAK,
                       linewidth=0.6, label="Weak / unreliable"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=8.5, framealpha=0.92, edgecolor="#CBD5E1")

    fig.text(
        0.01, 0.004,
        f"* p<0.05   ** p<0.01   |   Dashed lines at ±{analysis_cfg.MIN_RELIABLE_RHO} "
        f"(reliability threshold for n≈336)",
        fontsize=7.5, color="#64748B", fontfamily=plot_cfg.FONT_FAMILY,
    )

    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plot_path = f"{output_dir}/correlation_plot.png"
    fig.savefig(plot_path, dpi=plot_cfg.FIGURE_DPI,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return plot_path


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISTRIBUTION ANALYSIS (Covariate Shift)
# ══════════════════════════════════════════════════════════════════════════════

def run_distribution_analysis(
    labeled: pd.DataFrame,
    unlabeled: pd.DataFrame,
    feature_cols: list,
    imputer,
    output_dir: str = None,
) -> pd.DataFrame:
    """
    KS-test for covariate shift: labeled vs. unlabeled per feature.
    Also computes per-class mean/std for labeled data.
    Flags features where KS statistic > analysis_cfg.KS_SHIFT_THRESHOLD.
    Saves distribution_analysis.csv.
    """
    output_dir = output_dir or str(Paths.OUTPUT_DIR)

    print(f"\n{'═' * 70}")
    print("DISTRIBUTION ANALYSIS — Labeled vs. Unlabeled (Covariate Shift)")
    print(f"{'═' * 70}")
    print(f"  Labeled: {len(labeled)} | Unlabeled: {len(unlabeled)}")
    print(f"  KS shift flag threshold: {analysis_cfg.KS_SHIFT_THRESHOLD}\n")

    valid_cols = [f for f in feature_cols
                  if f in labeled.columns and f in unlabeled.columns]

    X_lab   = imputer.transform(labeled[valid_cols].astype(float))
    X_unlab = imputer.transform(unlabeled[valid_cols].astype(float))

    classes = ["Mainstream", "Premium", "Super Premium"]
    rows    = []

    for i, feat in enumerate(valid_cols):
        x_lab, x_unlab = X_lab[:, i], X_unlab[:, i]
        ks_stat, ks_pval = stats.ks_2samp(x_lab, x_unlab)

        row = {
            "feature"      : feat,
            "ks_statistic" : round(ks_stat, 4),
            "ks_pvalue"    : round(ks_pval, 4),
            "shift_flagged": ks_stat > analysis_cfg.KS_SHIFT_THRESHOLD,
            "lab_mean"     : round(float(np.mean(x_lab)), 4),
            "lab_std"      : round(float(np.std(x_lab)),  4),
            "unlab_mean"   : round(float(np.mean(x_unlab)), 4),
            "unlab_std"    : round(float(np.std(x_unlab)),  4),
            "mean_diff_pct": round(
                abs(np.mean(x_lab) - np.mean(x_unlab)) /
                (abs(np.mean(x_lab)) + 1e-9) * 100, 1
            ),
        }
        for cls in classes:
            mask  = labeled["Category"] == cls
            x_cls = X_lab[mask.values, i]
            key   = cls.lower().replace(" ", "_")
            row[f"{key}_mean"] = round(float(np.mean(x_cls)), 4) if len(x_cls) else np.nan
            row[f"{key}_std"]  = round(float(np.std(x_cls)),  4) if len(x_cls) else np.nan
        rows.append(row)

    dist_df = pd.DataFrame(rows).sort_values("ks_statistic", ascending=False)
    flagged = dist_df[dist_df["shift_flagged"]]

    print(f"  Features with KS > {analysis_cfg.KS_SHIFT_THRESHOLD}: "
          f"{len(flagged)} / {len(dist_df)}\n")
    print(f"  {'Feature':<44s} {'KS':>6s}  {'p':>7s}  "
          f"{'Lab μ':>8s}  {'Unlab μ':>8s}  {'Δ%':>6s}  Flag")
    print(f"  {'─' * 93}")
    for _, row in dist_df.head(25).iterrows():
        flag = "⚠ SHIFT" if row["shift_flagged"] else ""
        print(
            f"  {row['feature']:<44s} "
            f"{row['ks_statistic']:>6.3f}  {row['ks_pvalue']:>7.4f}  "
            f"{row['lab_mean']:>8.3f}  {row['unlab_mean']:>8.3f}  "
            f"{row['mean_diff_pct']:>5.1f}%  {flag}"
        )

    print(f"\n  {'─' * 70}")
    print(f"  PER-CLASS MEANS — Top 15 features by KS")
    print(f"  {'─' * 70}")
    print(f"  {'Feature':<40s} {'Mainstream':>12s} {'Premium':>10s} "
          f"{'SuperPrem':>10s} {'Unlabeled':>10s}")
    print(f"  {'─' * 78}")
    for _, row in dist_df.head(15).iterrows():
        print(
            f"  {row['feature']:<40s} "
            f"{row['mainstream_mean']:>12.3f} "
            f"{row['premium_mean']:>10.3f} "
            f"{row['super_premium_mean']:>10.3f} "
            f"{row['unlab_mean']:>10.3f}"
        )

    out_path = f"{output_dir}/distribution_analysis.csv"
    dist_df.to_csv(out_path, index=False)
    print(f"\n  CSV saved → {out_path}")

    return dist_df