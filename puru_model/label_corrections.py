"""
label_corrections.py
────────────────────────────────────────────────────────────────────────────────
Rule-based label corrections for restaurant category classification.
Operates ONLY on training data — never call on test/unlabeled data.

Correction rules (derived from domain knowledge, not data-driven):
  Fix 1 : Suburban low-price Premium       → Mainstream
  Fix 2 : High-price premium-area Mainstream → Premium
  Fix 3 : 5-star hotel high-price Premium  → Super Premium

Usage:
    from label_corrections import apply_label_corrections
    labeled_corrected, correction_log = apply_label_corrections(labeled_df)
────────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from typing import Tuple

from config import lc_cfg, Paths, data_cfg


# ── Main function ─────────────────────────────────────────────────────────────

def apply_label_corrections(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply rule-based label corrections to a labeled dataframe.

    Parameters
    ----------
    df      : DataFrame with 'Category', 'address', 'cost_for_two_both',
              'host_hotel_star' columns. Must contain only labeled rows
              (Mainstream / Premium / Super Premium).
    verbose : Print correction summary if True.

    Returns
    -------
    corrected_df    : DataFrame with updated 'Category' column.
    correction_log  : DataFrame logging every changed row with
                      [index, name, original_category, new_category, rule].
    """
    df = df.copy()

    # ── Prep working columns (don't overwrite originals) ──────────────────
    cost  = pd.to_numeric(df["cost_for_two_both"], errors="coerce")
    addr  = df["address"].fillna("").str.lower()
    stars = df["host_hotel_star"].map(lc_cfg.HOTEL_STAR_MAP).fillna(0)

    # ── Build masks ───────────────────────────────────────────────────────
    is_suburban     = addr.apply(lambda x: any(s in x for s in lc_cfg.SUBURBAN_AREAS))
    is_premium_area = addr.apply(lambda x: any(a in x for a in lc_cfg.PREMIUM_AREAS))

    fix1_mask = (
        (df["Category"] == "Premium")
        & is_suburban
        & (cost < lc_cfg.FIX1_MAX_PRICE)
    )
    fix2_mask = (
        (df["Category"] == "Mainstream")
        & (cost >= lc_cfg.FIX2_MIN_PRICE)
        & is_premium_area
    )
    fix3_mask = (
        (df["Category"] == "Premium")
        & (stars == lc_cfg.FIX3_MIN_STARS)
        & (cost >= lc_cfg.FIX3_MIN_PRICE)
    )

    # ── Apply corrections & build log ─────────────────────────────────────
    log_rows = []

    def _log_and_apply(mask, new_cat, rule_name):
        for idx in df.index[mask]:
            log_rows.append({
                "index"             : idx,
                "name"              : df.at[idx, "name"] if "name" in df.columns else "",
                "address"           : df.at[idx, "address"] if "address" in df.columns else "",
                "original_category" : df.at[idx, "Category"],
                "new_category"      : new_cat,
                "cost_for_two_both" : cost.at[idx],
                "hotel_stars"       : stars.at[idx],
                "rule"              : rule_name,
        })
        df.loc[mask, "Category"] = new_cat

    _log_and_apply(
        fix1_mask, "Mainstream",
        f"Fix1: suburban & cost < {lc_cfg.FIX1_MAX_PRICE}"
    )
    _log_and_apply(
        fix2_mask, "Premium",
        f"Fix2: premium area & cost >= {lc_cfg.FIX2_MIN_PRICE}"
    )
    _log_and_apply(
        fix3_mask, "Super Premium",
        f"Fix3: {lc_cfg.FIX3_MIN_STARS}-star hotel & cost >= {lc_cfg.FIX3_MIN_PRICE}"
    )

    correction_log = pd.DataFrame(log_rows) if log_rows else pd.DataFrame(
        columns=["index", "name", "address",
                 "original_category", "new_category",
                 "cost_for_two_both", "hotel_stars", "rule"]
    )

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print(f"{'─' * 60}")
        print("LABEL CORRECTIONS")
        print(f"{'─' * 60}")
        print(f"  Fix 1 (suburban low-price Premium → Mainstream):       {fix1_mask.sum():>3d}")
        print(f"  Fix 2 (premium-area high-price Mainstream → Premium):  {fix2_mask.sum():>3d}")
        print(f"  Fix 3 (5-star hotel high-price Premium → Super Prem):  {fix3_mask.sum():>3d}")
        print(f"  ─────────────────────────────────────────────────────")
        print(f"  Total corrections:                                     {len(correction_log):>3d}")
        print(f"\n  Post-correction distribution:")
        for cat, cnt in df["Category"].value_counts().items():
            print(f"    {cat:<20s}: {cnt}")
        print(f"{'─' * 60}\n")

    return df, correction_log

def apply_no_label_corrections(
    df: pd.DataFrame,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    No-op version: returns dataframe unchanged.
    """
    
    correction_log = pd.DataFrame(
        columns=[
            "index", "name", "address",
            "original_category", "new_category",
            "cost_for_two_both", "hotel_stars", "rule"
        ]
    )

    if verbose:
        print("No label corrections applied.")
        print("\nCategory distribution:")
        print(df["Category"].value_counts())

    return df.copy(), correction_log


# ── Standalone run — saves correction log to CSV ──────────────────────────────

if __name__ == "__main__":
    import sys

    input_csv  = sys.argv[1] if len(sys.argv) > 1 else str(Paths.INPUT_CSV)
    output_csv = sys.argv[2] if len(sys.argv) > 2 else str(Paths.correction_log_csv())

    print(f"Loading: {input_csv}")
    df_raw = pd.read_csv(input_csv, low_memory=False)

    for col in data_cfg.LEAKAGE_COLUMNS:
        if col in df_raw.columns:
            df_raw = df_raw.drop(columns=[col])
            print(f"  Dropped leakage column: {col}")

    df_raw["Category"] = df_raw["Category"].str.strip().str.title()
    labeled_only = df_raw[df_raw["Category"].isin(data_cfg.VALID_CATEGORIES)].copy()

    print(f"Labeled rows: {len(labeled_only)}")
    print(f"Distribution before: {dict(labeled_only['Category'].value_counts())}")

    corrected, log = apply_label_corrections(labeled_only)
    log.to_csv(output_csv, index=False)
    print(f"\nCorrection log saved → {output_csv}  ({len(log)} rows changed)")