import pandas as pd
import numpy as np
import re

OUTPUT_DIR = "/mnt/data/image_recognition/brown_forman_req/new_output/"
cost_for_two = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/zomato_cost_for_two.csv")
poi_zomato_combined = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step2_poi_with_all_features.csv")
mentions = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/browformanbars.csv")

all_pois = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/3kpois.csv")
cost_for_two['poi_code'].dropna(inplace=True)

all_pois['poi_code'] = all_pois['poi_code'].apply(lambda x: x.split('_')[3])




mentions_merged = pd.merge(poi_zomato_combined,mentions,how='left',on='poi_code')
# Step 1: Find poi_codes missing from mentions
missing_pois = all_pois[~all_pois['poi_code'].isin(mentions_merged['poi_code'])]

# Step 2: Keep only mentions' columns from the missing rows
missing_pois_subset = missing_pois[
    [col for col in mentions_merged.columns if col in missing_pois.columns]
]

# Step 3: Append missing rows to mentions
final_df = pd.concat([mentions_merged, missing_pois_subset], ignore_index=True)

final_df = pd.merge(final_df,cost_for_two, how = "left", on='poi_code')
# Parse "₹1,400 for two" -> 1400.0
final_df['cost_for_two_food'] = (
    final_df['cost_for_two']
    .str.replace(r'[₹,]', '', regex=True)   # remove ₹ and commas
    .str.extract(r'(\d+(?:\.\d+)?)')          # grab the number
    [0]
    .astype(float)
)

# ── Drinks cost for two ───────────────────────────────────────────────────────
# Median avg drink price across all available categories × 4
# (2 drinks per person × 2 people). Prices capped at ₹5,000 to exclude
# data-entry outliers (raw data has values like ₹7,940,485).
DRINK_PRICE_AVG_COLS = [c for c in final_df.columns if c.endswith('_item_price_avg')]
MAX_PLAUSIBLE_DRINK_PRICE = 5000

def drinks_cost_for_two(row):
    prices = []
    for col in DRINK_PRICE_AVG_COLS:
        val = row[col]
        if pd.notna(val):
            try:
                p = float(val)
                if 0 < p <= MAX_PLAUSIBLE_DRINK_PRICE:
                    prices.append(p)
            except (ValueError, TypeError):
                pass
    if not prices:
        return np.nan
    median_price = sorted(prices)[len(prices) // 2]
    return round(median_price * 4, 2)   # 2 drinks × 2 people

final_df['cost_for_two_drinks'] = final_df.apply(drinks_cost_for_two, axis=1)

final_df['cost_for_two_both'] = (
    final_df['cost_for_two_food'].fillna(0) +
    final_df['cost_for_two_drinks'].fillna(0)
)
def extract_all_prices(review_text):
    if pd.isna(review_text):
        return []

    # normalize unicode first
    review_text = review_text.replace("\\u20b9", "₹").replace("\\u2013", "-")

    matches = re.findall(r'GUIDED_DINING_PRICE_RANGE=₹?([\d,]+)-([\d,]+)', review_text)

    prices = []
    for low, high in matches:
        low = int(low.replace(",", ""))
        high = int(high.replace(",", ""))
        prices.append((low + high) / 2)

    return prices


def get_avg_price(review_text):
    prices = extract_all_prices(review_text)
    return np.mean(prices) if prices else None

final_df["dining_price"] = final_df["reviews"].apply(get_avg_price)

final_df.to_csv(OUTPUT_DIR+'step3_include_all.csv', index = False)