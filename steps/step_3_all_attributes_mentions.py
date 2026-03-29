import pandas as pd
import numpy as np

data_all = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/output/restaurant_brand_data_include_all.csv")
df1 = pd.read_csv('/mnt/data/image_recognition/brown_forman_req/input/3k_pois.csv')
df1.rename(columns={'poi_code_primary': 'poi_code'}, inplace=True)

# Standardize the restaurant name column before merging
# so both datasets contribute to a single restaurant_name field.
df1 = df1.rename(columns={'name': 'restaurant_name'}).copy()

# Treat blank strings as missing values so combine_first can fill from either dataset.
df1 = df1.replace(r'^\s*$', pd.NA, regex=True)
df2 = data_all.replace(r'^\s*$', pd.NA, regex=True)


def coalesce_suffix_columns(df):
    df = df.copy()

    for col in list(df.columns):
        if col.endswith('_x'):
            base_col = col[:-2]
            y_col = f'{base_col}_y'

            if y_col in df.columns:
                df[base_col] = df[col].combine_first(df[y_col])
                df = df.drop(columns=[col, y_col])
            else:
                df = df.rename(columns={col: base_col})
        elif col.endswith('_y') and f"{col[:-2]}_x" not in df.columns:
            df = df.rename(columns={col: col[:-2]})

    return df


# Remove any existing _x/_y duplicates inside df2 first.
data_all = coalesce_suffix_columns(data_all)

common_cols = [col for col in df1.columns if col in data_all.columns and col != 'poi_code']

merged_df = df1.merge(data_all, on='poi_code', how='outer', suffixes=('_df1', '_df2'))

# For columns present in both dataframes, keep one final column and
# fill missing values from whichever side has data.
for col in common_cols:
    merged_df[col] = merged_df[f'{col}_df1'].combine_first(merged_df[f'{col}_df2'])
    merged_df = merged_df.drop(columns=[f'{col}_df1', f'{col}_df2'])

# Keep poi_code and restaurant_name near the front for easier review.
front_cols = [col for col in ['poi_code', 'restaurant_name'] if col in merged_df.columns]
other_cols = [col for col in merged_df.columns if col not in front_cols]
merged_df = merged_df[front_cols + other_cols]
data_all = merged_df[
    ~(
        merged_df['filename'].isna() &
        (merged_df['offerings__serves_liquor'] == False)
    )
].copy()


# Parse "₹1,400 for two" -> 1400.0
data_all['cost_for_two_food'] = (
    data_all['cost_for_two']
    .str.replace(r'[₹,]', '', regex=True)   # remove ₹ and commas
    .str.extract(r'(\d+(?:\.\d+)?)')          # grab the number
    [0]
    .astype(float)
)

# ── Drinks cost for two ───────────────────────────────────────────────────────
# Median avg drink price across all available categories × 4
# (2 drinks per person × 2 people). Prices capped at ₹5,000 to exclude
# data-entry outliers (raw data has values like ₹7,940,485).
DRINK_PRICE_AVG_COLS = [c for c in data_all.columns if c.endswith('_item_price_avg')]
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

data_all['cost_for_two_drinks'] = data_all.apply(drinks_cost_for_two, axis=1)

data_all['cost_for_two_both'] = (
    data_all['cost_for_two_food'].fillna(0) +
    data_all['cost_for_two_drinks'].fillna(0)
)

data = pd.read_csv("//mnt/data/image_recognition/brown_forman_req/input/brown_forman_poi_data.csv")
data.rename(columns={"poi_code_primary": "poi_code"}, inplace=True)

final_df = pd.merge(data_all, data,
                    on='poi_code', 
                    how='left')

final_df = final_df.loc[:, ~final_df.columns.duplicated()]

mentions = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/input/browformanbars.csv")

mentions.columns
mentions.rename(columns={" source": "mentions", "hotels":"restaurant_name"}, inplace=True)
mentions.columns

data_all = pd.merge(
    final_df,
    mentions,
    on = "poi_code",
    how="left"
)

def collapse_xy_columns(df):
    x_cols = {c[:-2] for c in df.columns if c.endswith('_x')}
    y_cols = {c[:-2] for c in df.columns if c.endswith('_y')}
    all_xy_bases = x_cols | y_cols

    for base in sorted(all_xy_bases):
        x_col, y_col = f'{base}_x', f'{base}_y'
        if x_col in df.columns and y_col in df.columns:
            df[base] = df[x_col]
            df.drop(columns=[x_col, y_col], inplace=True)
        elif x_col in df.columns:
            df.rename(columns={x_col: base}, inplace=True)
        elif y_col in df.columns:
            df.rename(columns={y_col: base}, inplace=True)
    return df

final_df = collapse_xy_columns(final_df)

data_all = collapse_xy_columns(data_all)

data_all = data_all.dropna(subset=['restaurant_name'])
final_df.to_csv("/mnt/data/image_recognition/brown_forman_req/output/restaurant_brand_data_include_all_attributes.csv", index=False)
data_all.to_csv("/mnt/data/image_recognition/brown_forman_req/output/restaurant_brand_data_include_all_attributes_mentions.csv", index=False)

print(data_all['reviews'].iloc[0])

