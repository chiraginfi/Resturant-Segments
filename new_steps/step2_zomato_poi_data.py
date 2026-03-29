import pandas as pd
import numpy as np
import os
import re


def pivot_categories_to_columns(df):
    print("\nPivoting categories...")

    df['item_price'] = pd.to_numeric(df['price'], errors='coerce')

    df_with_categories = df[df['category'].notna()].copy()

    if len(df_with_categories) == 0:
        return pd.DataFrame(), []

    agg_df = (
        df_with_categories
        .groupby(['poi_code', 'category'], as_index=False)
        .agg(
            brand_name=('brand_name', lambda x: ', '.join(x.dropna().unique())),
            item_price=('item_price', 'mean'),
            item_count=('item_price', 'count')
        )
    )

    brand_pivot = agg_df.pivot(index='poi_code', columns='category', values='brand_name').add_suffix('_brand_name')
    price_pivot = agg_df.pivot(index='poi_code', columns='category', values='item_price').add_suffix('_item_price_avg')
    count_pivot = agg_df.pivot(index='poi_code', columns='category', values='item_count').add_suffix('_item_count')

    result = brand_pivot.join(price_pivot, how='outer').join(count_pivot, how='outer')

    result = result.reset_index()

    return result, agg_df['category'].unique()





def main():
    # -----------------------------
    # PATHS
    # -----------------------------
    base_path = "/mnt/data/image_recognition/brown_forman_req/new_input"
    output_path = "/mnt/data/image_recognition/brown_forman_req/new_output"

    bridge_file_path = os.path.join(base_path, "bridge_file.csv")
    poi_data_path = os.path.join(output_path, "step1_poi_data_club.csv")
    brand_data_path = os.path.join(base_path, "brownforman_brand_items.csv")

    print("Loading data...")

    brdige_file = pd.read_csv(bridge_file_path)
    poi_data = pd.read_csv(poi_data_path)
    zomato_brand_data = pd.read_csv(brand_data_path)

    # -----------------------------
    # MERGE
    # -----------------------------
    print("\nMerging...")

    df = pd.merge(brdige_file, zomato_brand_data, on="filename", how="left")

    if "poicode" in df.columns:
        df = df.rename(columns={"poicode": "poi_code"})

    df = pd.merge(df, poi_data, on="url_hash", how="right")

    print("Merged shape:", df.shape)

    # -----------------------------
    # BRAND DICT
    # -----------------------------
    print("\nCreating brand dict...")

    brand_counts = (
        df.groupby(["poi_code", "brand_name"])
        .size()
        .reset_index(name="count")
    )

    brand_dict = (
        brand_counts.groupby("poi_code")
        .apply(lambda x: dict(zip(x["brand_name"], x["count"])))
        .reset_index(name="brand_count_dict")
    )

    # -----------------------------
    # CATEGORY DICT
    # -----------------------------
    print("Creating category dict...")

    brand_counts_cat = (
        df.groupby(["poi_code", "category", "brand_name"])
        .size()
        .reset_index(name="count")
    )

    brand_dict_cat = (
        brand_counts_cat.groupby("poi_code")
        .apply(lambda x: {
            cat: dict(zip(g["brand_name"], g["count"]))
            for cat, g in x.groupby("category")
        })
        .reset_index(name="brand_category_dict")
    )

    # -----------------------------
    # PIVOT FEATURES
    # -----------------------------
    pivot_df, categories = pivot_categories_to_columns(df)

    print("Pivot shape:", pivot_df.shape)

    # -----------------------------
    # FINAL MERGE (BASE = POI)
    # -----------------------------
    print("\nFinal merging...")

    final_df = poi_data.copy()

    final_df = pd.merge(final_df, brand_dict, on="poi_code", how="left")
    final_df = pd.merge(final_df, brand_dict_cat, on="poi_code", how="left")

    if not pivot_df.empty:
        final_df = pd.merge(final_df, pivot_df, on="poi_code", how="left")

    # -----------------------------
    # CLEANING
    # -----------------------------
    final_df["brand_count_dict"] = final_df["brand_count_dict"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )

    final_df["brand_category_dict"] = final_df["brand_category_dict"].apply(
        lambda x: x if isinstance(x, dict) else {}
    )
    

    # -----------------------------
    # SAVE
    # -----------------------------
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "step2_poi_with_all_features.csv")
    final_df.to_csv(output_file, index=False)

    print("\n✅ DONE")
    print("Final shape:", final_df.shape)
    print("Saved:", output_file)

    # print(final_df['dining_price'].iloc[0])

    return final_df


if __name__ == "__main__":
    main()