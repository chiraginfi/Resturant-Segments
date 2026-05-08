

import numpy as np

import pandas as pd

df = pd.read_csv('/mnt/data/image_recognition/brown_forman_req/new_output/step5_zomato_poi_wbi.csv')

df["cost_final"] = df["cost_for_two_both"].replace(0, np.nan).fillna(df["dining_price"])

cost_norm = (
    (df["cost_final"] - df["cost_final"].min()) /
    (df["cost_final"].max() - df["cost_final"].min())
)

reviews_log = np.log1p(df["reviews_count"])
reviews_norm = (
    (reviews_log - reviews_log.min()) /
    (reviews_log.max() - reviews_log.min())
)

df["score"] = (
    0.35 * cost_norm +
    0.10 * df["offerings__serves_wine"] +
    0.10 * df["amenities__has_bar_onsite"] +
    0.05 * df["planning__accepts_reservations"] +
    0.15 * df["atmosphere__feels_upscale"] +
    0.10 * (df["ratings"] / 5) +
    0.05 * reviews_norm +
    0.05 * df["offerings__has_private_dining_room"]
)

# --- 1. Define premium pincodes ---
premium_pincodes = {
    "400050","400052","400054","400049","400051","400053","400058",
    "400061","400018","400030","400026","400006","400005",
    "400001","400032","400013","400011"
}

# --- 2. Extract pincode from address ---
df["pincode"] = df["address"].str.extract(r'(\b400\d{3}\b)')

# --- 3. Premium flag ---
df["premium_place"] = df["pincode"].isin(premium_pincodes).astype(int)
df.drop(columns=['cost_final'],inplace=True)

df.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step6_data_score_premium_palce.csv", index= False)