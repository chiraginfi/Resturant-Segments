import pandas as pd

df1 = pd.read_csv('/mnt/data/image_recognition/brown_forman_req/input/3k_pois.csv')

df2 = pd.read_csv('/mnt/data/image_recognition/brown_forman_req/input/Brown Forman_ Attribute_3k_pois.csv')

df2['poi_code_primary'] = df2['poi_code'].str.split('_').str[3]

liquor_poi_codes = set(
    df2.loc[
        df2['offerings__serves_liquor'].astype(str).str.strip().str.upper().eq('TRUE'),
        'poi_code_primary',
    ]
)

updated_mask = df1['poi_code_primary'].isin(liquor_poi_codes)
df1.loc[updated_mask, 'offerings__serves_liquor'] = True

df1.to_csv('/mnt/data/image_recognition/brown_forman_req/input/3k_pois.csv', index=False)

print(f"Updated {updated_mask.sum()} rows in df1.")

