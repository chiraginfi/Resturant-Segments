import pandas as pd

data = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/restaurant_brand_data_final.csv")

wbi = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/brown_forman_wbi.csv")

wbi = wbi[['poi_code','poi_wbi','home_h9']]

wbi['poi_code'] = wbi['poi_code'].apply(lambda x: x.split('_')[3])
wbi = (
    wbi.groupby('poi_code', as_index=False)['poi_wbi']
    .median()
    .rename(columns={'poi_wbi': 'wbi'})
)

df = pd.merge(data,wbi,how='left',on='poi_code')

print(df.columns)
print(df.shape)

df.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step5_zomato_poi_wbi.csv", index=False)