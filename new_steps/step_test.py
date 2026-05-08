import pandas as pd

df_poi = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/dfpoidata.csv")
df_h3 = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/df_h3index.csv")

df_po_h3 = pd.merge(df_poi,df_h3,how="left",on="poi_code")

df_po_h3.to_csv("/mnt/data/image_recognition/brown_forman_req/new_input/input_macthed_poi.csv", index = False)