from numpy.random import f
import pandas as pd

df_left = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/brown_forman_webisite_hotel_stars_of_host_open_close_hours_left_data.csv")
df_extra = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step7_restaurant__segmented.csv")

new_cols = [col for col in df_left.columns if col not in df_extra.columns]

# always include join key
cols_to_add = ["poi_code"] + new_cols

# merge (left join on df_extra)
df_final = df_extra.merge(
    df_left[cols_to_add],
    on="poi_code",
    how="left"
)
print(df_final.shape)
df_final.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step_7_2_output.csv",index=False)