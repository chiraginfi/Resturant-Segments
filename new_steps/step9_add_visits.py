import pandas as pd

df_left = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step8_data_output.csv")
visits = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/visits_brown_forman.csv")

visits.rename(columns={'poicode':'poi_code'}, inplace=True)
visits['poi_code'] = visits['poi_code'].apply(lambda x: x.split('_')[3])
visits.dropna(subset=['poi_code'],inplace=True)

df_grouped = (
    visits.groupby(['poi_code', 'day_type'])['total_estimated_visit']
    .sum()
    .unstack()
    .reset_index()
)

# optional rename
df_grouped.rename(columns={
    'weekday': 'weekday_visits',
    'weekend': 'weekend_visits'
}, inplace=True)

merged = pd.merge(df_left, df_grouped, how="left",on='poi_code')
print(df_grouped.shape)
print(df_left.shape)
print(merged.shape)
merged.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step9_output.csv", index= False)