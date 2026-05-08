import pandas as pd
OUTPUT_DIR = "/mnt/data/image_recognition/brown_forman_req/new_output/"
matched = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/input_macthed_poi.csv")
unmatched = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/unmatched_88.csv")
unmatched_url = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_input/unmatched_927_merged.csv")

unmatched_url = unmatched_url[['placeid_url', 'url_hash']]
unmatched_url.rename(columns = {'placeid_url':'poi_code'}, inplace=True)

matched['poi_code'] = matched['poi_code'].apply(lambda x: x.split('_')[3])


unmatched_data_url = pd.merge(unmatched,unmatched_url, how='inner',on='poi_code')

matched_cols = set(matched.columns)
unmatched_cols = set(unmatched_data_url.columns)

# columns only in matched
only_in_matched = matched_cols - unmatched_cols

# columns only in unmatched
only_in_unmatched = unmatched_cols - matched_cols

print("Only in matched:\n", only_in_matched)
print("\nOnly in unmatched:\n", only_in_unmatched)

for col in unmatched_data_url.columns:
    if col not in matched.columns:
        matched[col] = None

# concat
df_final = pd.concat(
    [matched[unmatched_data_url.columns], unmatched_data_url],
    ignore_index=True
)


print(df_final.shape)
df_final.to_csv(OUTPUT_DIR+"step1_poi_data_club.csv",index=False)
