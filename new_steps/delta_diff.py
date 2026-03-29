import pandas as pd

# Load data
old_pred = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/model_input/predict_data_v6_predictions.csv")
new_pred = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/model_input/predict_data_v7_predictions.csv")

# # FIX: comma missing
# old_pred = old_pred[['poi_code', 'predicted_label']]
# new_pred = new_pred[['poi_code', 'predicted_label']]

old_pred = old_pred.rename(columns={'predicted_label': 'old_pred'})
new_pred = new_pred.rename(columns={'predicted_label': 'new_pred'})

# Merge (keep full old_pred data + new_pred column)
df = old_pred.merge(
    new_pred[['poi_code', 'new_pred']],
    on='poi_code',
    how='outer'
)

# Change flag
df['changed'] = df['old_pred'] != df['new_pred']

# Filter only changed rows
delta_df = df[df['changed']].copy()

# Optional: sort for readability
delta_df = delta_df.sort_values(['old_pred', 'new_pred'])

delta_df.to_csv("/mnt/data/image_recognition/brown_forman_req/new_output/delta_diff.csv", index = False)