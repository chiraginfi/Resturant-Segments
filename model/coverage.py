import pandas as pd

df = pd.read_csv('/mnt/data/image_recognition/brown_forman_req/output/feat_data.csv')

total_rows = len(df)
coverage_summary = []

for column in df.columns:
    non_zero = df[column][(df[column] != 0) & (df[column].notna())]
    coverage_pct = round(len(non_zero) / total_rows * 100, 2)
    coverage_summary.append({'column': column, 'non_zero_count': len(non_zero), 'total': total_rows, 'coverage_%': coverage_pct})

summary_df = pd.DataFrame(coverage_summary)
summary_df.to_csv('/mnt/data/image_recognition/brown_forman_req/output/coverage_summary.csv', index=False)
print(summary_df.to_string(index=False))