import pandas as pd

df = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/new_output/step3_include_all.csv")

# Strip column name whitespace (carries over from source CSVs)
df.columns = df.columns.str.strip()

# ── 1. Collapse _x / _y duplicates → single clean column ─────────────────────
# Detect all _x/_y pairs dynamically and keep only one (prefer _x value).
x_cols = {c[:-2] for c in df.columns if c.endswith('_x')}
y_cols = {c[:-2] for c in df.columns if c.endswith('_y')}
all_xy_bases = x_cols | y_cols

for base in sorted(all_xy_bases):
    x_col, y_col = f'{base}_x', f'{base}_y'
    if x_col in df.columns and y_col in df.columns:
        df[base] = df[x_col]
        df.drop(columns=[x_col, y_col], inplace=True)
    elif x_col in df.columns:
        df.rename(columns={x_col: base}, inplace=True)
    elif y_col in df.columns:
        df.rename(columns={y_col: base}, inplace=True)

print(f"Collapsed {len(all_xy_bases)} _x/_y pair(s): {sorted(all_xy_bases)}")

# ── 2. Convert all True/False string columns → 1 / 0 ─────────────────────────
def _to_int_flag(val):
    s = str(val).strip().upper()
    if s == 'TRUE':  return 1
    if s == 'FALSE': return 0
    return val  # preserve NaN or unexpected values unchanged

bool_cols = [c for c in df.columns
             if set(df[c].dropna().astype(str).str.strip().str.upper().unique())
                .issubset({'TRUE', 'FALSE'})]

print(f"Converting {len(bool_cols)} boolean columns to 0/1:")
for col in bool_cols:
    print(f"  {col}")
    df[col] = df[col].map(_to_int_flag)

# ── 3. Save ───────────────────────────────────────────────────────────────────
output_path = "/mnt/data/image_recognition/brown_forman_req/new_output/restaurant_brand_data_final.csv"
df.to_csv(output_path, index = False)
print(f"\nDone. Shape: {df.shape}")
print(f"Saved to: {output_path}")
