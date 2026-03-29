import pandas as pd

# Load the column-based output
df = pd.read_csv("/mnt/data/image_recognition/brown_forman_req/output/restaurant_brand_data_columns.csv")

print("=== COLUMN-BASED DATA STRUCTURE ===\n")

# Show first restaurant with all their available brands/prices
print("Sample Restaurant Data:")
print("-" * 50)

# Find a restaurant that has multiple categories
sample_restaurant = df.dropna(subset=[col for col in df.columns if '_brand_name' in col], how='all').iloc[0]

print(f"Restaurant: {sample_restaurant['restaurant_name']}")
print(f"POI Code: {sample_restaurant['poi_code']}")
print(f"Address: {sample_restaurant['address'][:50]}...")
print(f"Cost for two: {sample_restaurant['cost_for_two']}")
print("\nAvailable Brands & Prices:")

# Extract category data
categories = []
for col in df.columns:
    if '_brand_name' in col:
        category = col.replace('_brand_name', '')
        categories.append(category)

for category in sorted(categories):
    brand_col = f"{category}_brand_name"
    price_col = f"{category}_item_price"
    
    brand = sample_restaurant[brand_col] if pd.notna(sample_restaurant[brand_col]) else None
    price = sample_restaurant[price_col] if pd.notna(sample_restaurant[price_col]) else None
    
    if brand:
        print(f"  • {category.title()}: {brand} - ₹{price}")

print(f"\n=== COLUMN SUMMARY ===")
print(f"Total columns: {len(df.columns)}")
print(f"Restaurant info columns: {len([col for col in df.columns if '_brand_name' not in col and '_item_price' not in col])}")
print(f"Category columns: {len([col for col in df.columns if '_brand_name' in col or '_item_price' in col])}")

print(f"\nCategory columns (each has brand_name and item_price):")
for category in sorted(categories):
    print(f"  • {category}_brand_name, {category}_item_price")

print(f"\n=== DATA COMPARISON ===")
print("BEFORE (Row-wise): Each row = one restaurant-brand-item combination")
print("AFTER (Column-wise): Each row = one restaurant with all brands/prices as columns")
print(f"\nRows reduced from ~97K to {len(df)} restaurants")
print(f"Columns expanded to {len(df.columns)} to accommodate all categories")