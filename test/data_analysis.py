import pandas as pd
import os

def analyze_data_loss():
    """
    Detailed analysis of data loss during joins
    """
    # Define file paths
    base_path = "/mnt/data/image_recognition/brown_forman_req/input"
    brand_items_file = os.path.join(base_path, "brownforman_brand_items.csv")
    places_file = os.path.join(base_path, "zomato_club_poi_data.csv")
    bridge_file = os.path.join(base_path, "final_file_name.csv")
    
    # Load the CSV files
    print("=== LOADING DATA ===")
    brand_items_df = pd.read_csv(brand_items_file)
    places_df = pd.read_csv(places_file)
    bridge_df = pd.read_csv(bridge_file)
    
    print(f"1. Brand items (brownforman_brand_items.csv): {brand_items_df.shape[0]:,} records")
    print(f"2. Places (zomato_club_poi_data.csv): {places_df.shape[0]:,} records") 
    print(f"3. Bridge (final_file_name.csv): {bridge_df.shape[0]:,} records")
    
    # Analyze unique values in key columns
    print("\n=== UNIQUE VALUES ANALYSIS ===")
    print(f"Unique filenames in brand_items: {brand_items_df['filename'].nunique():,}")
    print(f"Unique filenames in bridge: {bridge_df['filename'].nunique():,}")
    print(f"Unique poi_codes in places: {places_df['poi_code'].nunique():,}")
    print(f"Unique poi_codes in bridge: {bridge_df['poicode'].nunique():,}")
    
    # Check for null/empty values in key joining columns
    print("\n=== NULL/EMPTY VALUES CHECK ===")
    print(f"Null filenames in brand_items: {brand_items_df['filename'].isnull().sum()}")
    print(f"Null filenames in bridge: {bridge_df['filename'].isnull().sum()}")
    print(f"Null poi_codes in places: {places_df['poi_code'].isnull().sum()}")
    print(f"Null poi_codes in bridge: {bridge_df['poicode'].isnull().sum()}")
    print(f"Empty poi_codes in bridge: {(bridge_df['poicode'] == '').sum()}")
    
    # Step 1: First join analysis
    print("\n=== STEP 1: BRIDGE + BRAND_ITEMS JOIN ===")
    bridge_brand_join = pd.merge(bridge_df, brand_items_df, on='filename', how='inner')
    print(f"Result: {bridge_brand_join.shape[0]:,} records")
    
    # Check what's missing in first join
    bridge_filenames = set(bridge_df['filename'])
    brand_filenames = set(brand_items_df['filename'])
    missing_in_brand = bridge_filenames - brand_filenames
    missing_in_bridge = brand_filenames - bridge_filenames
    
    print(f"Bridge filenames not in brand_items: {len(missing_in_brand):,}")
    print(f"Brand_items filenames not in bridge: {len(missing_in_bridge):,}")
    
    if missing_in_brand:
        print("Sample missing bridge filenames:", list(missing_in_brand)[:5])
    if missing_in_bridge:
        print("Sample missing brand filenames:", list(missing_in_bridge)[:5])
    
    # Step 2: Second join analysis
    print("\n=== STEP 2: RESULT + PLACES JOIN ===")
    bridge_brand_join['poi_code'] = bridge_brand_join['poicode']  # Rename for consistency
    
    final_joined = pd.merge(bridge_brand_join, places_df, on='poi_code', how='inner')
    print(f"Final result: {final_joined.shape[0]:,} records")
    print(f"Records lost in step 2: {bridge_brand_join.shape[0] - final_joined.shape[0]:,}")
    
    # Check what's missing in second join
    bridge_poi_codes = set(bridge_brand_join['poi_code'].dropna())
    places_poi_codes = set(places_df['poi_code'].dropna())
    missing_places = bridge_poi_codes - places_poi_codes
    
    print(f"Bridge poi_codes not found in places: {len(missing_places):,}")
    if missing_places:
        print("Sample missing poi_codes:", list(missing_places)[:5])
    
    # Analyze the missing poi_codes
    print("\n=== MISSING POI_CODE ANALYSIS ===")
    missing_records = bridge_brand_join[bridge_brand_join['poi_code'].isin(missing_places)]
    if not missing_records.empty:
        print(f"Records with missing poi_codes: {len(missing_records):,}")
        print("Sample records that couldn't be joined:")
        print(missing_records[['filename', 'poi_code', 'cost_for_two']].head())
        
        # Check if these are empty poi_codes
        empty_poi_codes = missing_records['poi_code'].isin(['', pd.NA, None]).sum()
        print(f"Empty/null poi_codes in missing records: {empty_poi_codes}")
    
    # Summary statistics
    print("\n=== FINAL SUMMARY ===")
    print(f"Total unique restaurants in final dataset: {final_joined['poi_code'].nunique():,}")
    print(f"Total unique brands in final dataset: {final_joined['brand_name'].nunique():,}")
    print(f"Total brand-restaurant combinations: {final_joined.shape[0]:,}")
    
    # Show brand distribution
    print("\n=== BRAND DISTRIBUTION ===")
    brand_counts = final_joined['brand_name'].value_counts()
    print(f"Top 10 brands by item count:")
    print(brand_counts.head(10))
    
    return final_joined

if __name__ == "__main__":
    result = analyze_data_loss()