import pandas as pd
import numpy as np
import os

def analyze_join_data_loss():
    """
    Analyze why the inner join between final_file_name.csv and brownforman_brand_items.csv
    results in fewer records than expected
    """
    
    # Load the files
    base_path = "/mnt/data/image_recognition/brown_forman_req/input"
    final_file_df = pd.read_csv(os.path.join(base_path, "final_file_name.csv"))
    brand_items_df = pd.read_csv(os.path.join(base_path, "brownforman_brand_items.csv"))
    
    print("=== DATA ANALYSIS ===")
    print(f"final_file_name.csv total rows: {len(final_file_df)}")
    print(f"final_file_name.csv unique filenames: {final_file_df['filename'].nunique()}")
    print(f"brownforman_brand_items.csv total rows: {len(brand_items_df)}")
    print(f"brownforman_brand_items.csv unique filenames: {brand_items_df['filename'].nunique()}")
    
    # Get unique filenames from both datasets
    filenames_final = set(final_file_df['filename'].unique())
    filenames_brand = set(brand_items_df['filename'].unique())
    
    print(f"\nUnique filenames in final_file_name.csv: {len(filenames_final)}")
    print(f"Unique filenames in brownforman_brand_items.csv: {len(filenames_brand)}")
    
    # Find intersections and differences
    common_filenames = filenames_final.intersection(filenames_brand)
    only_in_final = filenames_final - filenames_brand
    only_in_brand = filenames_brand - filenames_final
    
    print(f"\nCommon filenames: {len(common_filenames)}")
    print(f"Only in final_file_name.csv: {len(only_in_final)}")
    print(f"Only in brownforman_brand_items.csv: {len(only_in_brand)}")
    
    # Perform inner join to see actual results
    joined_df = pd.merge(final_file_df, brand_items_df, on='filename', how='inner')
    print(f"\nInner join result: {len(joined_df)} rows")
    
    # Show some examples of non-matching filenames
    print(f"\nSample filenames only in final_file_name.csv (first 10):")
    for filename in list(only_in_final)[:10]:
        print(f"  - {filename}")
    
    print(f"\nSample filenames only in brownforman_brand_items.csv (first 10):")
    for filename in list(only_in_brand)[:10]:
        print(f"  - {filename}")
    
    print(f"\nSample common filenames (first 10):")
    for filename in list(common_filenames)[:10]:
        print(f"  - {filename}")
    
    # Check for potential formatting issues
    print("\n=== POTENTIAL FORMATTING ISSUES ===")
    
    # Check for whitespace issues
    filenames_final_stripped = set(final_file_df['filename'].str.strip().unique())
    filenames_brand_stripped = set(brand_items_df['filename'].str.strip().unique())
    
    if len(filenames_final_stripped.intersection(filenames_brand_stripped)) > len(common_filenames):
        print("WARNING: Found whitespace issues in filenames!")
        
    # Check for case sensitivity issues
    filenames_final_lower = set(final_file_df['filename'].str.lower().unique())
    filenames_brand_lower = set(brand_items_df['filename'].str.lower().unique())
    
    common_lower = filenames_final_lower.intersection(filenames_brand_lower)
    if len(common_lower) > len(common_filenames):
        print(f"WARNING: Case sensitivity issues found! {len(common_lower)} matches when ignoring case vs {len(common_filenames)} exact matches")
    
    # Analyze the distribution of records per filename in brand_items
    print(f"\n=== BRAND ITEMS DISTRIBUTION ===")
    filename_counts = brand_items_df['filename'].value_counts()
    print(f"Average items per filename: {filename_counts.mean():.2f}")
    print(f"Min items per filename: {filename_counts.min()}")
    print(f"Max items per filename: {filename_counts.max()}")
    print(f"Median items per filename: {filename_counts.median()}")
    
    # Show top filenames by item count
    print(f"\nTop 10 filenames by item count:")
    for filename, count in filename_counts.head(10).items():
        print(f"  - {filename}: {count} items")
    
    # Create detailed summary
    summary = {
        'final_file_total_rows': len(final_file_df),
        'brand_items_total_rows': len(brand_items_df),
        'final_file_unique_filenames': len(filenames_final),
        'brand_items_unique_filenames': len(filenames_brand),
        'common_filenames': len(common_filenames),
        'only_in_final': len(only_in_final),
        'only_in_brand': len(only_in_brand),
        'inner_join_result_rows': len(joined_df),
        'expected_if_all_matched': len(final_file_df) if len(filenames_final) <= len(filenames_brand) else len(brand_items_df)
    }
    
    print(f"\n=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    return summary, only_in_final, only_in_brand, common_filenames

if __name__ == "__main__":
    summary, only_in_final, only_in_brand, common = analyze_join_data_loss()