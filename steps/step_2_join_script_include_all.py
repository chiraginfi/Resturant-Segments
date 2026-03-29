import pandas as pd
import numpy as np
import os

"""
Restaurant Brand Data Join Script - Include All with Outer Joins

This script joins three CSV files using OUTER joins and pivots categories into columns to create a dataset where:
1. Each row represents a restaurant (poi_code) OR brand item (even if no restaurant match)
2. Each category becomes two columns: {category}_brand_name and {category}_item_price
3. Restaurant location and service information is preserved when available
4. Brand items without restaurant matches are included
5. Restaurants without brand items are included

Key difference from column script: Uses OUTER joins instead of INNER joins to preserve all data.
"""

def load_and_join_data():
    """
    Join three CSV files using OUTER joins and return the merged dataset
    """
    # Define file paths
    base_path = "/mnt/data/image_recognition/brown_forman_req/input"
    brand_items_file = os.path.join(base_path, "brownforman_brand_items.csv")
    places_file = os.path.join(base_path, "zomato_club_poi_data.csv")
    bridge_file = os.path.join(base_path, "final_file_name.csv")
    
    # Load the CSV files
    print("Loading CSV files...")
    brand_items_df = pd.read_csv(brand_items_file)
    places_df = pd.read_csv(places_file)
    bridge_df = pd.read_csv(bridge_file)

    # Build a filename -> name lookup from brand items for fallback restaurant names
    filename_name_lookup = (
        brand_items_df[['filename', 'name']]
        .dropna(subset=['filename'])
        .drop_duplicates(subset=['filename'])
        .rename(columns={'name': 'filename_restaurant_name'})
    )
    
    print(f"Brand items shape: {brand_items_df.shape}")
    print(f"Places shape: {places_df.shape}")
    print(f"Bridge data shape: {bridge_df.shape}")
    
    # Join the data using OUTER joins
    print("\nJoining bridge_data with brand_items on filename (OUTER JOIN)...")
    bridge_brand_join = pd.merge(bridge_df, brand_items_df, on='filename', how='outer')
    print(f"Bridge-Brand outer join shape: {bridge_brand_join.shape}")
    
    # Rename column for consistent joining
    bridge_brand_join = bridge_brand_join.rename(columns={'poicode': 'poi_code'})
    
    # Join with places data using OUTER join
    print("Joining with places data on poi_code (OUTER JOIN)...")
    final_joined = pd.merge(bridge_brand_join, places_df, on='poi_code', how='outer')
    final_joined = pd.merge(final_joined, filename_name_lookup, on='filename', how='left')
    print(f"Final joined shape (outer): {final_joined.shape}")
    
    # Show join statistics
    print("\nJoin statistics:")
    print(f"Records with brand data: {bridge_brand_join['brand_name'].notna().sum()}")
    print(f"Records with restaurant data: {final_joined['name_y'].notna().sum()}")
    print(f"Records with both brand and restaurant: {(final_joined['brand_name'].notna() & final_joined['name_y'].notna()).sum()}")
    print(f"Brand items without restaurant match: {(final_joined['brand_name'].notna() & final_joined['name_y'].isna()).sum()}")
    print(f"Restaurants without brand items: {(final_joined['brand_name'].isna() & final_joined['name_y'].notna()).sum()}")
    
    return final_joined

def pivot_categories_to_columns(df):
    """
    Pivot categories into columns with brand_name and item_price for each category
    Handle missing data gracefully from outer joins
    """
    print("\nPivoting categories to columns...")
    
    # Prepare the data
    df['item_price'] = df['price']
    df['restaurant_name'] = df['name_y']  # from zomato data (may be NaN for brand-only records)
    if 'filename_restaurant_name' in df.columns:
        # Fill missing restaurant names using the brand-items filename -> name mapping.
        df['restaurant_name'] = df['restaurant_name'].fillna(df['filename_restaurant_name'])
    
    # Get unique categories (excluding NaN)
    categories = df['category'].dropna().unique()
    print(f"Found categories: {sorted(categories)}")
    
    # For records with category data, aggregate by poi_code-category:
    #   brand_name  -> comma-joined unique names
    #   item_price  -> average of numeric prices
    #   item_count  -> number of items seen
    df_with_categories = df[df['category'].notna()].copy()
    if len(df_with_categories) > 0:
        # Convert price to numeric so we can average it; non-numeric become NaN
        df_with_categories['item_price'] = pd.to_numeric(
            df_with_categories['item_price'], errors='coerce'
        )

        def join_unique_brands(series):
            vals = series.dropna().unique()
            return ', '.join(str(v) for v in vals) if len(vals) > 0 else np.nan

        agg_df = (
            df_with_categories
            .groupby(['poi_code', 'category'], as_index=False)
            .agg(
                brand_name=('brand_name', join_unique_brands),
                item_price=('item_price', 'mean'),
                item_count=('item_price', 'count'),
            )
        )

        # Round avg price to 2 decimal places where present
        agg_df['item_price'] = agg_df['item_price'].round(2)

        # Create pivot table for brand_name
        brand_pivot = agg_df.pivot(
            index='poi_code',
            columns='category',
            values='brand_name'
        ).add_suffix('_brand_name')

        # Create pivot table for avg item_price
        price_pivot = agg_df.pivot(
            index='poi_code',
            columns='category',
            values='item_price'
        ).add_suffix('_item_price_avg')

        # Create pivot table for item_count
        count_pivot = agg_df.pivot(
            index='poi_code',
            columns='category',
            values='item_count'
        ).add_suffix('_item_count')
    else:
        # No category data found, create empty pivot tables
        brand_pivot = pd.DataFrame()
        price_pivot = pd.DataFrame()
        count_pivot = pd.DataFrame()

    # Get restaurant information (take first occurrence per poi_code)
    # This includes restaurants without brand data due to outer join
    restaurant_info = df.groupby('poi_code').first().reset_index()

    # Select restaurant columns (handle missing columns gracefully)
    restaurant_columns = [
        'poi_code', 'filename', 'restaurant_name', 'address', 'location_status',
        'reviews_count', 'reviews', 'ratings', 'offerings__serves_happy_hour_drinks',
        'offerings__serves_liquor', 'highlights__serves_wine_notable', 'offerings__serves_wine',
        'planning__accepts_reservations', 'offerings__serves_beer', 'offerings__serves_cocktails',
        'cost_for_two'
    ]

    # Filter to existing columns
    available_restaurant_cols = [col for col in restaurant_columns if col in restaurant_info.columns]
    restaurant_df = restaurant_info[available_restaurant_cols]

    # Merge all data together using outer joins to preserve all records
    result = restaurant_df.set_index('poi_code')

    if not brand_pivot.empty:
        result = result.join(brand_pivot, how='outer')
    if not price_pivot.empty:
        result = result.join(price_pivot, how='outer')
    if not count_pivot.empty:
        result = result.join(count_pivot, how='outer')

    # Reset index to make poi_code a column again
    result = result.reset_index()

    # Reorder columns to interleave brand, avg price, and count for each category
    base_columns = [col for col in result.columns if not any(cat in col for cat in categories)]
    category_columns = []

    for category in sorted(categories):
        brand_col = f"{category}_brand_name"
        price_col = f"{category}_item_price_avg"
        count_col = f"{category}_item_count"
        if brand_col in result.columns:
            category_columns.append(brand_col)
        if price_col in result.columns:
            category_columns.append(price_col)
        if count_col in result.columns:
            category_columns.append(count_col)

    # Final column order
    final_columns = base_columns + category_columns
    result = result[final_columns]
    
    return result, categories

def main():
    """
    Main function to execute the outer join and create column-based output
    """
    try:
        # Load and join data using outer joins
        joined_df = load_and_join_data()
        
        # Pivot categories to columns
        result_df, categories = pivot_categories_to_columns(joined_df)
        
        print(f"\nFinal result shape: {result_df.shape}")
        print(f"Available columns: {list(result_df.columns)}")
        
        # Display sample data
        print("\nSample data (first 5 rows):")
        print(result_df.head())
        
        # Display basic statistics
        print(f"\nTotal records: {len(result_df)}")
        print(f"Records with restaurant names: {result_df['restaurant_name'].notna().sum()}")
        print(f"Records with brand data: {sum(result_df[col].notna().sum() for col in result_df.columns if '_brand_name' in col)}")
        print(f"Categories found: {sorted(categories)}")
        
        # Show category coverage statistics
        print("\nCategory coverage by record:")
        for category in sorted(categories):
            brand_col = f"{category}_brand_name"
            if brand_col in result_df.columns:
                non_null_count = result_df[brand_col].notna().sum()
                percentage = (non_null_count / len(result_df)) * 100
                print(f"{category}: {non_null_count}/{len(result_df)} records ({percentage:.1f}%)")
        
        # Show sample data for records with brand data
        print("\nSample records with category data:")
        sample_with_brands = result_df.dropna(subset=[col for col in result_df.columns if '_brand_name' in col], how='all').head(3)
        
        for idx, row in sample_with_brands.iterrows():
            restaurant_name = row['restaurant_name'] if pd.notna(row['restaurant_name']) else "No Restaurant Match"
            print(f"\nRecord: {restaurant_name} (POI: {row['poi_code']})")
            for category in sorted(categories):
                brand_col = f"{category}_brand_name"
                price_col = f"{category}_item_price_avg"
                count_col = f"{category}_item_count"
                if brand_col in result_df.columns and pd.notna(row[brand_col]):
                    brand = row[brand_col]
                    price = row[price_col] if price_col in result_df.columns and pd.notna(row[price_col]) else 'N/A'
                    count = int(row[count_col]) if count_col in result_df.columns and pd.notna(row[count_col]) else 'N/A'
                    print(f"  {category}: {brand} - avg ₹{price} ({count} items)")
        
        # Show sample restaurants without brand data (if any)
        restaurants_without_brands = result_df[
            (result_df['restaurant_name'].notna()) & 
            (result_df[[col for col in result_df.columns if '_brand_name' in col]].isna().all(axis=1))
        ]
        
        if len(restaurants_without_brands) > 0:
            print(f"\nSample restaurants without brand data ({len(restaurants_without_brands)} total):")
            for idx, row in restaurants_without_brands.head(3).iterrows():
                print(f"  {row['restaurant_name']} (POI: {row['poi_code']})")

        # ---------------------------------------------------------------
        # Filter: keep rows that have at least one brand, one item price,
        # or a cost_for_two value.  Rows with none of those AND where
        # offerings__serves_liquor is explicitly False are discarded.
        # ---------------------------------------------------------------
        brand_cols = [col for col in result_df.columns if col.endswith('_brand_name')]
        price_cols = [col for col in result_df.columns if col.endswith('_item_price_avg')]

        has_brand = result_df[brand_cols].notna().any(axis=1) if brand_cols else pd.Series(False, index=result_df.index)
        has_price = result_df[price_cols].notna().any(axis=1) if price_cols else pd.Series(False, index=result_df.index)
        has_cost_for_two = (
            result_df['cost_for_two'].notna() if 'cost_for_two' in result_df.columns
            else pd.Series(False, index=result_df.index)
        )

        has_any_useful_data = has_brand | has_price | has_cost_for_two

        if 'offerings__serves_liquor' in result_df.columns:
            serves_liquor_false = result_df['offerings__serves_liquor'].isin([False, 'False', 'false', 0, '0'])
        else:
            serves_liquor_false = pd.Series(False, index=result_df.index)

        # Discard rows where nothing useful exists AND serves_liquor is False
        discard_mask = (~has_any_useful_data) & serves_liquor_false
        before_filter = len(result_df)
        result_df = result_df[~discard_mask].reset_index(drop=True)
        print(f"\nFiltering: removed {before_filter - len(result_df)} rows with no brand/price/cost_for_two and serves_liquor=False")
        print(f"Rows remaining after filter: {len(result_df)}")

        # Save the result
        output_path = "/mnt/data/image_recognition/brown_forman_req/output"
        os.makedirs(output_path, exist_ok=True)

        # Drop filename from the final saved output after using it for fallback mapping.
        # result_df = result_df.drop(columns=['filename'], errors='ignore')
        
        output_file = os.path.join(output_path, "restaurant_brand_data_include_all.csv")
        result_df.to_csv(output_file, index=False)
        print(f"\nResult saved to: {output_file}")
        
        return result_df
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    result = main()