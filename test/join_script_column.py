import pandas as pd
import numpy as np
import os

"""
Restaurant Brand Data Join Script - Column Format

This script joins three CSV files and pivots categories into columns to create a dataset where:
1. Each row represents a restaurant (poi_code)
2. Each category becomes two columns: {category}_brand_name and {category}_item_price
3. Restaurant location and service information is preserved

Example output columns:
- poi_code, restaurant_name, address, etc.
- wine_brand_name, wine_item_price
- beer_brand_name, beer_item_price
- cocktails_brand_name, cocktails_item_price
- etc.
"""

def load_and_join_data():
    """
    Join three CSV files and return the merged dataset
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
    
    print(f"Brand items shape: {brand_items_df.shape}")
    print(f"Places shape: {places_df.shape}")
    print(f"Bridge data shape: {bridge_df.shape}")
    
    # Join the data
    print("\nJoining bridge_data with brand_items on filename...")
    bridge_brand_join = pd.merge(bridge_df, brand_items_df, on='filename', how='inner')
    print(f"Bridge-Brand join shape: {bridge_brand_join.shape}")
    
    # Rename column for consistent joining
    bridge_brand_join = bridge_brand_join.rename(columns={'poicode': 'poi_code'})
    
    # Join with places data
    print("Joining with places data on poi_code...")
    final_joined = pd.merge(bridge_brand_join, places_df, on='poi_code', how='inner')
    print(f"Final joined shape: {final_joined.shape}")
    
    return final_joined

def pivot_categories_to_columns(df):
    """
    Pivot categories into columns with brand_name and item_price for each category
    """
    print("\nPivoting categories to columns...")
    
    # Prepare the data
    df['item_price'] = df['price']
    df['restaurant_name'] = df['name_y']  # from zomato data
    
    # Get unique categories
    categories = df['category'].unique()
    print(f"Found categories: {sorted(categories)}")
    
    # For each poi_code-category combination, take the first occurrence
    # (in case there are multiple items per restaurant per category)
    df_deduplicated = df.groupby(['poi_code', 'category']).first().reset_index()
    
    # Create pivot table for brand_name
    brand_pivot = df_deduplicated.pivot(
        index='poi_code', 
        columns='category', 
        values='brand_name'
    ).add_suffix('_brand_name')
    
    # Create pivot table for item_price
    price_pivot = df_deduplicated.pivot(
        index='poi_code', 
        columns='category', 
        values='item_price'
    ).add_suffix('_item_price')
    
    # Get restaurant information (take first occurrence per poi_code)
    restaurant_info = df.groupby('poi_code').first().reset_index()
    
    # Select restaurant columns
    restaurant_columns = [
        'poi_code', 'restaurant_name', 'address', 'location_status', 
        'reviews_count', 'reviews', 'ratings', 'offerings__serves_happy_hour_drinks',
        'offerings__serves_liquor', 'highlights__serves_wine_notable', 'offerings__serves_wine',
        'planning__accepts_reservations', 'offerings__serves_beer', 'offerings__serves_cocktails',
        'cost_for_two'
    ]
    
    # Filter to existing columns
    available_restaurant_cols = [col for col in restaurant_columns if col in restaurant_info.columns]
    restaurant_df = restaurant_info[available_restaurant_cols]
    
    # Merge all data together
    result = restaurant_df.set_index('poi_code')
    result = result.join(brand_pivot, how='left')
    result = result.join(price_pivot, how='left')
    
    # Reset index to make poi_code a column again
    result = result.reset_index()
    
    # Reorder columns to interleave brand and price columns for each category
    base_columns = [col for col in result.columns if not any(cat in col for cat in categories)]
    category_columns = []
    
    for category in sorted(categories):
        brand_col = f"{category}_brand_name"
        price_col = f"{category}_item_price"
        if brand_col in result.columns:
            category_columns.append(brand_col)
        if price_col in result.columns:
            category_columns.append(price_col)
    
    # Final column order
    final_columns = base_columns + category_columns
    result = result[final_columns]
    
    return result, categories

def main():
    """
    Main function to execute the join and create column-based output
    """
    try:
        # Load and join data
        joined_df = load_and_join_data()
        
        # Pivot categories to columns
        result_df, categories = pivot_categories_to_columns(joined_df)
        
        print(f"\nFinal result shape: {result_df.shape}")
        print(f"Available columns: {list(result_df.columns)}")
        
        # Display sample data
        print("\nSample data:")
        print(result_df.head())
        
        # Display basic statistics
        print(f"\nTotal restaurants: {len(result_df)}")
        print(f"Categories found: {sorted(categories)}")
        
        # Show category coverage statistics
        print("\nCategory coverage by restaurant:")
        for category in sorted(categories):
            brand_col = f"{category}_brand_name"
            if brand_col in result_df.columns:
                non_null_count = result_df[brand_col].notna().sum()
                percentage = (non_null_count / len(result_df)) * 100
                print(f"{category}: {non_null_count}/{len(result_df)} restaurants ({percentage:.1f}%)")
        
        # Show sample data for a few restaurants with data
        print("\nSample restaurants with category data:")
        sample_restaurants = result_df.dropna(subset=[col for col in result_df.columns if '_brand_name' in col], how='all').head(3)
        
        for idx, row in sample_restaurants.iterrows():
            print(f"\nRestaurant: {row['restaurant_name']} (POI: {row['poi_code']})")
            for category in sorted(categories):
                brand_col = f"{category}_brand_name"
                price_col = f"{category}_item_price"
                if brand_col in result_df.columns and pd.notna(row[brand_col]):
                    brand = row[brand_col]
                    price = row[price_col] if price_col in result_df.columns else 'N/A'
                    print(f"  {category}: {brand} - ₹{price}")
        
        # Save the result
        output_path = "/mnt/data/image_recognition/brown_forman_req/output"
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, "restaurant_brand_data_columns.csv")
        result_df.to_csv(output_file, index=False)
        print(f"\nResult saved to: {output_file}")
        
        return result_df
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    result = main()