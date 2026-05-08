import pandas as pd
import numpy as np
import os

"""
Restaurant Brand Data Join Script

This script joins three CSV files to create a comprehensive dataset containing:
1. Restaurant location and service information (from zomato_club_poi_data.csv)
2. Brand items and pricing information (from brownforman_brand_items.csv) 
3. Cost for two information (from final_file_name.csv)

Output includes poi_code, restaurant details, brand information, item prices, and cost data.
"""

def load_and_join_data():
    """
    Join three CSV files:
    1. brownforman_brand_items.csv - brand item data with prices
    2. zomato_club_poi_data.csv - location data with poi_code
    3. final_file_name.csv - bridge table linking filename to poicode with cost_for_two
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
    
    # Step 1: Join bridge_data with brand_items on filename
    print("\nJoining bridge_data with brand_items on filename...")
    bridge_brand_join = pd.merge(
        bridge_df, 
        brand_items_df, 
        on='filename', 
        how='inner'
    )
    
    print(f"Bridge-Brand join shape: {bridge_brand_join.shape}")
    
    # Step 2: Join the result with places on poicode/poi_code
    print("Joining with places data on poi_code...")
    
    # Rename column for consistent joining
    bridge_brand_join = bridge_brand_join.rename(columns={'poicode': 'poi_code'})
    
    # Join with places data
    final_joined = pd.merge(
        bridge_brand_join,
        places_df,
        on='poi_code',
        how='inner'
    )
    
    print(f"Final joined shape: {final_joined.shape}")
    
    return final_joined

def prepare_final_dataset(df):
    """
    Prepare final dataset with required columns and rename for clarity
    """
    print("\nPreparing final dataset...")
    
    # Rename price to item_price for clarity
    df['item_price'] = df['price']
    
    return df

def main():
    """
    Main function to execute the join and prepare final dataset
    """
    try:
        # Load and join data
        joined_df = load_and_join_data()
        
        # Prepare final dataset
        final_df = prepare_final_dataset(joined_df)
        
        # Rename the name column from zomato data for clarity (name_y is the restaurant name from zomato_club_poi_data)
        final_df['restaurant_name'] = final_df['name_y']
        
        # Select the requested columns
        output_columns = [
            # Place information (from zomato_club_poi_data.csv)
            'poi_code', 'restaurant_name', 'address', 'location_status', 
            'reviews_count', 'reviews', 'ratings', 'offerings__serves_happy_hour_drinks',
            'offerings__serves_liquor', 'highlights__serves_wine_notable', 'offerings__serves_wine',
            'planning__accepts_reservations', 'offerings__serves_beer', 'offerings__serves_cocktails',
            
            # Brand and item information (from brownforman_brand_items.csv)
            'brand_name', 'category', 'items_name',
            
            # Price and cost information
            'item_price', 'cost_for_two'
        ]
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in output_columns if col in final_df.columns]
        result_df = final_df[available_columns]
        
        # Re-order columns to have restaurant_name right after poi_code
        if 'restaurant_name' in result_df.columns:
            cols = result_df.columns.tolist()
            cols.remove('restaurant_name')
            # Insert restaurant_name after poi_code
            poi_index = cols.index('poi_code')
            cols.insert(poi_index + 1, 'restaurant_name')
            result_df = result_df[cols]
        
        print(f"\nFinal result shape: {result_df.shape}")
        print(f"Available columns: {list(result_df.columns)}")
        
        # Display sample data
        print("\nSample data:")
        print(result_df.head(10))
        
        # Display basic statistics
        print("\nBasic Statistics:")
        print(f"Total restaurants: {result_df['poi_code'].nunique()}")
        print(f"Total brands: {result_df['brand_name'].nunique()}")
        print(f"Total items: {len(result_df)}")
        
        # Display brand statistics  
        print("\nBrand Statistics:")
        # Convert item_price to numeric, handling non-numeric values
        result_df['item_price_numeric'] = pd.to_numeric(result_df['item_price'], errors='coerce')
        
        brand_stats = result_df.groupby('brand_name').agg({
            'item_price_numeric': ['count', 'min', 'max', 'mean'],
            'poi_code': 'nunique'
        }).round(2)
        brand_stats.columns = ['item_count', 'min_price', 'max_price', 'avg_price', 'restaurant_count']
        print(brand_stats.head(10))
        
        # Remove the temporary numeric column before saving
        if 'item_price_numeric' in result_df.columns:
            result_df = result_df.drop('item_price_numeric', axis=1)
        
        # Save the result
        output_path = "/mnt/data/image_recognition/brown_forman_req/output"
        os.makedirs(output_path, exist_ok=True)
        
        output_file = os.path.join(output_path, "restaurant_brand_data_joined.csv")
        result_df.to_csv(output_file, index=False)
        print(f"\nResult saved to: {output_file}")
        
        # Also save brand statistics
        brand_stats_file = os.path.join(output_path, "brand_statistics_summary.csv")
        brand_stats.to_csv(brand_stats_file)
        print(f"Brand statistics saved to: {brand_stats_file}")
        
        return result_df
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    result = main()