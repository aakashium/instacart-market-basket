import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import reduce_mem_usage

def optimize():
    raw_path = 'data/raw/'
    processed_path = 'data/processed/'
    
    # Create processed directory if it doesn't exist
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
        print(f"Created directory: {processed_path}")

    # List of all files to optimize
    files = [
        'orders.csv', 
        'order_products__prior.csv', 
        'order_products__train.csv', 
        'products.csv', 
        'aisles.csv', 
        'departments.csv'
    ]

    for file in files:
        file_path = os.path.join(raw_path, file)
        if os.path.exists(file_path):
            print(f"--- Optimizing {file} ---")
            df = pd.read_csv(file_path)
            
            # Apply our memory reduction function
            df_optimized = reduce_mem_usage(df)
            
            # Save as Parquet for speed and efficiency
            output_name = file.replace('.csv', '.parquet')
            df_optimized.to_parquet(os.path.join(processed_path, output_name))
            print(f"Saved to {output_name}\n")
        else:
            print(f"Warning: {file} not found in {raw_path}")

if __name__ == "__main__":
    optimize()