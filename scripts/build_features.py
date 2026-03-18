import pandas as pd
import sys
import os

# 1. Path setup 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import get_user_product_features, get_product_features
from src.labeler import generate_labels
from src.matrix_builder import build_final_matrix

def main():
    # 2. Load Optimized Data
    print("Loading optimized parquet files...")
    orders = pd.read_parquet('data/processed/orders.parquet')
    prior = pd.read_parquet('data/processed/order_products__prior.parquet')
    train = pd.read_parquet('data/processed/order_products__train.parquet')

    # 3. Mapping user_id
    print("Mapping user_id to train set...")
    train = train.merge(orders[['order_id', 'user_id']], on='order_id', how='left')

    # 4. Generate Features 
    user_product_features = get_user_product_features(prior, orders)
    product_features = get_product_features(prior)

    # 5. Generate Labels 
    labels = generate_labels(user_product_features, train)

    # 6. Assemble the Final Matrix
    final_matrix = build_final_matrix(user_product_features, product_features, labels)

    # 7. Save for Modeling
    final_matrix.to_parquet('data/processed/final_training_matrix.parquet')
    print("\nSuccess! Final matrix saved to data/processed/final_training_matrix.parquet")

if __name__ == "__main__":
    main()