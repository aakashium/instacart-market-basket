import pandas as pd

def build_final_matrix(user_product_features, product_features, labels, user_features=None):
    """
    Assembles the final training matrix for the ML model.
    """
    print("Assembling final training matrix...")
    
    # 1. Start with the labels (The user-product pairs and their 1/0 targets)
    data = labels.copy()
    
    # 2. Join User-Product Features (Recency, Intensity, Consistency)
    data = data.merge(user_product_features, on=['user_id', 'product_id'], how='left')
    
    # 3. Join Global Product Features (Popularity, Reorder Rate)
    data = data.merge(product_features, on='product_id', how='left')
    
    # 4. Fill NaNs for features
    # For new users, their 'user-product' features will be 0
    data = data.fillna(0)
    
    print(f"Final matrix shape: {data.shape}")
    return data

