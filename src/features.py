import pandas as pd 

def create_user_product_features(orders, order_products_prior):
    # Create User-Product Interaction Features
    print("Generating User-Product features...")
    user_product = order_products_prior.groupby(['user_id', 'product_id']).agg({
        'order_id': 'count',  # Total times user bought product 
        'add_to_cart_order': 'mean'  # Avg position in cart
    }).rename(columns={'order_id': 'up_total_bought', 'add_to_cart_order': 'up_avg_cart_pos'})

    # Create Product Features 
    print("Generating Product features...")
    product_features = order_products_prior.groupby('product_id').agg({
        'reordered': 'mean',  # Probability of product being reoredered
        'user_id': 'nunique'  # Number of unique users who bought particular product
    }).rename(columns={'reordered': 'p_reordered_ratio', 'user_id': 'p_unique_users'})

    return user_product.reset_index(), product_features.reset_index()