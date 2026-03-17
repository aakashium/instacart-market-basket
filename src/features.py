import pandas as pd 

def get_user_product_features(order_products_prior, orders):
    """
    Generates features at the User-Product interaction level.
    Focuses on Intensity and Recency.
    """
    print("Generating User-Product features...")

    # Merge to get order_number and days_since_prior_order for recency
    df = order_products_prior.merge(orders[['order_id', 'user_id', 'order_number']], on='order_id')
    
    user_product = df.groupby(['user_id', 'product_id']).agg({
        'order_id': 'count',                 # Total times user bought product
        'order_number': ['min', 'max'],      # First and last time they bought 
        'add_to_cart_order': 'mean'          # Average position in cart
    })
    
    # Flatten multi-index columns
    user_product.columns = ['up_total_bought', 'up_first_order', 'up_last_order', 'up_avg_cart_pos']
    user_product = user_product.reset_index()
    
    # Feature: Reorder Probability (How many times bought / orders since first purchase)
    user_total_orders = orders.groupby('user_id')['order_number'].max().rename('user_total_orders')
    user_product = user_product.merge(user_total_orders, on='user_id')
    
    user_product['up_order_rate'] = user_product['up_total_bought'] / (user_product['user_total_orders'] - user_product['up_first_order'] + 1)
    
    return user_product

def get_product_features(order_products_prior):
    """
    Generates features for each product (Global Popularity).
    Useful for 'Option B' (New Users).
    """
    print("Building Product features...")
    
    product_features = order_products_prior.groupby('product_id').agg({
        'reordered': ['count', 'mean'],     # Total sales and reorder rate
        'add_to_cart_order': 'mean'         
    })
    
    product_features.columns = ['p_total_sales', 'p_reorder_ratio', 'p_avg_cart_pos']
    return product_features.reset_index()