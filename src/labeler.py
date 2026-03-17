import pandas as pd

def generate_labels(user_product_features, order_products_train):
    """
    Creates the 'reordered' target variable (1 or 0).
    Input: 
        - user_product_features: The list of all (user, product) pairs from history.
        - order_products_train: The actual products bought in the 'next' order.
    """
    print("Generating training labels...")
    
    # We only care about products the user has seen before 
    # Merge the train order data with our feature skeleton
    labels = user_product_features[['user_id', 'product_id']].merge(
        order_products_train[['user_id', 'product_id', 'reordered']], 
        on=['user_id', 'product_id'], 
        how='left'
    )
    
    # If it's NaN, it means they didn't buy it in the train order -> 0
    labels['target'] = labels['reordered'].fillna(0).astype('int8')
    
    return labels.drop(columns=['reordered'])