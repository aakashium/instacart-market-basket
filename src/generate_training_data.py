import pandas as pd

def build_training_set(orders, user_product_features, product_features):
    # Filter for the 'train' and 'test' evaluation sets provided by Instacart
    train_orders = orders[orders['eval_set'] == 'train']
    
    # Merge user-product features with the orders we want to predict
    # This creates the skeleton of training data
    training_data = train_orders[['user_id', 'order_id']].merge(
        user_product_features, on='user_id', how='left'
    )
    
    # Add 'New User' flag for Cold Start
    training_data['is_new_user'] = training_data['up_total_bought'].isnull().astype(int)
    
    # Fill missing values for new users with global product averages 
    training_data = training_data.merge(product_features, on='product_id', how='left')
    
    return training_data

