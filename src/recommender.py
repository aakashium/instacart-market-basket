import pandas as pd
import numpy as np

class InstacartRecommender:
    def __init__(self, model, product_lookup, best_threshold=0.21):
        self.model = model
        self.products = product_lookup  
        self.threshold = best_threshold

    def recommend(self, user_id, user_product_matrix):
        """
        Main recommendation engine logic.
        """
        # 1. Filter matrix for the specific user
        user_data = user_product_matrix[user_product_matrix['user_id'] == user_id].copy()
        
        # 2. Existing User with History
        if not user_data.empty:
            # Prepare features for the model (drop IDs and target)
            features = [col for col in user_data.columns if col not in ['user_id', 'product_id', 'target']]
            
            # Get probabilities
            probs = self.model.predict_proba(user_data[features])[:, 1]
            user_data['prediction_score'] = probs
            
            # Apply Optimal Threshold
            recommendations = user_data[user_data['prediction_score'] >= self.threshold]
            recommendations = recommendations.sort_values(by='prediction_score', ascending=False)
            
            # Merge with product names
            final_list = recommendations.merge(self.products, on='product_id')
            return final_list[['product_name', 'prediction_score']].head(10)
        
        # 3. New User 
        else:
            print(f"New User {user_id} detected. Falling back to Global Trends...")
            # Recommend the top 10 products globally by reorder rate
            return "Global Top 10 List" 

 