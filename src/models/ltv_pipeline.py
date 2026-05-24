import sys
sys.path.insert(0, r'C:/Users/abdul/Desktop/main_project/src/features')
from feature_engineering import FeatureEngineer

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os


class LTVPipeline:
    
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        self.label_encoders = {}
        self.feature_cols = None
    
    def prepare_features(self, df):
        cat_cols = ['top_category', 'top_payment', 'customer_state']
        
        df = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        self.feature_cols = [
            'total_orders', 'avg_order_value',
            'avg_delivery_time', 'avg_delay',
            'total_items', 'freight_ratio',
            'top_category', 'top_payment', 'customer_state'
        ]
        
        X = df[self.feature_cols]
        y = df['ltv_log']
        
        return X, y
    
    def train(self, df):
        X, y = self.prepare_features(df)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("LTV Modeli eğitildi ✅")
        return self
    
    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        
        # Log'dan gerçek değere çevir
        y_pred_real = np.expm1(y_pred)
        y_test_real = np.expm1(self.y_test)
        
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)
        
        print(f"\n--- LTV Model Performansı ---")
        print(f"MAE: {mae:.2f} BRL")
        print(f"R2 Score: {r2:.4f}")
        return self
    
    def save(self, path='models/ltv_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_cols': self.feature_cols
        }, path)
        print(f"Model kaydedildi: {path}")
        return self


if __name__ == "__main__":
    fe = FeatureEngineer('postgresql://postgres:ack@localhost:5432/main_project')
    fe.load_data()
    fe.create_customer_features()
    ltv_df = fe.create_ltv_target()
    
    pipeline = LTVPipeline()
    pipeline.train(ltv_df)
    pipeline.evaluate()
    pipeline.save()