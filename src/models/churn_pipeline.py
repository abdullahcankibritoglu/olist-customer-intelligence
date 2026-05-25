import sys
sys.path.insert(0, r'C:/Users/abdul/Desktop/main_project/src/features')
from feature_engineering import FeatureEngineer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib
import os


class ChurnPipeline:
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='auc'
        )
        self.label_encoders = {}
        self.feature_cols = None
    
    def prepare_features(self, df):
        """Feature'ları hazırla, encode et"""
        
    
        cat_cols = ['top_category', 'top_payment', 'customer_state', 'segment']
        
        df = df.copy()
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        # Feature listesi
        self.feature_cols = [
            'total_orders', 'total_revenue', 'avg_order_value',
            'avg_delivery_time', 'avg_delay', 'total_items',
            'freight_ratio',

            'top_category', 'top_payment', 'customer_state'
        ]
        
        X = df[self.feature_cols]
        y = df['churn']
        
        return X, y
    
    def train(self, df):
        """Modeli eğit"""
        X, y = self.prepare_features(df)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("Model eğitildi ✅")
        return self
    
    def evaluate(self):
        """Model performansını değerlendir"""
        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        
        print("\n--- Model Performansı ---")
        print(f"AUC Score: {roc_auc_score(self.y_test, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        return self
    
    def save(self, path='models/churn_model.pkl'):
        """Modeli kaydet"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_cols': self.feature_cols
        }, path)
        print(f"Model kaydedildi: {path}")
        return self


if __name__ == "__main__":
    
    
    fe = fe = FeatureEngineer('postgresql://postgres:ack@localhost:5432/main_project')
    fe.load_data()
    fe.create_customer_features()
    df = fe.get_features()
    
    # Pipeline
    pipeline = ChurnPipeline()
    pipeline.train(df)
    pipeline.evaluate()
    pipeline.save()
