import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os



class FeatureEngineer:
    
    def __init__(self):
       
        password = input("Enter your PostgreSQL password: ")
        DB_URL = f"postgresql://postgres:{password}@localhost:5432/main_project"
        self.engine = create_engine(DB_URL)
        self.reference_date = pd.Timestamp('2018-08-29')
        self.engine = create_engine(DB_URL)
        self.reference_date = pd.Timestamp('2018-08-29')
    
    def load_data(self):
        self.df = pd.read_sql("""
            SELECT 
                m.customer_id,
                m.order_id,
                m.order_purchase_timestamp,
                m.payment_value,
                m.delivery_time_days,
                m.delivery_delay_days,
                m.product_category_name_english,
                m.customer_state,
                m.payment_type,
                m.price,
                m.freight_value,
                r.recency,
                r.frequency,
                r.monetary,
                r.r_score,
                r.f_score,
                r.m_score,
                r.segment
            FROM vw_master_delivered m
            LEFT JOIN rfm_segments r ON m.customer_id = r.customer_id
        """, self.engine)
        print(f"Veri yüklendi: {self.df.shape}")
        return self
    
    def create_customer_features(self):
        customer = self.df.groupby('customer_id').agg(
            last_order_date=('order_purchase_timestamp', 'max'),
            total_orders=('order_id', 'nunique'),
            total_revenue=('payment_value', 'sum'),
            avg_order_value=('payment_value', 'mean'),
            avg_delivery_time=('delivery_time_days', 'mean'),
            avg_delay=('delivery_delay_days', 'mean'),
            total_items=('order_id', 'count'),
            freight_total=('freight_value', 'sum'),
            price_total=('price', 'sum'),
            top_category=('product_category_name_english', lambda x: x.mode()[0]),
            top_payment=('payment_type', lambda x: x.mode()[0]),
            customer_state=('customer_state', 'first'),
            recency=('recency', 'first'),
            frequency=('frequency', 'first'),
            monetary=('monetary', 'first'),
            r_score=('r_score', 'first'),
            f_score=('f_score', 'first'),
            m_score=('m_score', 'first'),
            segment=('segment', 'first')
        ).reset_index()
        
        customer['freight_ratio'] = customer['freight_total'] / (customer['price_total'] + 1)
        customer['churn'] = (
            (self.reference_date - customer['last_order_date']).dt.days > 180
        ).astype(int)
        
        self.customer_df = customer
        print(f"Churn dağılımı:\n{customer['churn'].value_counts()}")
        return self
    
    def get_features(self):
        return self.customer_df

    def create_ltv_target(self):
        ltv = self.customer_df[['customer_id', 'monetary', 'total_orders', 
                             'avg_order_value', 'avg_delivery_time',
                             'avg_delay', 'total_items', 'freight_ratio',
                             'top_category', 'top_payment', 'customer_state']].copy()
        ltv['ltv_log'] = np.log1p(ltv['monetary'])
        return ltv

if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.load_data()
    fe.create_customer_features()
    df = fe.get_features()
    print(df.head())