import shap
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Modeli yükle
model_data = joblib.load(r'C:/Users/abdul/Desktop/main_project/models/churn_model.pkl')
model = model_data['model']
feature_cols = model_data['feature_cols']
label_encoders = model_data['label_encoders']

# Veriyi hazırla
fe = FeatureEngineer('postgresql://postgres:ack@localhost:5432/main_project')
fe.load_data()
fe.create_customer_features()
df = fe.get_features()

# Encode et
import pandas as pd
df = df.copy()
cat_cols = ['top_category', 'top_payment', 'customer_state']
for col in cat_cols:
    le = label_encoders[col]
    df[col] = le.transform(df[col].astype(str))

X = df[feature_cols]

# SHAP hesapla
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 1. Feature Importance — Bar Plot
plt.figure()
shap.summary_plot(shap_values, X, plot_type='bar', show=False)
plt.title('Churn Model — Feature Importance')
plt.tight_layout()
os.makedirs(r'C:/Users/abdul/Desktop/main_project/reports', exist_ok=True)
plt.savefig(r'C:/Users/abdul/Desktop/main_project/reports/shap_importance.png', dpi=150)
plt.show()

# 2. SHAP Summary Plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title('Churn Model — SHAP Summary')
plt.tight_layout()
plt.savefig(r'C:/Users/abdul/Desktop/main_project/reports/shap_summary.png', dpi=150)
plt.show()

print("SHAP görselleri kaydedildi ✅")