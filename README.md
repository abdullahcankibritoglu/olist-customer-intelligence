# 🛒 Olist E-Commerce — Customer Intelligence System

> **End-to-end data science project:** From raw Brazilian e-commerce data to a live prediction system — featuring data engineering, advanced analytics, machine learning, explainable AI, and a deployed web application.

---

## 🎯 Project Overview

This project transforms the [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (100K+ orders) into a **production-grade customer intelligence platform** capable of predicting churn, estimating lifetime value, and delivering actionable business insights to decision-makers.

**The core business questions this system answers:**
- Which customers are about to leave — and why?
- What is each customer worth over their lifetime?
- Where are the logistics bottlenecks hurting retention?
- Which customer segments drive the most revenue?

---

## 🏗️ Architecture

```
Raw CSV Data (Kaggle)
        │
        ▼
┌─────────────────────┐
│   PostgreSQL        │  Star Schema · PK/FK · Timestamp casting
│   Data Warehouse    │  Derived features · Master views
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Advanced          │  RFM Segmentation · Cohort Analysis
│   Analytics         │  Power BI CEO Dashboard
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   ML Pipeline       │  Churn Prediction (XGBoost · AUC 0.83)
│   OOP Architecture  │  LTV Prediction   (XGBoost · R² 0.94)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Explainable AI    │  SHAP Feature Importance
│   + Deployment      │  Streamlit Web Application
└─────────────────────┘
```

---

## 📊 Key Results

| Model | Metric | Score |
|---|---|---|
| Churn Prediction | AUC | **0.83** |
| Churn Prediction | Recall (churned) | **0.89** |
| LTV Prediction | R² Score | **0.94** |
| LTV Prediction | MAE | **8.74 BRL** |

| Business Insight | Finding |
|---|---|
| Retention rate | ~3% (97% one-time buyers) |
| Churn risk segment | 22,747 customers at risk |
| Top churn driver | Delivery time (SHAP #1 feature) |
| Platform revenue | 19.78M BRL across 96K orders |
| Champion customers | 4,022 (highest LTV segment) |

---

## 🔬 Phase Breakdown

### Phase 1 — Data Engineering (Days 1–6)
- Designed and implemented **Star Schema** in PostgreSQL
- Cast all timestamp columns from TEXT → TIMESTAMP
- Detected and resolved NULL values (610 uncategorized products fixed)
- Built Portuguese → English category translation mapping
- Engineered derived features: `delivery_time_days`, `delivery_delay_days`, `estimated_delivery_days`
- Created `vw_master` and `vw_master_delivered` views as Single Source of Truth

### Phase 2 — Advanced Analytics & Business Intelligence (Days 7–12)
- **RFM Analysis** (pure SQL): Scored 96K customers across Recency, Frequency, Monetary using `NTILE(5)`
- **6 customer segments**: Champion · Loyal · Promising · Needs Attention · At Risk · Lost
- **Power BI Dashboard**: KPI cards, monthly revenue trend, delivery performance by state, segment revenue breakdown
- **Cohort Analysis**: Revealed near-zero retention — 97% of customers are one-time buyers, validating the churn modeling approach

### Phase 3 — Machine Learning (Days 13–18)
- OOP-based `FeatureEngineer` class with automated feature generation
- **Churn model** (XGBoost Classifier): Identifies customers likely to leave with 89% recall
- **LTV model** (XGBoost Regressor): Predicts customer lifetime value with log-transform target
- Avoided data leakage by excluding RFM raw scores from features
- Models serialized with `joblib` for deployment

### Phase 4 — XAI & Deployment (Days 19–24)
- **SHAP analysis**: `avg_delivery_time` and `avg_delay` are the top churn drivers
- **Streamlit app**: Two-tab interface for real-time Churn and LTV prediction
- Dark-mode UI with Plotly gauge charts, risk tiers, and recommended actions

---

## 📁 Project Structure

```
olist-customer-intelligence/
├── src/
│   ├── features/
│   │   └── feature_engineering.py    # FeatureEngineer class
│   └── models/
│       ├── churn_pipeline.py         # ChurnPipeline class
│       ├── ltv_pipeline.py           # LTVPipeline class
│       └── shap_analysis.py          # Explainability module
├── streamlit_app/
│   └── app.py                        # Live prediction web app
├── notebooks/
│   └── 02_cohort_analysis.ipynb      # Retention analysis
├── sql/
│   ├── schema/                       # CREATE TABLE scripts
│   ├── cleaning/                     # Data quality scripts
│   └── analytics/                   # RFM, cohort queries
├── reports/
│   ├── shap_importance.png
│   ├── shap_summary.png
│   └── cohort_heatmap.png
├── .env.example                      # Environment template
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/olist-customer-intelligence.git
cd olist-customer-intelligence
pip install -r requirements.txt
```

### 2. Configure Database
```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Run Feature Engineering
```bash
python src/features/feature_engineering.py
```

### 4. Train Models
```bash
python src/models/churn_pipeline.py
python src/models/ltv_pipeline.py
```

### 5. Launch Web App
```bash
streamlit run streamlit_app/app.py
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Database | PostgreSQL · pgAdmin |
| Data Processing | Python · Pandas · NumPy · SQLAlchemy |
| Machine Learning | Scikit-learn · XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib · Seaborn · Plotly |
| Business Intelligence | Power BI · DAX |
| Web App | Streamlit |
| Version Control | Git · GitHub |

---

## 💡 Business Insights

> **"Delivery time is the #1 churn driver."**
> SHAP analysis shows that customers with longer delivery times are significantly more likely to churn. Reducing average delivery time by 5 days could materially improve retention.

> **"97% of customers never come back."**
> Cohort analysis revealed near-zero retention across all cohorts. The platform's growth is entirely dependent on new customer acquisition — making churn prediction and LTV optimization critical investments.

> **"22,747 customers are at risk right now."**
> The RFM model identifies a high-risk segment that represents 34% of total platform revenue. Targeted retention campaigns for this group have the highest ROI potential.

---

## 📈 What Makes This Different

Most portfolio projects stop at EDA or a single model. This project:

- Starts at the **database level** with proper schema design
- Uses **OOP architecture** — no notebooks in production code
- Connects analytics to ML — **RFM segments feed into churn features**
- Explains model decisions with **SHAP** — not just accuracy metrics
- Ships a **live web application** anyone can interact with
- Frames every technical finding as a **business decision**

---

## 👤 Author

**Abdullah Can Kibritoğlu**


---

*Dataset: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle*
