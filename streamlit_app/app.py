import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Sayfa Ayarları ──────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Customer Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

* { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

[data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e8f0;
}
[data-testid="stHeader"] {
    background: #0a0a0f;
}
.metric-card {
    background: linear-gradient(135deg, #12121e 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 8px 0;
}
.metric-value { font-size: 2.5rem; font-weight: 700; font-family: 'Space Mono', monospace; }
.metric-label { font-size: 0.85rem; color: #8888aa; text-transform: uppercase; letter-spacing: 2px; margin-top: 8px; }

.churn-high { color: #ff4d6d; }
.churn-low  { color: #4dffb4; }
.churn-mid  { color: #ffd166; }

.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #5555aa;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a1a3a;
}

div[data-testid="stSlider"] > div { color: #8888aa; }
.stSelectbox label, .stSlider label, .stNumberInput label {
    color: #8888aa !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: #12121e !important;
    border: 1px solid #2a2a4a !important;
    color: #e8e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Model Yükle ──────────────────────────────────────────────────
@st.cache_resource
def load_models():
    churn = joblib.load(r'C:/Users/abdul/Desktop/main_project/models/churn_model.pkl')
    ltv   = joblib.load(r'C:/Users/abdul/Desktop/main_project/models/ltv_model.pkl')
    return churn, ltv

churn_data, ltv_data = load_models()
churn_model    = churn_data['model']
churn_encoders = churn_data['label_encoders']
churn_features = churn_data['feature_cols']

ltv_model    = ltv_data['model']
ltv_encoders = ltv_data['label_encoders']
ltv_features = ltv_data['feature_cols']

# ── Başlık ───────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 40px 0 32px 0;">
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#5555aa; 
                letter-spacing:4px; text-transform:uppercase; margin-bottom:12px;">
        OLIST E-COMMERCE · CUSTOMER INTELLIGENCE SYSTEM
    </div>
    <h1 style="font-size:2.8rem; margin:0; background:linear-gradient(90deg,#a78bfa,#60a5fa,#4dffb4);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1;">
        Predict. Understand.<br>Retain.
    </h1>
</div>
""", unsafe_allow_html=True)

# ── Tab Menü ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔮  Churn Prediction", "💎  LTV Prediction"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — CHURN
# ════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

        total_orders     = st.slider("Total Orders", 1, 30, 1)
        total_revenue    = st.number_input("Total Revenue (BRL)", 10.0, 15000.0, 150.0, step=10.0)
        avg_order_value  = st.number_input("Avg Order Value (BRL)", 10.0, 5000.0, 150.0, step=10.0)
        avg_delivery     = st.slider("Avg Delivery Time (days)", 1, 60, 12)
        avg_delay        = st.slider("Avg Delay (days, negative = early)", -30, 30, -5)
        total_items      = st.slider("Total Items Purchased", 1, 50, 1)
        freight_ratio    = st.slider("Freight Ratio (0=free, 1=expensive)", 0.0, 1.0, 0.15)

        categories = ['health_beauty','bed_bath_table','computers_accessories',
                      'furniture_decor','watches_gifts','sports_leisure',
                      'housewares','auto','garden_tools','cool_stuff',
                      'office_furniture','toys','baby','telephony','perfumery']
        payments   = ['credit_card','boleto','voucher','debit_card']
        states     = ['SP','RJ','MG','RS','PR','SC','BA','GO','PE','CE',
                      'ES','MT','MS','MA','RO','PB','AM','AP','RR','TO','AC','AL','SE','PA','PI','RN','DF']

        top_category = st.selectbox("Top Category", categories)
        top_payment  = st.selectbox("Payment Type", payments)
        customer_state = st.selectbox("Customer State", states)

        predict_btn = st.button("⚡ RUN PREDICTION", use_container_width=True)

    with col_result:
        if predict_btn:
            # Encode
            input_dict = {
                'total_orders': total_orders,
                'total_revenue': total_revenue,
                'avg_order_value': avg_order_value,
                'avg_delivery_time': avg_delivery,
                'avg_delay': avg_delay,
                'total_items': total_items,
                'freight_ratio': freight_ratio,
                'top_category': churn_encoders['top_category'].transform([top_category])[0],
                'top_payment':  churn_encoders['top_payment'].transform([top_payment])[0],
                'customer_state': churn_encoders['customer_state'].transform([customer_state])[0],
            }
            X = pd.DataFrame([input_dict])[churn_features]
            prob = churn_model.predict_proba(X)[0][1]
            pred = int(prob > 0.5)

            # Renk
            if prob >= 0.7:
                color, label, emoji = '#ff4d6d', 'HIGH RISK', '🔴'
            elif prob >= 0.4:
                color, label, emoji = '#ffd166', 'MEDIUM RISK', '🟡'
            else:
                color, label, emoji = '#4dffb4', 'LOW RISK', '🟢'

            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}33; margin-bottom:16px;">
                <div style="font-size:4rem;">{emoji}</div>
                <div class="metric-value" style="color:{color};">{prob:.1%}</div>
                <div class="metric-label">Churn Probability</div>
                <div style="color:{color}; font-family:'Space Mono',monospace; 
                            font-size:0.9rem; margin-top:12px; letter-spacing:2px;">
                    {label}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'color': color, 'size': 36}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#555577'},
                    'bar': {'color': color, 'thickness': 0.3},
                    'bgcolor': '#12121e',
                    'bordercolor': '#2a2a4a',
                    'steps': [
                        {'range': [0, 40],  'color': '#0d2f1f'},
                        {'range': [40, 70], 'color': '#2f2a0d'},
                        {'range': [70, 100],'color': '#2f0d1a'},
                    ],
                    'threshold': {'line': {'color': color, 'width': 3}, 'value': prob * 100}
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8e8f0',
                height=280,
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Öneri
            st.markdown('<div class="section-title">Recommended Action</div>', unsafe_allow_html=True)
            if prob >= 0.7:
                st.error("🚨 **Immediate action required.** Send retention campaign, offer discount, prioritize support.")
            elif prob >= 0.4:
                st.warning("⚠️ **Monitor closely.** Engage with loyalty program or personalized offer.")
            else:
                st.success("✅ **Customer is healthy.** Continue standard engagement.")
        else:
            st.markdown("""
            <div style="height:400px; display:flex; align-items:center; justify-content:center;
                        flex-direction:column; color:#333355; text-align:center;">
                <div style="font-size:4rem; margin-bottom:16px;">🔮</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.8rem; letter-spacing:2px;">
                    CONFIGURE PROFILE & RUN PREDICTION
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# TAB 2 — LTV
# ════════════════════════════════════════════════════════════════
with tab2:
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-title">Customer Profile</div>', unsafe_allow_html=True)

        l_orders    = st.slider("Total Orders", 1, 30, 1, key="ltv_orders")
        l_aov       = st.number_input("Avg Order Value (BRL)", 10.0, 5000.0, 150.0, key="ltv_aov")
        l_delivery  = st.slider("Avg Delivery Time (days)", 1, 60, 12, key="ltv_del")
        l_delay     = st.slider("Avg Delay (days)", -30, 30, -5, key="ltv_delay")
        l_items     = st.slider("Total Items", 1, 50, 1, key="ltv_items")
        l_freight   = st.slider("Freight Ratio", 0.0, 1.0, 0.15, key="ltv_freight")
        l_category  = st.selectbox("Top Category", categories, key="ltv_cat")
        l_payment   = st.selectbox("Payment Type", payments, key="ltv_pay")
        l_state     = st.selectbox("Customer State", states, key="ltv_state")

        ltv_btn = st.button("💎 CALCULATE LTV", use_container_width=True)

    with col_r:
        if ltv_btn:
            input_dict = {
                'total_orders': l_orders,
                'avg_order_value': l_aov,
                'avg_delivery_time': l_delivery,
                'avg_delay': l_delay,
                'total_items': l_items,
                'freight_ratio': l_freight,
                'top_category': ltv_encoders['top_category'].transform([l_category])[0],
                'top_payment':  ltv_encoders['top_payment'].transform([l_payment])[0],
                'customer_state': ltv_encoders['customer_state'].transform([l_state])[0],
            }
            X = pd.DataFrame([input_dict])[ltv_features]
            ltv_log = ltv_model.predict(X)[0]
            ltv_val = np.expm1(ltv_log)

            if ltv_val >= 500:
                color, tier = '#4dffb4', 'PLATINUM'
            elif ltv_val >= 200:
                color, tier = '#60a5fa', 'GOLD'
            else:
                color, tier = '#a78bfa', 'STANDARD'

            st.markdown(f"""
            <div class="metric-card" style="border-color:{color}33;">
                <div class="metric-value" style="color:{color};">R$ {ltv_val:,.2f}</div>
                <div class="metric-label">Predicted Lifetime Value</div>
                <div style="color:{color}; font-family:'Space Mono',monospace;
                            font-size:0.9rem; margin-top:12px; letter-spacing:2px;">
                    {tier} CUSTOMER
                </div>
            </div>
            """, unsafe_allow_html=True)

            # LTV Breakdown bar
            fig2 = go.Figure(go.Bar(
                x=[ltv_val],
                y=['LTV'],
                orientation='h',
                marker_color=color,
                text=[f'R$ {ltv_val:,.2f}'],
                textposition='inside',
                textfont=dict(color='#0a0a0f', size=14)
            ))
            fig2.add_vline(x=204, line_dash="dash", line_color="#555577",
                           annotation_text="Avg: R$204", annotation_font_color="#555577")
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e8e8f0',
                height=120,
                margin=dict(t=20, b=20, l=20, r=20),
                xaxis=dict(showgrid=False, color='#555577'),
                yaxis=dict(showgrid=False, color='#555577')
            )
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown('<div class="section-title">Business Insight</div>', unsafe_allow_html=True)
            if ltv_val >= 500:
                st.success(f"💎 **Platinum customer.** Prioritize VIP treatment, dedicated support, exclusive offers.")
            elif ltv_val >= 200:
                st.info(f"🥇 **High value customer.** Enroll in loyalty program, cross-sell premium categories.")
            else:
                st.warning(f"📈 **Growth potential.** Nurture with targeted campaigns to increase order frequency.")
        else:
            st.markdown("""
            <div style="height:400px; display:flex; align-items:center; justify-content:center;
                        flex-direction:column; color:#333355; text-align:center;">
                <div style="font-size:4rem; margin-bottom:16px;">💎</div>
                <div style="font-family:'Space Mono',monospace; font-size:0.8rem; letter-spacing:2px;">
                    CONFIGURE PROFILE & CALCULATE LTV
                </div>
            </div>
            """, unsafe_allow_html=True)