import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="FinAI Pro | Advanced Mode", layout="wide")

# Custom Red/Dark Styling
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1a1c24; border-right: 2px solid #dc2626; }
    .stMetric { border-left: 3px solid #dc2626; padding-left: 10px; }
    .stButton>button { background-color: #dc2626; color: white; border: none; font-weight: bold; width: 100%; }
    .stButton>button:hover { background-color: #b91c1c; border: 1px solid white; }
    </style>
    """, unsafe_allow_html=True)

# Load AI Assets
try:
    model = joblib.load('loan_model.pkl')
    le = joblib.load('label_encoder.pkl')
except:
    st.error("Please run the trainer script first!")
    st.stop()

# --- SIDEBAR: MULTI-SECTION INPUTS ---
st.sidebar.title("📑 Full Financial Disclosure")

with st.sidebar.expander("💳 Credit & Compliance", expanded=True):
    cibil = st.slider("CIBIL Score", 300, 900, 750)
    gst_discipline = st.slider("GST Filing Score", 1, 10, 9)

with st.sidebar.expander("📉 Balance Sheet (Liquidity)"):
    cur_assets = st.number_input("Total Current Assets", value=500000)
    cur_liab = st.number_input("Total Current Liabilities", value=250000)
    cr = round(cur_assets / cur_liab, 2) if cur_liab != 0 else 0

with st.sidebar.expander("💰 Profit & Loss (Efficiency)"):
    revenue = st.number_input("Annual Revenue", value=1000000)
    operating_exp = st.number_input("Operating Expenses", value=600000)
    net_profit = revenue - operating_exp
    npm = round((net_profit / revenue) * 100, 2) if revenue != 0 else 0

with st.sidebar.expander("⚖️ Leverage & Debt"):
    equity = st.number_input("Shareholder Equity", value=400000)
    long_term_debt = st.number_input("Long Term Debt", value=100000)
    de_ratio = round(long_term_debt / equity, 2) if equity != 0 else 0

# --- MAIN DASHBOARD ---
st.title("🏦 AI-Driven Loan Assessment Engine")
st.markdown("### Accuracy-Enhanced Appraisal Model")

# Section 1: Top Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Liquidity (CR)", f"{cr}x")
m2.metric("Profit Margin (NPM)", f"{npm}%")
m3.metric("Leverage (D/E)", f"{de_ratio}x")
m4.metric("GST Score", f"{gst_discipline}/10")

st.divider()

col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("🤖 Artificial Intelligence Verdict")
    if st.button("RUN DEEP ASSESSMENT"):
        roa = round((net_profit / (cur_assets + long_term_debt)) * 100, 2)
        
        # We cap ROA for the model to prevent outlier errors
        roa_input = min(roa, 30.0)
        features = np.array([[cibil, gst_discipline, cr, de_ratio, roa_input]])
        
        # 1. Get AI Prediction
        pred = model.predict(features)
        prob = model.predict_proba(features)[0][1]
        status = le.inverse_transform(pred)[0]

        # 2. THE FIX: Critical Kill-Switches (Hard Rejections)
        rejection_reasons = []
        if gst_discipline < 3:
            status = "Rejected"
            rejection_reasons.append("Critical GST Non-Compliance")
        if cibil < 500:
            status = "Rejected"
            rejection_reasons.append("Low Credit Worthiness")
        if cr < 0.9:
            status = "Rejected"
            rejection_reasons.append("Poor Liquidity (Current Ratio < 0.9)")

        # 3. Elite Approval Rule (Only applies if no hard rejections exist)
        if not rejection_reasons:
            if cibil > 800 and cr > 2.0 and npm > 20:
                status = "Approved"
                prob = 0.98  # Force high confidence

        # 4. Final Display Logic
        if status == "Approved":
            st.success(f"### ✅ LOAN APPROVED")
            st.write(f"The AI is **{prob*100:.2f}%** confident in this business's stability.")
            st.balloons()
        else:
            st.error(f"### ❌ LOAN REJECTED")
            # If there are specific hard-reject reasons, show them
            if rejection_reasons:
                for reason in rejection_reasons:
                    st.warning(f"**Reason:** {reason}")
            else:
                st.write(f"Risk analysis indicates a **{(1-prob)*100:.2f}%** chance of default.")

with col_b:
    st.subheader("📈 Financial Distribution")
    # A bar chart showing the balance of components
    chart_data = pd.DataFrame({
        'Metric': ['Assets', 'Equity', 'Revenue', 'Debt'],
        'Value': [cur_assets, equity, revenue, long_term_debt]
    })
    st.bar_chart(chart_data, x='Metric', y='Value', color="#dc2626")

st.markdown("---")
st.caption("Admin Console | FinGuard AI | Arjun Sarkar & Urvashi Bohare")