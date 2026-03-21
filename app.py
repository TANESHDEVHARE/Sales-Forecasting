"""
Sales Forecasting Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from app.config import ROOT
from app.model_loader import ModelRegistry
from app.predictor import forecast

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")

# MLflow Images at Top
st.markdown("**MLflow Tracked Models**")
col1, col2 = st.columns(2)
col1.image("Assets/MLflow - I.png")
col2.image("Assets/MLflow - II.png")

st.title("📈 Sales Forecasting Dashboard")


@st.cache_resource
def load_registry():
    registry = ModelRegistry()
    registry.load_all()
    return registry

if 'registry' not in st.session_state:
    with st.spinner("Loading models..."):
        try:
            st.session_state.registry = load_registry()
        except:
            st.error("Run experiment.ipynb first")
            st.stop()

registry = st.session_state.registry

# Simple Sidebar
state = st.sidebar.selectbox("State", registry.states)
horizon = st.sidebar.slider("Weeks", 1, 52, 8)

if st.sidebar.button("Generate Forecast"):
    result = forecast(registry, state, horizon)
    
    df = pd.DataFrame(result["forecasts"])
    df["date"] = pd.to_datetime(df["date"])
    
    st.success(f"**{result['model_name']}** - Best Model")
    
    col1, col2 = st.columns(2)
    col1.metric("Total Revenue", f"${df['predicted_sales'].sum():,.0f}")
    col2.metric("Avg Weekly", f"${df['predicted_sales'].mean():,.0f}")
    
    fig = px.line(df, x="date", y="predicted_sales")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Data")
    display_df = df.copy()
    display_df['predicted_sales'] = display_df['predicted_sales'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df)
    
    st.download_button("Download CSV", df.to_csv(index=False), f"{state}_forecast.csv")



