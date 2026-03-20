"""
Sales Forecasting Dashboard for Streamlit Cloud.

Direct model loading - no FastAPI backend required.
"""

import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Project modules (relative imports)
import sys
sys.path.append(str(Path(__file__).parent))  # Add project root to path

from app.config import (
    ROOT, MODELS_DIR, FEATURES_CSV, RESULTS_CSV, COMPARISON_CSV,
    FEATURE_COLS, TREE_MODELS, HORIZON, MLFLOW_UI, EXPERIMENT_ID
)
from app.model_loader import ModelRegistry
from app.predictor import forecast

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")

# Cache registry loading (startup once)
@st.cache_resource
def load_registry():
    registry = ModelRegistry()
    registry.load_all()
    return registry

# Load on startup
try:
    registry = load_registry()
    states = registry.states
    available_models = ["Auto (best per state)"] + (registry.available_models or [])
    st.success(f"✅ Loaded {len(states)} states, {len(registry.models)} models")
except Exception as e:
    st.error(f"❌ Failed to load models/data: {e}")
    st.stop()

# Sidebar
st.sidebar.title("Forecast Settings")
selected_state = st.sidebar.selectbox("State", states)
selected_model = st.sidebar.selectbox("Model", available_models)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 52, HORIZON)
run_btn = st.sidebar.button("Generate Forecast", type="primary", width='stretch')

# Header
st.title("📈 Sales Forecasting Dashboard")
st.markdown("Beverage Sales Forecasting Case Study")

# Model comparison 
with st.expander("📊 Model Comparison (all states avg)", expanded=False):

    col1, col2 = st.columns([4, 1])

    with col2:
        if MLFLOW_UI:
            st.link_button("🔗 Open MLflow", MLFLOW_UI)
        else:
            st.info("MLflow (local)")

    comp = registry.get_comparison()

    if comp and comp.get("comparison"):

        comp_df = pd.DataFrame(comp["comparison"]).T
        comp_df.index.name = "Model"
        comp_df = comp_df.sort_values("RMSE")

        st.dataframe(
            comp_df[["MAE", "RMSE", "MAPE"]],
            width='stretch',
        )

        st.markdown("### 🔍 MLflow Runs")

        for model in comp_df.index:
            run_id = comp["comparison"][model].get("run_id")

            if run_id and run_id != "N/A":
                url = f"{MLFLOW_UI}/#/experiments/{EXPERIMENT_ID}/runs/{run_id}"
                st.markdown(f"[👉 View {model} Run]({url})")

        best_name = comp_df.index[0]
        best_rmse = comp_df.loc[best_name, "RMSE"]

        st.success(f"Best overall model: {best_name} (RMSE={best_rmse:.2f})")

    else:
        st.info("Direct links to MLflow")

# Per-state metrics 
if selected_state:
    with st.expander(f"📋 Metrics for {selected_state}", expanded=False):

        col1, col2 = st.columns([4, 1])

        with col2:
            if MLFLOW_UI:
                st.link_button("🔗 MLflow Experiments", f"{MLFLOW_UI}/#/experiments")
            else:
                st.info("MLflow (local)")

        sm = registry.get_state_metrics(selected_state)
        if sm:
            sm_df = pd.DataFrame(sm)
            sm_df = sm_df[["model", "MAE", "RMSE", "MAPE"]].sort_values("RMSE")
            st.dataframe(sm_df, width='stretch', hide_index=True)
            if registry.get_best_model_name(selected_state):
                st.info(f"Best model for {selected_state}: **{registry.get_best_model_name(selected_state)}**")
        else:
            st.info("Direct links to MLflow")

# Forecast section
st.header("Forecast")

if run_btn and selected_state:
    model_param = None if selected_model.startswith("Auto") else selected_model
    
    with st.spinner(f"Generating {horizon}-week forecast for {selected_state}…"):
        try:
            result = forecast(registry, selected_state, horizon, model_param)
            
            st.success(
                f"**{result['model_name']}** → {result['state']} "
                f"({result['horizon']} weeks)"
            )
            
            fc_df = pd.DataFrame(result["forecasts"])
            fc_df["date"] = pd.to_datetime(fc_df["date"])
            
            # Chart
            st.subheader("Forecast Chart")
            chart_df = fc_df.set_index("date")[["predicted_sales"]].rename(
                columns={"predicted_sales": "Predicted Sales ($)"}
            )
            st.line_chart(chart_df, width='stretch')
            
            # Table
            st.subheader("Forecast Data")
            display = fc_df.copy()
            display.columns = ["Date", "State", "Predicted Sales ($)"]
            display["Predicted Sales ($)"] = display["Predicted Sales ($)"].map("{:,.2f}".format)
            st.dataframe(display, width='stretch', hide_index=True)
            
            # Download
            csv = fc_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                file_name=f"forecast_{selected_state.lower().replace(' ','_')}_{horizon}w.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Forecast failed: {e}")
else:
    st.info("Select a state and click **Generate Forecast** in the sidebar.")

# Footer
st.markdown("---")
st.caption("Deployed on Streamlit Cloud")

