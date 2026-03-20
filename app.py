"""
Sales Forecasting Dashboard.

Connects to the FastAPI backend for all data. No direct file access.
"""

import pandas as pd
import streamlit as st
import requests

#  Config 
API_BASE = "http://localhost:8000"

MLFLOW_UI = "http://127.0.0.1:5000"
EXPERIMENT_ID = "1"

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")


#  API helpers 

def api_get(path: str):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API. Start it with: `uvicorn api.main:app --port 8000`")
        st.stop()
    except requests.HTTPError as e:
        #st.info("Direct links to MLflow")
        return None

def api_post(path: str, body: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("Cannot reach the API. Start it with: `uvicorn api.main:app --port 8000`")
        st.stop()
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")
        return None


#  Sidebar 

st.sidebar.title("Forecast Settings")

states_data = api_get("/states")
models_data = api_get("/models")

states = states_data["states"] if states_data else []
models = models_data["models"] if models_data else []

selected_state = st.sidebar.selectbox("State", states, index=0 if states else None)
selected_model = st.sidebar.selectbox(
    "Model",
    ["Auto (best per state)"] + models,
    index=0,
)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", 1, 52, 8)
run_btn = st.sidebar.button("Generate Forecast", type="primary", use_container_width=True)


#  Header 

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Beverage Sales Forecasting Case Study")


#  Model comparison 

with st.expander("📊 Model Comparison (all states avg)", expanded=False):

    col1, col2 = st.columns([4, 1])

    with col2:
        st.link_button("🔗 Open MLflow", MLFLOW_UI)

    metrics_data = api_get("/metrics")

    if metrics_data and metrics_data.get("comparison"):

        comp_df = pd.DataFrame(metrics_data["comparison"]).T
        comp_df.index.name = "Model"
        comp_df = comp_df.sort_values("RMSE")

        st.dataframe(
            comp_df[["MAE", "RMSE", "MAPE"]],
            use_container_width=True,
        )

        st.markdown("### 🔍 MLflow Runs")

        for model in comp_df.index:
            run_id = metrics_data["comparison"][model].get("run_id")

            if run_id and run_id != "N/A":
                url = f"{MLFLOW_UI}/#/experiments/{EXPERIMENT_ID}/runs/{run_id}"
                st.markdown(f"[👉 View {model} Run]({url})")

        best_name = comp_df.index[0]
        best_rmse = comp_df.loc[best_name, "RMSE"]

        st.success(f"Best overall model: {best_name} (RMSE={best_rmse:.2f})")

    else:
        st.info("Direct links to MLflow")


#  Per-state metrics 

if selected_state:
    with st.expander(f"📋 Metrics for {selected_state}", expanded=False):

        col1, col2 = st.columns([4, 1])

        with col2:
            st.link_button("🔗 MLflow Experiments", f"{MLFLOW_UI}/#/experiments")

        sm = api_get(f"/metrics/{selected_state}")
        if sm and sm.get("metrics"):
            sm_df = pd.DataFrame(sm["metrics"])
            sm_df = sm_df[["model", "MAE", "RMSE", "MAPE"]].sort_values("RMSE")
            st.dataframe(sm_df, use_container_width=True, hide_index=True)
            if sm.get("best_model"):
                st.info(f"Best model for {selected_state}: **{sm['best_model']}**")
        else:
            st.info("Direct links to MLflow")


#  Forecast 

st.header("Forecast")

if run_btn and selected_state:
    model_param = None if selected_model.startswith("Auto") else selected_model

    with st.spinner(f"Generating {horizon}-week forecast for {selected_state}…"):
        payload = {"state": selected_state, "horizon": horizon}
        if model_param:
            payload["model_name"] = model_param
        result = api_post("/forecast", payload)

    if result:
        st.success(
            f"**{result['model_name']}** → {result['state']} "
            f"({result['horizon']} weeks)"
        )

        fc_df = pd.DataFrame(result["forecasts"])
        fc_df["date"] = pd.to_datetime(fc_df["date"])

        # chart
        st.subheader("Forecast Chart")
        chart_df = fc_df.set_index("date")[["predicted_sales"]].rename(
            columns={"predicted_sales": "Predicted Sales ($)"}
        )
        st.line_chart(chart_df, use_container_width=True)

        # table
        st.subheader("Forecast Data")
        display = fc_df.copy()
        display.columns = ["Date", "State", "Predicted Sales ($)"]
        display["Predicted Sales ($)"] = display["Predicted Sales ($)"].map("{:,.2f}".format)
        st.dataframe(display, use_container_width=True, hide_index=True)

        # download
        csv = fc_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            file_name=f"forecast_{selected_state.lower().replace(' ','_')}_{horizon}w.csv",
            mime="text/csv",
        )
else:
    st.info("Select a state and click **Generate Forecast** in the sidebar.")


#  Footer 
st.markdown("---")
st.caption("API docs: http://localhost:8000/docs")
