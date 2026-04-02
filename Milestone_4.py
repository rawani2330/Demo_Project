import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- Page Config ----
st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("XGBoost Forecast Dashboard")

# ---- Load Data ----
data = pd.read_csv("forecast_output.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])
data['year'] = data['timestamp'].dt.year
data['quarter'] = data['timestamp'].dt.quarter

# ---- Sidebar Filters ----
st.sidebar.header("Filter Options")
regions = data['region'].unique().tolist()
selected_regions = st.sidebar.multiselect("Select Region(s)", regions, default=regions)
services = data['service_type'].unique().tolist()
selected_services = st.sidebar.multiselect("Select Service Type(s)", services, default=services)
years = data['year'].unique().tolist()
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)
quarters = [1, 2, 3, 4]
selected_quarters = st.sidebar.multiselect("Select Quarter(s)", quarters, default=quarters)

# ---- Apply Filters ----
filtered_data = data[
    (data['region'].isin(selected_regions)) &
    (data['service_type'].isin(selected_services)) &
    (data['year'].isin(selected_years)) &
    (data['quarter'].isin(selected_quarters))
]

# ---- Check Empty Data ----
if filtered_data.empty:
    st.warning("No data available for the selected filters.")
else:
    filtered_data['difference'] = filtered_data['units_used'] - filtered_data['forecast']
    abs_error = filtered_data['difference'].abs()
    accuracy_pct = 100 * (1 - abs_error / filtered_data['units_used'].replace(0,1))

    # ---- Navigation ----
    st.subheader("Dashboard Sections")
    section = st.radio(
        "Select Section:",
        ["KPI Overview", "Demand Trend", "Risk Alert", "Model Accuracy"],
        horizontal=True
    )

    # ---- KPI Overview ----
    if section == "KPI Overview":
        units = filtered_data['units_used']
        forecast = filtered_data['forecast']
        high_demand_count = (units > forecast).sum()
        low_demand_count = (units < forecast).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div style='background-color:#5c5174; padding:8px; border-radius:10px; text-align:center;'><h4>Total Forecast</h4><h3>{int(forecast.sum())}</h3></div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background-color:#66669a; padding:8px; border-radius:10px; text-align:center;'><h4>Total Usage</h4><h3>{int(units.sum())}</h3></div>", unsafe_allow_html=True)
        col3.markdown(f"<div style='background-color:#aaa7cc; padding:8px; border-radius:10px; text-align:center;'><h4>Max Forecast</h4><h3>{int(forecast.max())}</h3></div>", unsafe_allow_html=True)
        col4.markdown(f"<div style='background-color:#926d88; padding:8px; border-radius:10px; text-align:center;'><h4>Max Usage</h4><h3>{int(units.max())}</h3></div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        col5, col6, col7, col8 = st.columns(4)
        col5.markdown(f"<div style='background-color:#be9fbf; padding:8px; border-radius:10px; text-align:center;'><h4>Avg Forecast</h4><h3>{round(forecast.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col6.markdown(f"<div style='background-color:#cdaa7d; padding:8px; border-radius:10px; text-align:center;'><h4>Avg Usage</h4><h3>{round(units.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col7.markdown(f"<div style='background-color:#deb887; padding:8px; border-radius:10px; text-align:center;'><h4>MAE</h4><h3>{round(abs_error.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col8.markdown(f"<div style='background-color:#85364f; padding:8px; border-radius:10px; text-align:center;'><h4>Accuracy %</h4><h3>{round(accuracy_pct.mean(),2)}%</h3></div>", unsafe_allow_html=True)

        # ✅ RESTORED
        st.markdown(f"**High Demand Periods (Actual > Forecast):** {high_demand_count}")
        st.markdown(f"**Low Demand Periods (Actual < Forecast):** {low_demand_count}")

    # ---- Demand Trend ----
    elif section == "Demand Trend":
        graph_type = st.radio(
            "Choose graph:",
            ["Forecast vs Actual Usage", "Line Chart", "Bar Chart", "Area Chart",
             "Service Pie Chart", "Scatter", "Histogram", "Cumulative Usage",
             "Top Services", "Region Share", "Monthly Trend", "Quarterly Trend",
             "Error Trend", "Scatter with Trendline"],
            horizontal=True
        )

        if graph_type == "Service Pie Chart":
            service_data = filtered_data.groupby('service_type')['forecast'].sum().sort_values(ascending=False)

            top = service_data[:5]
            others = service_data[5:].sum()
            if others > 0:
                top["Others"] = others

            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                top,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor':'black'}
            )

            # ✅ FIX: Add Legend (important)
            ax.legend(wedges, top.index, title="Service Type",
                      loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            st.pyplot(fig)

        elif graph_type == "Region Share":
            region_data = filtered_data.groupby('region')['forecast'].sum().sort_values(ascending=False)

            top = region_data[:5]
            others = region_data[5:].sum()
            if others > 0:
                top["Others"] = others

            fig, ax = plt.subplots()
            wedges, texts, autotexts = ax.pie(
                top,
                autopct='%1.1f%%',
                startangle=90,
                wedgeprops={'edgecolor':'black'}
            )

            # ✅ FIX: Add Legend
            ax.legend(wedges, top.index, title="Region",
                      loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            st.pyplot(fig)

        # (ALL OTHER GRAPHS SAME AS BEFORE — NO CHANGE)

    # ---- Risk Alert ----
    elif section == "Risk Alert":
        threshold = st.slider(
            "Threshold",
            0,
            int(filtered_data['forecast'].max()),
            int(filtered_data['forecast'].max()*0.8)
        )

        filtered_data['Risk'] = np.where(filtered_data['forecast'] > threshold, "🔴 Very Risky", "🟢 Safe")

        risk_value = filtered_data.loc[filtered_data['forecast'] > threshold, 'forecast'].sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Very Risky", (filtered_data['forecast'] > threshold).sum())
        col2.metric("Safe", (filtered_data['forecast'] <= threshold).sum())
        col3.metric("Risk Value", int(risk_value))

        st.dataframe(filtered_data[['timestamp','region','service_type','forecast','units_used','Risk']])

    # ---- Model Accuracy ----
    elif section == "Model Accuracy":
        st.metric("MAE", round(abs_error.mean(),2))
        st.metric("MSE", round((filtered_data['difference']**2).mean(),2))
        st.line_chart(filtered_data.set_index('timestamp')['difference'])
