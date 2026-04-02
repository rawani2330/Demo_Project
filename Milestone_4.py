import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- Page Config ----
st.set_page_config(page_title="Demand Forecast Dashboard", layout="wide")
st.title("XGBoost Forecast Dashboard")

# ---- Load Data ----
data = pd.read_csv("forecast_output.csv")

# FIX: Safe datetime conversion
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
data = data.dropna(subset=['timestamp'])

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
].copy()

# ---- Check Empty ----
if filtered_data.empty:
    st.warning("No data available for selected filters.")
else:
    filtered_data = filtered_data.copy()

    # ---- Metrics ----
    filtered_data['difference'] = filtered_data['units_used'] - filtered_data['forecast']
    abs_error = filtered_data['difference'].abs()

    # FIX: divide by zero handled
    accuracy_pct = np.where(
        filtered_data['units_used'] == 0,
        0,
        100 * (1 - abs_error / filtered_data['units_used'])
    )

    # ---- Navigation ----
    section = st.radio(
        "Select Section:",
        ["KPI Overview", "Demand Trend", "Risk Alert", "Model Accuracy"],
        horizontal=True
    )

    # ================= KPI =================
    if section == "KPI Overview":
        units = filtered_data['units_used']
        forecast = filtered_data['forecast']

        high_demand = (units > forecast).sum()
        low_demand = (units < forecast).sum()

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div style='background:#5c5174;padding:10px;border-radius:10px;text-align:center'><h4>Total Forecast</h4><h3>{int(forecast.sum())}</h3></div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background:#66669a;padding:10px;border-radius:10px;text-align:center'><h4>Total Usage</h4><h3>{int(units.sum())}</h3></div>", unsafe_allow_html=True)
        col3.markdown(f"<div style='background:#aaa7cc;padding:10px;border-radius:10px;text-align:center'><h4>Max Forecast</h4><h3>{int(forecast.max())}</h3></div>", unsafe_allow_html=True)
        col4.markdown(f"<div style='background:#926d88;padding:10px;border-radius:10px;text-align:center'><h4>Max Usage</h4><h3>{int(units.max())}</h3></div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        col5, col6, col7, col8 = st.columns(4)
        col5.markdown(f"<div style='background:#be9fbf;padding:10px;border-radius:10px;text-align:center'><h4>Avg Forecast</h4><h3>{round(forecast.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col6.markdown(f"<div style='background:#cdaa7d;padding:10px;border-radius:10px;text-align:center'><h4>Avg Usage</h4><h3>{round(units.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col7.markdown(f"<div style='background:#deb887;padding:10px;border-radius:10px;text-align:center'><h4>MAE</h4><h3>{round(abs_error.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col8.markdown(f"<div style='background:#85364f;padding:10px;border-radius:10px;text-align:center'><h4>Accuracy %</h4><h3>{round(accuracy_pct.mean(),2)}%</h3></div>", unsafe_allow_html=True)

        st.markdown(f"**High Demand Periods (Actual > Forecast):** {high_demand}")
        st.markdown(f"**Low Demand Periods (Actual < Forecast):** {low_demand}")

    # ================= DEMAND TREND =================
    elif section == "Demand Trend":

        graph_type = st.radio(
            "Choose Graph:",
            ["Forecast vs Actual Usage", "Line Chart", "Bar Chart", "Area Chart",
             "Service Pie Chart", "Scatter", "Histogram", "Cumulative Usage",
             "Top Services", "Region Share", "Monthly Trend", "Quarterly Trend",
             "Error Trend", "Scatter with Trendline"],
            horizontal=True
        )

        filtered_data = filtered_data.sort_values('timestamp')

        if graph_type == "Forecast vs Actual Usage":
            fig, ax = plt.subplots()
            ax.plot(filtered_data['timestamp'], filtered_data['units_used'], label="Actual")
            ax.plot(filtered_data['timestamp'], filtered_data['forecast'], label="Forecast")
            ax.legend()
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif graph_type == "Service Pie Chart":
            service_data = filtered_data.groupby('service_type')['forecast'].sum().sort_values(ascending=False)
            top = service_data.head(5).copy()
            others = service_data.iloc[5:].sum()
            if others > 0:
                top.loc["Others"] = others  # FIXED

            fig, ax = plt.subplots(figsize=(7,7))
            ax.pie(top.values, labels=top.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        elif graph_type == "Region Share":
            region_data = filtered_data.groupby('region')['forecast'].sum().sort_values(ascending=False)
            top = region_data.head(5).copy()
            others = region_data.iloc[5:].sum()
            if others > 0:
                top.loc["Others"] = others  # FIXED

            fig, ax = plt.subplots(figsize=(7,7))
            ax.pie(top.values, labels=top.index, autopct='%1.1f%%')
            ax.axis('equal')
            st.pyplot(fig)

        elif graph_type == "Monthly Trend":
            df_month = filtered_data.set_index('timestamp')
            monthly = df_month.resample('MS')[['units_used','forecast']].sum()  # FIXED
            st.line_chart(monthly)

        elif graph_type == "Quarterly Trend":
            q = filtered_data.groupby(['year','quarter'])[['units_used','forecast']].sum().reset_index()
            q['YQ'] = q['year'].astype(str) + "-Q" + q['quarter'].astype(str)
            st.bar_chart(q.set_index('YQ')[['units_used','forecast']])

        elif graph_type == "Error Trend":
            st.line_chart(filtered_data.set_index('timestamp')['difference'])

        elif graph_type == "Scatter with Trendline":
            fig, ax = plt.subplots()
            x = filtered_data['forecast']
            y = filtered_data['units_used']
            ax.scatter(x, y)

            if len(x) > 1:
                m, b = np.polyfit(x, y, 1)
                ax.plot(x, m*x + b)

            st.pyplot(fig)

        elif graph_type == "Line Chart":
            st.line_chart(filtered_data.set_index('timestamp')[['units_used','forecast']])

        elif graph_type == "Bar Chart":
            st.bar_chart(filtered_data.groupby('region')['forecast'].sum())

        elif graph_type == "Area Chart":
            st.area_chart(filtered_data.set_index('timestamp')['forecast'])

        elif graph_type == "Scatter":
            fig, ax = plt.subplots()
            ax.scatter(filtered_data['units_used'], filtered_data['forecast'])
            st.pyplot(fig)

        elif graph_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(filtered_data['units_used'], bins=20)
            st.pyplot(fig)

        elif graph_type == "Cumulative Usage":
            cum = filtered_data[['units_used','forecast']].cumsum()
            cum.index = filtered_data['timestamp']
            st.line_chart(cum)

        elif graph_type == "Top Services":
            st.bar_chart(filtered_data.groupby('service_type')[['units_used','forecast']].sum())

    # ================= RISK =================
    elif section == "Risk Alert":
        threshold = st.slider(
            "Set Threshold",
            0,
            int(filtered_data['forecast'].max()),
            int(filtered_data['forecast'].max()*0.8)
        )

        filtered_data['Risk'] = np.where(
            filtered_data['forecast'] > threshold,
            "🔴 Very Risky",
            "🟢 Safe"
        )

        st.metric("Very Risky Count", (filtered_data['forecast'] > threshold).sum())
        st.metric("Safe Count", (filtered_data['forecast'] <= threshold).sum())

        st.dataframe(filtered_data[['timestamp','region','service_type','forecast','units_used','Risk']])

    # ================= ACCURACY =================
    elif section == "Model Accuracy":
        st.metric("MAE", round(abs_error.mean(),2))
        st.metric("MSE", round((filtered_data['difference']**2).mean(),2))
        st.line_chart(filtered_data.set_index('timestamp')['difference'])
