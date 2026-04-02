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
        col1.markdown(f"<div style='background-color:#5c5174; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Total Forecast</h4><h3>{int(forecast.sum())}</h3></div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background-color:#66669a; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Total Usage</h4><h3>{int(units.sum())}</h3></div>", unsafe_allow_html=True)
        col3.markdown(f"<div style='background-color:#aaa7cc; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Max Forecast</h4><h3>{int(forecast.max())}</h3></div>", unsafe_allow_html=True)
        col4.markdown(f"<div style='background-color:#926d88; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Max Usage</h4><h3>{int(units.max())}</h3></div>", unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)

        col5, col6, col7, col8 = st.columns(4)
        col5.markdown(f"<div style='background-color:#be9fbf; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Avg Forecast</h4><h3>{round(forecast.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col6.markdown(f"<div style='background-color:#cdaa7d; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Avg Usage</h4><h3>{round(units.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col7.markdown(f"<div style='background-color:#deb887; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>MAE</h4><h3>{round(abs_error.mean(),2)}</h3></div>", unsafe_allow_html=True)
        col8.markdown(f"<div style='background-color:#85364f; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Accuracy %</h4><h3>{round(accuracy_pct.mean(),2)}%</h3></div>", unsafe_allow_html=True)

        st.markdown(f"**High Demand Periods:** {high_demand_count}")
        st.markdown(f"**Low Demand Periods:** {low_demand_count}")

    # ---- Demand Trend ----
    elif section == "Demand Trend":
        st.subheader("Select Graph to View")

        graph_type = st.radio(
            "Choose a graph type:",
            ["Forecast vs Actual Usage", "Line Chart", "Bar Chart", "Area Chart",
             "Service Pie Chart", "Scatter", "Histogram", "Cumulative Usage",
             "Top Services", "Region Share", "Monthly Trend", "Quarterly Trend",
             "Error Trend", "Scatter with Trendline"],
            horizontal=True
        )

        if graph_type == "Forecast vs Actual Usage":
            fig, ax = plt.subplots()
            ax.plot(filtered_data['timestamp'], filtered_data['units_used'])
            ax.plot(filtered_data['timestamp'], filtered_data['forecast'])
            st.pyplot(fig)

        elif graph_type == "Line Chart":
            st.line_chart(filtered_data.set_index('timestamp')[['units_used','forecast']])

        elif graph_type == "Bar Chart":
            st.bar_chart(filtered_data.groupby('region')['forecast'].sum())

        elif graph_type == "Area Chart":
            st.area_chart(filtered_data.set_index('timestamp')['forecast'])

        elif graph_type == "Service Pie Chart":
            service_data = filtered_data.groupby('service_type')['forecast'].sum().reset_index()
            fig, ax = plt.subplots()
            ax.pie(service_data['forecast'], labels=service_data['service_type'], autopct='%1.1f%%')
            st.pyplot(fig)

        elif graph_type == "Scatter":
            fig, ax = plt.subplots()
            ax.scatter(filtered_data['units_used'], filtered_data['forecast'])
            st.pyplot(fig)

        elif graph_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(filtered_data['units_used'].dropna(), bins=20)
            st.pyplot(fig)

        elif graph_type == "Cumulative Usage":
            temp = filtered_data.sort_values('timestamp')
            cumulative = temp[['units_used','forecast']].cumsum()
            cumulative.index = temp['timestamp']
            st.line_chart(cumulative)

        elif graph_type == "Top Services":
            service_summary = filtered_data.groupby('service_type')[['units_used','forecast']].sum()
            st.bar_chart(service_summary)

        elif graph_type == "Region Share":
            region_data = filtered_data.groupby('region')['forecast'].sum().reset_index()
            fig, ax = plt.subplots()
            ax.pie(region_data['forecast'], labels=region_data['region'], autopct='%1.1f%%')
            st.pyplot(fig)

        elif graph_type == "Monthly Trend":
            filtered_data['YearMonth'] = filtered_data['timestamp'].dt.to_period('M')
            monthly_data = filtered_data.groupby('YearMonth')[['units_used','forecast']].sum().reset_index()
            monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
            monthly_data = monthly_data.set_index('YearMonth')
            st.line_chart(monthly_data)

        elif graph_type == "Quarterly Trend":
            quarterly_data = filtered_data.groupby(['year','quarter'])[['units_used','forecast']].sum().reset_index()
            quarterly_data['Year-Quarter'] = quarterly_data['year'].astype(str) + "-Q" + quarterly_data['quarter'].astype(str)
            st.bar_chart(quarterly_data.set_index('Year-Quarter')[['units_used','forecast']])

        elif graph_type == "Error Trend":
            st.line_chart(filtered_data.set_index('timestamp')['difference'])

        elif graph_type == "Scatter with Trendline":
            fig, ax = plt.subplots()
            ax.scatter(filtered_data['forecast'], filtered_data['units_used'])
            m,b = np.polyfit(filtered_data['forecast'], filtered_data['units_used'],1)
            ax.plot(filtered_data['forecast'], m*filtered_data['forecast']+b, color='red')
            st.pyplot(fig)

    # ---- Risk Alert ----
    elif section == "Risk Alert":
        threshold = st.slider("Threshold", 0, int(filtered_data['forecast'].max()))
        high_risk = filtered_data[filtered_data['forecast'] > threshold]
        st.dataframe(high_risk)

    # ---- Model Accuracy ----
    elif section == "Model Accuracy":
        st.metric("MAE", round(abs_error.mean(),2))
        st.metric("MSE", round((filtered_data['difference']**2).mean(),2))
        st.line_chart(filtered_data.set_index('timestamp')['difference'])
