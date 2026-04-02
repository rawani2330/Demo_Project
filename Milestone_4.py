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

    # ---- KPI ----
    if section == "KPI Overview":
        units = filtered_data['units_used']
        forecast = filtered_data['forecast']

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Forecast", int(forecast.sum()))
        col2.metric("Total Usage", int(units.sum()))
        col3.metric("Max Forecast", int(forecast.max()))
        col4.metric("Max Usage", int(units.max()))

        st.markdown("<br>", unsafe_allow_html=True)

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Avg Forecast", round(forecast.mean(),2))
        col6.metric("Avg Usage", round(units.mean(),2))
        col7.metric("MAE", round(abs_error.mean(),2))
        col8.metric("Accuracy %", round(accuracy_pct.mean(),2))

    # ---- Demand Trend ----
    elif section == "Demand Trend":
        graph_type = st.radio(
            "Choose graph:",
            ["Forecast vs Actual Usage", "Line Chart", "Bar Chart", "Area Chart",
             "Service Pie Chart", "Scatter", "Histogram", "Cumulative Usage",
             "Top Services", "Region Share", "Monthly Trend"],
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

        # ---- FIXED PIE ----
        elif graph_type == "Service Pie Chart":
            service_data = filtered_data.groupby('service_type')['forecast'].sum().reset_index()
            if service_data.empty:
                st.warning("No data")
            else:
                fig, ax = plt.subplots()
                ax.pie(service_data['forecast'], labels=service_data['service_type'], autopct='%1.1f%%')
                st.pyplot(fig)

        # ---- FIXED SCATTER ----
        elif graph_type == "Scatter":
            fig, ax = plt.subplots()
            ax.scatter(filtered_data['units_used'], filtered_data['forecast'])
            st.pyplot(fig)

        # ---- FIXED HISTOGRAM ----
        elif graph_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(filtered_data['units_used'].dropna(), bins=20)
            st.pyplot(fig)

        # ---- FIXED CUMULATIVE ----
        elif graph_type == "Cumulative Usage":
            temp = filtered_data.sort_values('timestamp')
            cumulative = temp[['units_used','forecast']].cumsum()
            cumulative.index = temp['timestamp']
            st.line_chart(cumulative)

        # ---- FIXED TOP SERVICES ----
        elif graph_type == "Top Services":
            service_summary = filtered_data.groupby('service_type')[['units_used','forecast']].sum()
            st.bar_chart(service_summary)

        # ---- FIXED REGION SHARE ----
        elif graph_type == "Region Share":
            region_data = filtered_data.groupby('region')['forecast'].sum().reset_index()
            fig, ax = plt.subplots()
            ax.pie(region_data['forecast'], labels=region_data['region'], autopct='%1.1f%%')
            st.pyplot(fig)

        # ---- FIXED MONTHLY ----
        elif graph_type == "Monthly Trend":
            filtered_data['YearMonth'] = filtered_data['timestamp'].dt.to_period('M')
            monthly_data = filtered_data.groupby('YearMonth')[['units_used','forecast']].sum().reset_index()
            monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
            monthly_data = monthly_data.set_index('YearMonth')
            st.line_chart(monthly_data)

    # ---- Risk ----
    elif section == "Risk Alert":
        threshold = st.slider("Threshold", 0, int(filtered_data['forecast'].max()))
        st.dataframe(filtered_data[filtered_data['forecast'] > threshold])

    # ---- Accuracy ----
    elif section == "Model Accuracy":
        st.metric("MAE", round(abs_error.mean(),2))
        st.metric("MSE", round((filtered_data['difference']**2).mean(),2))
        st.line_chart(filtered_data.set_index('timestamp')['difference'])
