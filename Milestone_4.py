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
    # ---- Compute common metrics for all sections ----
    filtered_data['difference'] = filtered_data['units_used'] - filtered_data['forecast']
    abs_error = filtered_data['difference'].abs()
    accuracy_pct = 100 * (1 - abs_error / filtered_data['units_used'].replace(0,1))

    # ---- Top Navigation Bar ----
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

        # First row of KPI cards
        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f"<div style='background-color:#5c5174; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Total Forecast</h4><h3>{int(forecast.sum())}</h3>"
                      f"<p>Sum of all forecasted units</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div style='background-color:#66669a; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Total Usage</h4><h3>{int(units.sum())}</h3>"
                      f"<p>Sum of actual units used</p></div>", unsafe_allow_html=True)
        col3.markdown(f"<div style='background-color:#aaa7cc; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Max Forecast</h4><h3>{int(forecast.max())}</h3>"
                      f"<p>Peak forecast value</p></div>", unsafe_allow_html=True)
        col4.markdown(f"<div style='background-color:#926d88; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Max Actual Usage</h4><h3>{int(units.max())}</h3>"
                      f"<p>Peak actual usage</p></div>", unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Second row of KPI cards
        col5, col6, col7, col8 = st.columns(4)
        col5.markdown(f"<div style='background-color:#be9fbf; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Average Forecast</h4><h3>{round(forecast.mean(),2)}</h3>"
                      f"<p>Mean forecast value</p></div>", unsafe_allow_html=True)
        col6.markdown(f"<div style='background-color:#cdaa7d; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Average Actual Usage</h4><h3>{round(units.mean(),2)}</h3>"
                      f"<p>Mean actual usage</p></div>", unsafe_allow_html=True)
        col7.markdown(f"<div style='background-color:#deb887; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Mean Absolute Error</h4><h3>{round(abs_error.mean(),2)}</h3>"
                      f"<p>Forecast vs Actual</p></div>", unsafe_allow_html=True)
        col8.markdown(f"<div style='background-color:#85364f; padding:5px; border-radius:10px; text-align:center;'>"
                      f"<h4>Forecast Accuracy (%)</h4><h3>{round(accuracy_pct.mean(),2)}%</h3>"
                      f"<p>Average accuracy</p></div>", unsafe_allow_html=True)

        st.markdown(f"**High Demand Periods (Actual > Forecast):** {high_demand_count}")
        st.markdown(f"**Low Demand Periods (Actual < Forecast):** {low_demand_count}")

    # ---- Demand Trend ----
    elif section == "Demand Trend":
        st.subheader("Select Graph to View")
        graph_type = st.radio(
            "Choose a graph type:",
            ["Forecast vs Actual Usage", "Line Chart", "Bar Chart", "Area Chart", "Service Pie Chart",
             "Scatter", "Histogram", "Cumulative Usage", "Top Services", "Region Share",
             "Monthly Trend", "Quarterly Trend", "Error Trend", "Scatter with Trendline"],
            index=0,
            horizontal=True
        )

        # ---- Graphs ----
        if graph_type == "Forecast vs Actual Usage":
            st.subheader("Forecast vs Actual Usage Over Time")
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(filtered_data['timestamp'], filtered_data['units_used'], label="Actual Usage", linestyle='--', marker='o')
            ax.plot(filtered_data['timestamp'], filtered_data['forecast'], label="Forecast", linewidth=2, marker='x')
            ax.set_xlabel("Time")
            ax.set_ylabel("Units")
            ax.legend()
            ax.set_title("Forecast vs Actual Usage")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        elif graph_type == "Line Chart":
            st.line_chart(filtered_data.set_index('timestamp')[['units_used', 'forecast']])

        elif graph_type == "Bar Chart":
            st.subheader("Region-wise Forecast")
            region_data = filtered_data.groupby('region')['forecast'].sum()
            st.bar_chart(region_data)

        elif graph_type == "Area Chart":
            st.area_chart(filtered_data.set_index('timestamp')['forecast'])

        elif graph_type == "Service Pie Chart":
            st.subheader("Service Distribution")
            service_data = filtered_data.groupby('service_type')['forecast'].sum().sort_values(ascending=False)
            top_n = 5
            top_services = service_data[:top_n]
            others_sum = service_data[top_n:].sum()
            final_service_data = top_services.copy()
            if others_sum > 0:
                final_service_data["Others"] = others_sum
            explode = [0.05]*len(final_service_data)
            fig, ax = plt.subplots(figsize=(7,7))
            wedges, texts, autotexts = ax.pie(final_service_data, autopct='%1.1f%%', startangle=90, explode=explode, wedgeprops={'edgecolor':'black'})
            ax.legend(wedges, final_service_data.index, title="Service Type", loc="center left", bbox_to_anchor=(1,0,0.5,1))
            ax.set_title("Service Type Share")
            st.pyplot(fig)

        elif graph_type == "Scatter":
            fig, ax = plt.subplots()
            ax.scatter(filtered_data['units_used'], filtered_data['forecast'])
            ax.set_xlabel("Actual")
            ax.set_ylabel("Forecast")
            ax.set_title("Actual vs Forecast")
            st.pyplot(fig)

        elif graph_type == "Histogram":
            fig, ax = plt.subplots()
            ax.hist(filtered_data['units_used'], bins=20)
            ax.set_xlabel("Usage")
            ax.set_ylabel("Frequency")
            ax.set_title("Usage Distribution")
            st.pyplot(fig)

        elif graph_type == "Cumulative Usage":
            st.subheader("Cumulative Usage vs Forecast")
            cumulative = filtered_data[['units_used','forecast']].cumsum()
            st.area_chart(cumulative.set_index(filtered_data['timestamp']))

        elif graph_type == "Top Services":
            st.subheader("Top Service Types by Usage vs Forecast")
            service_summary = filtered_data.groupby('service_type')[['units_used','forecast']].sum()
            st.bar_chart(service_summary)

        elif graph_type == "Region Share":
            st.subheader("Region-wise Forecast Share")
            region_forecast = filtered_data.groupby('region')['forecast'].sum().sort_values(ascending=False)
            top_n = 5
            top_regions = region_forecast[:top_n]
            others_sum = region_forecast[top_n:].sum()
            final_region_data = top_regions.copy()
            if others_sum > 0:
                final_region_data["Others"] = others_sum
            explode = [0.05]*len(final_region_data)
            fig, ax = plt.subplots(figsize=(7,7))
            wedges, texts, autotexts = ax.pie(final_region_data, autopct='%1.1f%%', startangle=90, explode=explode, wedgeprops={'edgecolor':'black'})
            ax.legend(wedges, final_region_data.index, title="Region", loc="center left", bbox_to_anchor=(1,0,0.5,1))
            ax.set_title("Region-wise Forecast Share")
            st.pyplot(fig)

        elif graph_type == "Monthly Trend":
            st.subheader("Monthly Forecast Trend")
            monthly_data = filtered_data.resample('M', on='timestamp')[['units_used','forecast']].sum()
            st.line_chart(monthly_data)

        elif graph_type == "Quarterly Trend":
            st.subheader("Quarterly Forecast Trend")
            quarterly_data = filtered_data.groupby(['year','quarter'])[['units_used','forecast']].sum().reset_index()
            quarterly_data['Year-Quarter'] = quarterly_data['year'].astype(str) + "-Q" + quarterly_data['quarter'].astype(str)
            st.bar_chart(quarterly_data.set_index('Year-Quarter')[['units_used','forecast']])

        elif graph_type == "Error Trend":
            st.subheader("Prediction Error (Actual - Forecast)")
            st.line_chart(filtered_data.set_index('timestamp')['difference'])

        elif graph_type == "Scatter with Trendline":
            st.subheader("Actual vs Forecast Scatter with Trendline")
            fig, ax = plt.subplots()
            ax.scatter(filtered_data['forecast'], filtered_data['units_used'])
            m,b = np.polyfit(filtered_data['forecast'], filtered_data['units_used'],1)
            ax.plot(filtered_data['forecast'], m*filtered_data['forecast']+b, color='red', label='Trendline')
            ax.set_xlabel("Forecast")
            ax.set_ylabel("Actual Usage")
            ax.legend()
            st.pyplot(fig)

    # ---- Risk Alert ----
    elif section == "Risk Alert":
        st.subheader("Risk Alerts")
        threshold = st.slider("Set Usage Threshold", min_value=0, max_value=int(filtered_data['forecast'].max()), value=int(filtered_data['forecast'].max()*0.8))
        high_risk = filtered_data[filtered_data['forecast'] > threshold]
        if high_risk.empty:
            st.success("No high-risk periods detected.")
        else:
            st.warning(f"High-risk periods detected: {len(high_risk)}")
            st.dataframe(high_risk[['timestamp','region','service_type','forecast','units_used']])

    # ---- Model Accuracy ----
    elif section == "Model Accuracy":
        st.subheader("Model Accuracy Overview")
        st.metric("Mean Absolute Error (MAE)", round(abs_error.mean(),2))
        st.metric("Mean Squared Error (MSE)", round((filtered_data['difference']**2).mean(),2))
        st.line_chart(filtered_data.set_index('timestamp')['difference'])