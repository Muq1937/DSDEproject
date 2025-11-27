"""
Interactive Geospatial Dashboard for Urban Issue Forecasting
Built with Streamlit, Plotly, and Folium
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster, TimestampedGeoJson
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Urban Issue Forecasting Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load processed complaint data"""
    # In production, load from Delta Lake or processed parquet
    # For demo, generate sample data
    np.random.seed(42)
    n_samples = 10000

    date_range = pd.date_range(start='2021-08-01', end='2025-01-31', freq='6H')
    sample_dates = np.random.choice(date_range, size=n_samples)

    districts = [
        'ปทุมวัน', 'ห้วยขวาง', 'ดินแดง', 'คลองเตย', 'วัฒนา',
        'ราชเทวี', 'บางรัก', 'สาทร', 'ยานนาวา', 'พระโขนง',
        'ลาดพร้าว', 'จตุจักร', 'บางเขน', 'ประเวศ', 'บางนา'
    ]

    complaint_types = ['น้ำท่วม', 'จราจร', 'ความสะอาด', 'ถนน', 'ทางเท้า',
                      'สะพาน', 'ท่อระบายน้ำ', 'ไฟฟ้า', 'ร้องเรียน']

    df = pd.DataFrame({
        'ticket_id': [f'2021-{i:06d}' for i in range(n_samples)],
        'timestamp': sample_dates,
        'district': np.random.choice(districts, n_samples),
        'type': np.random.choice(complaint_types, n_samples),
        'lat': 13.7563 + np.random.uniform(-0.15, 0.15, n_samples),
        'lon': 100.5018 + np.random.uniform(-0.15, 0.15, n_samples),
        'state': np.random.choice(['เสร็จสิ้น', 'กำลังดำเนินการ', 'รอรับเรื่อง'],
                                 n_samples, p=[0.7, 0.25, 0.05]),
        'solve_days': np.random.gamma(shape=2, scale=20, size=n_samples),
        'anomaly_score': np.random.beta(2, 5, size=n_samples)
    })

    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['hour'] = df['timestamp'].dt.hour

    return df


@st.cache_data
def load_forecast_data():
    """Load forecasting predictions"""
    future_dates = pd.date_range(start=datetime.now(), periods=30, freq='D')

    # Simulate forecast with uncertainty
    trend = np.linspace(100, 120, 30)
    seasonality = 15 * np.sin(2 * np.pi * np.arange(30) / 7)
    forecast = trend + seasonality + np.random.normal(0, 5, 30)

    df_forecast = pd.DataFrame({
        'date': future_dates,
        'predicted': forecast,
        'lower_bound': forecast - 10,
        'upper_bound': forecast + 10
    })

    return df_forecast


def create_geospatial_map(df, map_type='heatmap'):
    """Create interactive geospatial visualization"""
    # Center on Bangkok
    center_lat, center_lon = 13.7563, 100.5018

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )

    if map_type == 'heatmap':
        # Heat map of complaint density
        heat_data = [[row['lat'], row['lon']] for idx, row in df.iterrows()]
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13).add_to(m)

    elif map_type == 'clusters':
        # Marker clusters
        marker_cluster = MarkerCluster().add_to(m)

        for idx, row in df.head(500).iterrows():  # Limit for performance
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"""
                    <b>District:</b> {row['district']}<br>
                    <b>Type:</b> {row['type']}<br>
                    <b>Date:</b> {row['timestamp'].strftime('%Y-%m-%d')}<br>
                    <b>Status:</b> {row['state']}
                """,
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(marker_cluster)

    elif map_type == 'choropleth':
        # District-level choropleth (simplified)
        pass  # Would need GeoJSON boundaries

    return m


def plot_time_series(df):
    """Plot complaint volume over time"""
    daily = df.groupby(df['timestamp'].dt.date).size().reset_index()
    daily.columns = ['date', 'complaints']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['complaints'],
        mode='lines',
        name='Daily Complaints',
        line=dict(color='#1f77b4', width=2),
        fill='tonexty',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))

    # Add 7-day moving average
    daily['ma7'] = daily['complaints'].rolling(window=7).mean()
    fig.add_trace(go.Scatter(
        x=daily['date'],
        y=daily['ma7'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title='Complaint Volume Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Complaints',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_forecast(df_forecast):
    """Plot forecasting predictions"""
    fig = go.Figure()

    # Prediction with confidence interval
    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['upper_bound'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['lower_bound'],
        mode='lines',
        name='Lower Bound',
        line=dict(width=0),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df_forecast['date'],
        y=df_forecast['predicted'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=3)
    ))

    fig.update_layout(
        title='30-Day Complaint Volume Forecast',
        xaxis_title='Date',
        yaxis_title='Predicted Complaints',
        hovermode='x unified',
        template='plotly_white',
        height=400
    )

    return fig


def plot_category_distribution(df):
    """Plot complaint type distribution"""
    category_counts = df['type'].value_counts()

    fig = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        labels={'x': 'Complaint Type', 'y': 'Count'},
        title='Complaint Distribution by Category',
        color=category_counts.values,
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def plot_district_heatmap(df):
    """District vs time heatmap"""
    pivot = df.pivot_table(
        index='district',
        columns=df['timestamp'].dt.month,
        values='ticket_id',
        aggfunc='count',
        fill_value=0
    )

    fig = px.imshow(
        pivot,
        labels=dict(x='Month', y='District', color='Complaints'),
        title='Complaint Intensity by District and Month',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )

    fig.update_layout(height=500)

    return fig


def plot_resolution_time(df):
    """Plot resolution time distribution"""
    fig = px.box(
        df,
        x='type',
        y='solve_days',
        title='Resolution Time by Complaint Type',
        labels={'solve_days': 'Days to Resolve', 'type': 'Complaint Type'},
        color='type'
    )

    fig.update_layout(
        template='plotly_white',
        height=400,
        showlegend=False
    )

    return fig


def main():
    """Main dashboard application"""

    # Header
    st.markdown('<div class="main-header">🏙️ Urban Issue Forecasting Dashboard</div>',
               unsafe_allow_html=True)
    st.markdown("### Bangkok Complaint Analysis & Prediction System")
    st.markdown("---")

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        df_forecast = load_forecast_data()

    # Sidebar filters
    st.sidebar.header("📊 Filters & Settings")

    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(max_date - timedelta(days=365), max_date),
        min_value=min_date,
        max_value=max_date
    )

    # District filter
    districts = ['All'] + sorted(df['district'].unique().tolist())
    selected_district = st.sidebar.selectbox("Select District", districts)

    # Complaint type filter
    types = ['All'] + sorted(df['type'].unique().tolist())
    selected_type = st.sidebar.selectbox("Select Complaint Type", types)

    # Map visualization type
    map_type = st.sidebar.radio(
        "Map Visualization",
        ['heatmap', 'clusters'],
        format_func=lambda x: 'Heat Map' if x == 'heatmap' else 'Marker Clusters'
    )

    # Apply filters
    df_filtered = df.copy()
    if len(date_range) == 2:
        df_filtered = df_filtered[
            (df_filtered['timestamp'].dt.date >= date_range[0]) &
            (df_filtered['timestamp'].dt.date <= date_range[1])
        ]

    if selected_district != 'All':
        df_filtered = df_filtered[df_filtered['district'] == selected_district]

    if selected_type != 'All':
        df_filtered = df_filtered[df_filtered['type'] == selected_type]

    # Key Metrics
    st.header("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Complaints",
            f"{len(df_filtered):,}",
            delta=f"{len(df_filtered) - len(df):,}"
        )

    with col2:
        avg_resolution = df_filtered['solve_days'].mean()
        st.metric(
            "Avg Resolution Time",
            f"{avg_resolution:.1f} days"
        )

    with col3:
        completion_rate = (df_filtered['state'] == 'เสร็จสิ้น').mean() * 100
        st.metric(
            "Completion Rate",
            f"{completion_rate:.1f}%"
        )

    with col4:
        anomaly_rate = (df_filtered['anomaly_score'] > 0.7).mean() * 100
        st.metric(
            "Anomaly Rate",
            f"{anomaly_rate:.1f}%"
        )

    st.markdown("---")

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Geospatial Analysis",
        "📊 Time Series & Forecasting",
        "📈 Analytics",
        "🔍 Anomaly Detection"
    ])

    with tab1:
        st.header("Interactive Geospatial Map")

        # Create and display map
        m = create_geospatial_map(df_filtered, map_type=map_type)
        folium_static(m, width=1200, height=600)

        # District statistics
        st.subheader("District Statistics")
        district_stats = df_filtered.groupby('district').agg({
            'ticket_id': 'count',
            'solve_days': 'mean',
            'anomaly_score': 'mean'
        }).round(2)
        district_stats.columns = ['Total Complaints', 'Avg Resolution Days', 'Avg Anomaly Score']
        district_stats = district_stats.sort_values('Total Complaints', ascending=False)

        st.dataframe(district_stats, use_container_width=True)

    with tab2:
        st.header("Time Series Analysis & Forecasting")

        # Historical time series
        st.plotly_chart(plot_time_series(df_filtered), use_container_width=True)

        # Forecast
        st.subheader("30-Day Forecast (LSTM Model)")
        st.plotly_chart(plot_forecast(df_forecast), use_container_width=True)

        # Seasonal patterns
        col1, col2 = st.columns(2)

        with col1:
            # Day of week pattern
            dow_counts = df_filtered.groupby('day_of_week').size()
            fig_dow = px.bar(
                x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                y=dow_counts.values,
                title='Complaints by Day of Week',
                labels={'x': 'Day', 'y': 'Count'}
            )
            st.plotly_chart(fig_dow, use_container_width=True)

        with col2:
            # Hour of day pattern
            hour_counts = df_filtered.groupby('hour').size()
            fig_hour = px.line(
                x=hour_counts.index,
                y=hour_counts.values,
                title='Complaints by Hour of Day',
                labels={'x': 'Hour', 'y': 'Count'},
                markers=True
            )
            st.plotly_chart(fig_hour, use_container_width=True)

    with tab3:
        st.header("Detailed Analytics")

        # Category distribution
        st.plotly_chart(plot_category_distribution(df_filtered), use_container_width=True)

        # District heatmap
        st.plotly_chart(plot_district_heatmap(df_filtered), use_container_width=True)

        # Resolution time
        st.plotly_chart(plot_resolution_time(df_filtered), use_container_width=True)

    with tab4:
        st.header("Anomaly Detection Results")

        # Filter anomalies
        anomalies = df_filtered[df_filtered['anomaly_score'] > 0.7]

        st.metric("Total Anomalies Detected", f"{len(anomalies):,}")

        if len(anomalies) > 0:
            # Anomaly timeline
            fig_anomaly = go.Figure()
            fig_anomaly.add_trace(go.Scatter(
                x=anomalies['timestamp'],
                y=anomalies['anomaly_score'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=anomalies['anomaly_score'],
                    colorscale='Reds',
                    showscale=True
                ),
                text=[f"{row['district']} - {row['type']}" for _, row in anomalies.iterrows()],
                hovertemplate='<b>%{text}</b><br>Score: %{y:.2f}<br>Date: %{x}<extra></extra>'
            ))

            fig_anomaly.update_layout(
                title='Anomaly Detection Timeline',
                xaxis_title='Date',
                yaxis_title='Anomaly Score',
                template='plotly_white',
                height=400
            )

            st.plotly_chart(fig_anomaly, use_container_width=True)

            # Anomaly table
            st.subheader("Recent Anomalies")
            anomaly_display = anomalies[['timestamp', 'district', 'type', 'anomaly_score']].sort_values(
                'anomaly_score', ascending=False
            ).head(20)
            st.dataframe(anomaly_display, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Urban Issue Forecasting System | DSDE M150-Lover Team | Chulalongkorn University</p>
            <p>Data Source: Traffy Fondue (Aug 2021 - Jan 2025) | Last Updated: {}</p>
        </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


if __name__ == "__main__":
    main()