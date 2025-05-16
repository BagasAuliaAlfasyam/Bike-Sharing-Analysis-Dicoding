import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Bike Sharing Analysis Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Load the actual data file
    day_data = pd.read_csv("./Dashboard/main_data.csv")
    
    # Convert date column to datetime
    day_data['dteday'] = pd.to_datetime(day_data['dteday'])
    
    # Create necessary derived columns for analysis
    
    # Create season labels
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    day_data['season_label'] = day_data['season'].map(season_map)
    
    # Create month labels
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    day_data['month_label'] = day_data['mnth'].map(month_map)
    
    # Create weather labels
    weather_map = {1: 'Clear', 2: 'Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    day_data['weather_label'] = day_data['weathersit'].map(weather_map)
    
    # Create weekday labels
    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    day_data['weekday_label'] = day_data['weekday'].map(weekday_map)
    
    # Create temperature bins for analysis
    day_data['temp_bin'] = pd.cut(
        day_data['temp_actual'] if 'temp_actual' in day_data.columns else day_data['temp'] * 41,
        bins=[0, 10, 20, 30, 40],
        labels=['Cold (0-10°C)', 'Cool (10-20°C)', 'Warm (20-30°C)', 'Hot (30-40°C)']
    )
    
    return day_data

# Load the data
day_data = load_data()

# Title and introduction
st.title("🚲 Bike Sharing Analysis Dashboard")
st.markdown("""
This dashboard analyzes bike sharing data to identify patterns and insights in rental behavior.
The analysis addresses key business questions about seasonal patterns, user types, peak usage times, and factors influencing rentals.
Use the filters in the sidebar to explore different aspects of the data.
""")

# Sidebar filters
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=100)
    st.title("Filters")
    
    # Date range selector
    min_date = day_data['dteday'].min().date()
    max_date = day_data['dteday'].max().date()
    date_range = st.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_data = day_data[(day_data['dteday'].dt.date >= start_date) & (day_data['dteday'].dt.date <= end_date)]
    else:
        filtered_data = day_data
    
    # Season filter
    seasons = sorted(filtered_data['season_label'].unique())
    selected_seasons = st.multiselect("Select Seasons", seasons, default=seasons)
    if selected_seasons:
        filtered_data = filtered_data[filtered_data['season_label'].isin(selected_seasons)]
    
    # Weather filter
    weathers = sorted(filtered_data['weather_label'].unique())
    selected_weather = st.multiselect("Select Weather", weathers, default=weathers)
    if selected_weather:
        filtered_data = filtered_data[filtered_data['weather_label'].isin(selected_weather)]
    
    # Day type filter
    day_type = st.radio("Day Type", ["All", "Weekday", "Weekend", "Holiday"])
    if day_type == "Weekday":
        filtered_data = filtered_data[(filtered_data['workingday'] == 1) & (filtered_data['holiday'] == 0)]
    elif day_type == "Weekend":
        filtered_data = filtered_data[(filtered_data['workingday'] == 0) & (filtered_data['holiday'] == 0)]
    elif day_type == "Holiday":
        filtered_data = filtered_data[filtered_data['holiday'] == 1]

# Dashboard metrics
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rides = filtered_data['cnt'].sum()
    st.metric("Total Rides", f"{total_rides:,}")

with col2:
    casual_rides = filtered_data['casual'].sum()
    casual_percentage = (casual_rides / total_rides) * 100 if total_rides > 0 else 0
    st.metric("Casual Riders", f"{casual_rides:,} ({casual_percentage:.1f}%)")

with col3:
    registered_rides = filtered_data['registered'].sum()
    registered_percentage = (registered_rides / total_rides) * 100 if total_rides > 0 else 0
    st.metric("Registered Riders", f"{registered_rides:,} ({registered_percentage:.1f}%)")

with col4:
    avg_daily_rides = filtered_data['cnt'].mean()
    st.metric("Avg. Daily Rides", f"{avg_daily_rides:.0f}")

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Seasonal & Weather Analysis", 
    "User Type Analysis", 
    "Peak Usage Times",
    "Factors Influencing Rentals"
])

# Tab 1: Seasonal & Weather Analysis
with tab1:
    st.subheader("How do bicycle usage patterns vary by season and weather conditions?")
    
    # Seasonal analysis
    seasonal_pattern = filtered_data.groupby('season_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Sort by season
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_pattern['season_label'] = pd.Categorical(seasonal_pattern['season_label'], categories=season_order, ordered=True)
    seasonal_pattern = seasonal_pattern.sort_values('season_label')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Season rental distribution
        seasonal_fig = px.bar(
            seasonal_pattern,
            x='season_label',
            y=['casual', 'registered'],
            title="Seasonal Rental Distribution",
            labels={'value': 'Number of Rentals', 'season_label': 'Season', 'variable': 'Rider Type'},
            barmode='group'
        )
        st.plotly_chart(seasonal_fig, use_container_width=True)
    
    with col2:
        # Weather impact
        weather_pattern = filtered_data.groupby('weather_label').agg({
            'cnt': 'sum',
            'casual': 'sum',
            'registered': 'sum'
        }).reset_index()
        
        weather_fig = px.bar(
            weather_pattern,
            x='weather_label',
            y=['casual', 'registered'],
            title="Weather Impact on Rentals",
            labels={'value': 'Number of Rentals', 'weather_label': 'Weather Condition', 'variable': 'Rider Type'},
            barmode='group'
        )
        st.plotly_chart(weather_fig, use_container_width=True)
    
    # Heatmap of season vs weather
    season_weather = filtered_data.groupby(['season_label', 'weather_label']).agg({
        'cnt': 'mean'
    }).reset_index()
    
    # Create a pivot table for the heatmap
    season_weather_pivot = season_weather.pivot(index='season_label', columns='weather_label', values='cnt')
    
    # Sort by season
    season_weather_pivot = season_weather_pivot.reindex(season_order)
    
    season_weather_fig = px.imshow(
        season_weather_pivot,
        labels=dict(x="Weather Condition", y="Season", color="Average Rentals"),
        x=season_weather_pivot.columns,
        y=season_weather_pivot.index,
        title="Average Rentals by Season and Weather",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(season_weather_fig, use_container_width=True)

# Tab 2: User Type Analysis
with tab2:
    st.subheader("What are the differences in behavior between casual and registered users?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User type distribution by season
        seasonal_user_fig = px.line(
            seasonal_pattern,
            x='season_label',
            y=['casual', 'registered'],
            title="Seasonal Usage Patterns by User Type",
            labels={'value': 'Number of Rentals', 'season_label': 'Season', 'variable': 'Rider Type'},
            markers=True
        )
        st.plotly_chart(seasonal_user_fig, use_container_width=True)
    
    with col2:
        # User type distribution by day of week
        weekly_pattern = filtered_data.groupby('weekday_label').agg({
            'casual': 'sum',
            'registered': 'sum'
        }).reset_index()
        
        # Sort by weekday
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern['weekday_label'] = pd.Categorical(weekly_pattern['weekday_label'], categories=day_order, ordered=True)
        weekly_pattern = weekly_pattern.sort_values('weekday_label')
        
        weekly_user_fig = px.bar(
            weekly_pattern,
            x='weekday_label',
            y=['casual', 'registered'],
            title="Weekly Usage Patterns by User Type",
            labels={'value': 'Number of Rentals', 'weekday_label': 'Day of Week', 'variable': 'Rider Type'},
            barmode='group'
        )
        st.plotly_chart(weekly_user_fig, use_container_width=True)
    
    # User type ratio analysis
    filtered_data['casual_pct'] = filtered_data['casual'] / filtered_data['cnt'] * 100
    filtered_data['registered_pct'] = filtered_data['registered'] / filtered_data['cnt'] * 100
    
    # User type ratio by month
    monthly_pattern = filtered_data.groupby('month_label').agg({
        'casual_pct': 'mean',
        'registered_pct': 'mean'
    }).reset_index()
    
    # Sort by month
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pattern['month_label'] = pd.Categorical(monthly_pattern['month_label'], categories=month_order, ordered=True)
    monthly_pattern = monthly_pattern.sort_values('month_label')
    
    monthly_ratio_fig = px.bar(
        monthly_pattern,
        x='month_label',
        y=['casual_pct', 'registered_pct'],
        title="Monthly User Type Ratio",
        labels={'value': 'Percentage of Total Rides', 'month_label': 'Month', 'variable': 'Rider Type'},
        barmode='stack'
    )
    st.plotly_chart(monthly_ratio_fig, use_container_width=True)

# Tab 3: Peak Usage Times
with tab3:
    st.subheader("What are the peak times for bicycle usage and how do weekdays affect rentals?")
    
    # Daily rentals time series
    daily_fig = px.line(
        filtered_data.sort_values('dteday'), 
        x='dteday', 
        y='cnt',
        title="Daily Bike Rentals Over Time",
        labels={'cnt': 'Number of Rentals', 'dteday': 'Date'},
        color_discrete_sequence=['blue']
    )
    st.plotly_chart(daily_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Monthly patterns
        monthly_usage = filtered_data.groupby('month_label').agg({
            'cnt': 'sum'
        }).reset_index()
        
        # Sort by month
        monthly_usage['month_label'] = pd.Categorical(monthly_usage['month_label'], categories=month_order, ordered=True)
        monthly_usage = monthly_usage.sort_values('month_label')
        
        monthly_fig = px.bar(
            monthly_usage,
            x='month_label',
            y='cnt',
            title="Monthly Rental Patterns",
            labels={'cnt': 'Number of Rentals', 'month_label': 'Month'},
            color_discrete_sequence=['darkblue']
        )
        st.plotly_chart(monthly_fig, use_container_width=True)
    
    with col2:
        # Weekly patterns
        weekly_usage = filtered_data.groupby('weekday_label').agg({
            'cnt': 'sum'
        }).reset_index()
        
        # Sort by weekday
        weekly_usage['weekday_label'] = pd.Categorical(weekly_usage['weekday_label'], categories=day_order, ordered=True)
        weekly_usage = weekly_usage.sort_values('weekday_label')
        
        weekly_fig = px.bar(
            weekly_usage,
            x='weekday_label',
            y='cnt',
            title="Weekly Rental Patterns",
            labels={'cnt': 'Number of Rentals', 'weekday_label': 'Day of Week'},
            color_discrete_sequence=['darkgreen']
        )
        st.plotly_chart(weekly_fig, use_container_width=True)
    
    # Compare weekday vs weekend patterns
    weekday_data = filtered_data[filtered_data['workingday'] == 1]
    weekend_data = filtered_data[filtered_data['workingday'] == 0]
    
    weekday_avg = weekday_data['cnt'].mean()
    weekend_avg = weekend_data['cnt'].mean()
    
    day_type_comparison = pd.DataFrame({
        'Day Type': ['Weekday', 'Weekend/Holiday'],
        'Average Rentals': [weekday_avg, weekend_avg]
    })
    
    day_type_fig = px.bar(
        day_type_comparison,
        x='Day Type',
        y='Average Rentals',
        title="Average Rentals: Weekday vs Weekend/Holiday",
        labels={'Average Rentals': 'Average Number of Rentals', 'Day Type': 'Day Type'},
        color_discrete_sequence=['purple']
    )
    st.plotly_chart(day_type_fig, use_container_width=True)

# Tab 4: Factors Influencing Rentals
with tab4:
    st.subheader("What are the main factors that influence the number of bicycle rentals?")
    
    # Temperature effect
    col1, col2 = st.columns(2)
    
    with col1:
        # Temperature bins effect
        temp_effect = filtered_data.groupby('temp_bin').agg({
            'cnt': 'mean'
        }).reset_index()
        
        temp_fig = px.bar(
            temp_effect,
            x='temp_bin',
            y='cnt',
            title="Temperature Effect on Average Rentals",
            labels={'cnt': 'Average Number of Rentals', 'temp_bin': 'Temperature Range'},
            color_discrete_sequence=['orange']
        )
        st.plotly_chart(temp_fig, use_container_width=True)
    
    with col2:
        # Humidity effect
        humidity_bins = pd.cut(
            filtered_data['hum'] * 100 if max(filtered_data['hum']) <= 1 else filtered_data['hum'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Very High (75-100%)']
        )
        
        humidity_data = filtered_data.copy()
        humidity_data['humidity_bin'] = humidity_bins
        
        humidity_effect = humidity_data.groupby('humidity_bin').agg({
            'cnt': 'mean'
        }).reset_index()
        
        humidity_fig = px.bar(
            humidity_effect,
            x='humidity_bin',
            y='cnt',
            title="Humidity Effect on Average Rentals",
            labels={'cnt': 'Average Number of Rentals', 'humidity_bin': 'Humidity Range'},
            color_discrete_sequence=['teal']
        )
        st.plotly_chart(humidity_fig, use_container_width=True)
    
    # Correlation heatmap for numerical features
    numerical_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    corr_matrix = filtered_data[numerical_cols].corr()
    
    corr_fig = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        title="Correlation Between Factors",
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    st.plotly_chart(corr_fig, use_container_width=True)
    
    # Interactive scatter plot for exploring relationships
    st.subheader("Explore Relationships Between Factors")
    
    x_axis = st.selectbox('Select X-axis variable:', 
                         options=['temp', 'atemp', 'hum', 'windspeed'],
                         index=0)
    
    y_axis = st.selectbox('Select Y-axis variable:', 
                         options=['cnt', 'casual', 'registered'],
                         index=0)
    
    color_by = st.selectbox('Color by:',
                           options=['season_label', 'weather_label', 'workingday'],
                           index=0)
    
    scatter_fig = px.scatter(
        filtered_data,
        x=x_axis,
        y=y_axis,
        color=color_by,
        opacity=0.7,
        title=f"Relationship Between {x_axis} and {y_axis}",
        labels={x_axis: x_axis, y_axis: y_axis, color_by: color_by},
        trendline="ols"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("### Data Insights")
st.markdown("""
Based on the analysis, we can draw the following conclusions:

1. **Seasonal Patterns**: Bike rentals show strong seasonal patterns with higher usage during warmer months.
2. **Weather Impact**: Clear weather significantly increases bike usage compared to rainy or snowy conditions.
3. **User Types**: Registered users show consistent usage on weekdays, while casual users prefer weekends.
4. **Peak Times**: Weekday peaks align with commuting hours, while weekend usage is more evenly distributed.
5. **Key Factors**: Temperature has the strongest positive correlation with rentals, while humidity and adverse weather have negative impacts.
""")