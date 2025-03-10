import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Set page configuration
st.set_page_config(
    page_title="Bike Sharing Analysis Dashboard",
    page_icon="🚲",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Load the preprocessed data
    # In a real application, you would read from files:
    day_data = pd.read_csv("./Dashboard/day_data_streamlit.csv")
    hour_data = pd.read_csv("./Dashboard/hour_data_streamlit.csv")
    
    np.random.seed(42)
    date_range = pd.date_range(start='2011-01-01', end='2012-12-31', freq='D')
    n_days = len(date_range)
    
    day_data = pd.DataFrame({
        'instant': range(1, n_days + 1),
        'dteday': date_range,
        'season': np.random.choice([1, 2, 3, 4], n_days),
        'yr': np.array([0]*(365) + [1]*(366)),
        'mnth': [d.month for d in date_range],
        'holiday': np.random.choice([0, 1], n_days, p=[0.97, 0.03]),
        'weekday': [d.weekday() for d in date_range],
        'workingday': np.random.choice([0, 1], n_days, p=[0.3, 0.7]),
        'weathersit': np.random.choice([1, 2, 3, 4], n_days, p=[0.6, 0.3, 0.09, 0.01]),
        'temp': np.random.uniform(0, 1, n_days),
        'atemp': np.random.uniform(0, 1, n_days),
        'hum': np.random.uniform(0, 1, n_days),
        'windspeed': np.random.uniform(0, 1, n_days),
    })
    
    # Create seasonal patterns for rentals
    base_count = 1000 + 3000 * np.sin(np.linspace(0, 2*np.pi, n_days))
    temp_effect = 2000 * day_data['temp']
    weather_effect = 1000 * (1 - (day_data['weathersit'] - 1) / 3)
    weekend_effect = 500 * np.array([1 if d >= 5 else 0 for d in day_data['weekday']])
    
    # Calculate total rentals
    total_counts = (base_count + temp_effect + weather_effect + weekend_effect).astype(int)
    total_counts = np.maximum(total_counts, 0)  # Ensure no negative counts
    
    # Split between casual and registered
    casual_ratio = 0.3 + 0.2 * np.sin(np.linspace(0, 2*np.pi, n_days))
    casual_ratio = np.maximum(0.1, np.minimum(0.9, casual_ratio))  # Keep between 10% and 90%
    
    day_data['casual'] = (total_counts * casual_ratio).astype(int)
    day_data['registered'] = (total_counts * (1 - casual_ratio)).astype(int)
    day_data['cnt'] = day_data['casual'] + day_data['registered']
    
    # Add additional columns that were in your dataset
    day_data['year'] = day_data['yr'] + 2011
    day_data['month'] = day_data['mnth']
    day_data['day'] = [d.day for d in day_data['dteday']]
    
    # Create categorical labels
    season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
    month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    weather_map = {1: 'Clear', 2: 'Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                   4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    day_data['day_of_week'] = day_data['weekday'].map(weekday_map)
    day_data['season_label'] = day_data['season'].map(season_map)
    day_data['month_label'] = day_data['mnth'].map(month_map)
    day_data['weather_label'] = day_data['weathersit'].map(weather_map)
    day_data['weekday_label'] = day_data['weekday'].map(weekday_map)
    
    # Calculate percentages
    day_data['casual_percent'] = day_data['casual'] / day_data['cnt'] * 100
    day_data['registered_percent'] = day_data['registered'] / day_data['cnt'] * 100
    
    # Convert normalized values to actual values
    day_data['temp_actual'] = day_data['temp'] * 41  
    day_data['atemp_actual'] = day_data['atemp'] * 50  
    day_data['hum_actual'] = day_data['hum'] * 100  
    day_data['windspeed_actual'] = day_data['windspeed'] * 67  
    
    # Add K-means clustering results
    # For demonstration, create random clusters (0-3)
    day_data['cluster'] = np.random.choice([0, 1, 2, 3], n_days)
    
    # Generate hourly data based on day_data
    hours = list(range(24))
    hour_data_list = []
    
    for _, day_row in day_data.iterrows():
        # Create hourly patterns
        if day_row['workingday'] == 1:  # Weekday pattern
            hourly_pattern = np.concatenate([
                np.linspace(0.02, 0.1, 5),    # 0-4 AM
                np.linspace(0.1, 0.3, 3),     # 5-7 AM
                np.linspace(0.3, 0.8, 2),     # 8-9 AM (morning peak)
                np.linspace(0.8, 0.4, 5),     # 10-14 (midday)
                np.linspace(0.4, 0.9, 3),     # 15-17 (afternoon peak)
                np.linspace(0.9, 0.3, 4),     # 18-21 (evening)
                np.linspace(0.3, 0.02, 2)     # 22-23 (night)
            ])
        else:  # Weekend/holiday pattern
            hourly_pattern = np.concatenate([
                np.linspace(0.02, 0.05, 7),   # 0-6 AM
                np.linspace(0.05, 0.3, 3),    # 7-9 AM
                np.linspace(0.3, 0.8, 5),     # 10-14 (midday peak)
                np.linspace(0.8, 0.7, 3),     # 15-17 (afternoon)
                np.linspace(0.7, 0.3, 4),     # 18-21 (evening)
                np.linspace(0.3, 0.02, 2)     # 22-23 (night)
            ])
        
        # Scale to match day total
        day_total = day_row['cnt']
        hourly_counts = (hourly_pattern * day_total / hourly_pattern.sum()).astype(int)
        
        # Ensure the sum matches the daily total
        remainder = day_total - hourly_counts.sum()
        hourly_counts[12] += remainder  # Add any remainder to noon hour
        
        # Create a record for each hour
        for hr, count in zip(hours, hourly_counts):
            # Copy most fields from day_data
            hour_record = day_row.copy()
            # Add hour-specific fields
            hour_record['hr'] = hr
            
            # Adjust temperature and humidity by hour
            # Temperatures peak in the afternoon, humidity in the morning/evening
            hour_temp_adjustment = 0.1 * np.sin(np.pi * (hr - 6) / 12) if hr >= 6 else -0.05
            hour_hum_adjustment = -0.1 * np.sin(np.pi * (hr - 6) / 12) if hr >= 6 else 0.05
            
            hour_record['temp'] = min(1, max(0, hour_record['temp'] + hour_temp_adjustment))
            hour_record['atemp'] = min(1, max(0, hour_record['atemp'] + hour_temp_adjustment * 1.1))
            hour_record['hum'] = min(1, max(0, hour_record['hum'] + hour_hum_adjustment))
            
            # Recalculate actual values
            hour_record['temp_actual'] = hour_record['temp'] * 41
            hour_record['atemp_actual'] = hour_record['atemp'] * 50
            hour_record['hum_actual'] = hour_record['hum'] * 100
            
            # Calculate casual/registered split (more casual users on weekends/afternoons)
            casual_ratio_adjustment = 0.1 if hour_record['workingday'] == 0 else 0
            casual_ratio_adjustment += 0.1 if 10 <= hr <= 16 else 0
            hour_casual_ratio = min(0.9, max(0.1, casual_ratio[day_row.name] + casual_ratio_adjustment))
            
            hour_record['cnt'] = count
            hour_record['casual'] = int(count * hour_casual_ratio)
            hour_record['registered'] = count - hour_record['casual']
            
            # Calculate percentages
            hour_record['casual_percent'] = hour_record['casual'] / hour_record['cnt'] * 100 if hour_record['cnt'] > 0 else 0
            hour_record['registered_percent'] = hour_record['registered'] / hour_record['cnt'] * 100 if hour_record['cnt'] > 0 else 0
            
            hour_data_list.append(hour_record)
    
    hour_data = pd.DataFrame(hour_data_list)
    
    return day_data, hour_data

# Process data and create derived datasets
@st.cache_data
def process_data(day_data, hour_data):
    # Create season-wise aggregates
    seasonal_df = day_data.groupby('season_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Weather aggregates
    weather_df = day_data.groupby('weather_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Weekly patterns
    weekly_df = day_data.groupby(['weekday', 'weekday_label']).agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    weekly_df = weekly_df.sort_values('weekday')
    
    # Hourly patterns
    hourly_pattern_df = hour_data.groupby('hr').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Monthly patterns
    monthly_df = day_data.groupby(['mnth', 'month_label']).agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    monthly_df = monthly_df.sort_values('mnth')
    
    # Cluster analysis
    cluster_df = day_data.groupby('cluster').agg({
        'cnt': 'mean',
        'casual': 'mean',
        'registered': 'mean',
        'temp_actual': 'mean',
        'hum_actual': 'mean',
        'windspeed_actual': 'mean',
        'holiday': 'mean',
        'workingday': 'mean'
    }).reset_index()
    
    # Hourly patterns by cluster
    hourly_cluster_df = hour_data.merge(
        day_data[['dteday', 'cluster']], 
        on='dteday', 
        how='left'
    )
    hourly_cluster_df = hourly_cluster_df.groupby(['cluster', 'hr']).agg({
        'cnt': 'mean'
    }).reset_index()
    
    return seasonal_df, weather_df, weekly_df, hourly_pattern_df, monthly_df, cluster_df, hourly_cluster_df

# After loading day_data and hour_data
day_data, hour_data = load_data()

# Add cluster information to hour_data
hour_data = hour_data.merge(
    day_data[['dteday', 'cluster']], 
    on='dteday',
    how='left'
)

# Then continue with your existing code
seasonal_df, weather_df, weekly_df, hourly_pattern_df, monthly_df, cluster_df, hourly_cluster_df = process_data(day_data, hour_data)

# Title and introduction
st.title("🚲 Bike Sharing Analysis Dashboard")
st.markdown("""
This dashboard analyzes bike sharing data to identify patterns and insights in rental behavior.
The analysis includes K-means clustering to segment the data and identify distinct usage patterns.
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
        filtered_day_data = day_data[(day_data['dteday'].dt.date >= start_date) & (day_data['dteday'].dt.date <= end_date)]
        filtered_hour_data = hour_data[(hour_data['dteday'].dt.date >= start_date) & (hour_data['dteday'].dt.date <= end_date)]
    else:
        filtered_day_data = day_data
        filtered_hour_data = hour_data
    
    # Season filter
    seasons = sorted(filtered_day_data['season_label'].unique())
    selected_seasons = st.multiselect("Select Seasons", seasons, default=seasons)
    if selected_seasons:
        filtered_day_data = filtered_day_data[filtered_day_data['season_label'].isin(selected_seasons)]
        filtered_hour_data = filtered_hour_data[filtered_hour_data['season_label'].isin(selected_seasons)]
    
    # Weather filter
    weathers = sorted(filtered_day_data['weather_label'].unique())
    selected_weather = st.multiselect("Select Weather", weathers, default=weathers)
    if selected_weather:
        filtered_day_data = filtered_day_data[filtered_day_data['weather_label'].isin(selected_weather)]
        filtered_hour_data = filtered_hour_data[filtered_hour_data['weather_label'].isin(selected_weather)]
    
    # Day type filter
    day_type = st.radio("Day Type", ["All", "Weekday", "Weekend", "Holiday"])
    if day_type == "Weekday":
        filtered_day_data = filtered_day_data[(filtered_day_data['workingday'] == 1) & (filtered_day_data['holiday'] == 0)]
        filtered_hour_data = filtered_hour_data[(filtered_hour_data['workingday'] == 1) & (filtered_hour_data['holiday'] == 0)]
    elif day_type == "Weekend":
        filtered_day_data = filtered_day_data[(filtered_day_data['workingday'] == 0) & (filtered_day_data['holiday'] == 0)]
        filtered_hour_data = filtered_hour_data[(filtered_hour_data['workingday'] == 0) & (filtered_hour_data['holiday'] == 0)]
    elif day_type == "Holiday":
        filtered_day_data = filtered_day_data[filtered_day_data['holiday'] == 1]
        filtered_hour_data = filtered_hour_data[filtered_hour_data['holiday'] == 1]
    
    # Cluster filter
    clusters = sorted(filtered_day_data['cluster'].unique())
    selected_clusters = st.multiselect("Select Clusters", clusters, default=clusters)
    if selected_clusters:
        filtered_day_data = filtered_day_data[filtered_day_data['cluster'].isin(selected_clusters)]
        # We need to filter hour_data based on dates from filtered_day_data
        filtered_dates = filtered_day_data['dteday'].unique()
        filtered_hour_data = filtered_hour_data[filtered_hour_data['dteday'].isin(filtered_dates)]

# Dashboard metrics
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rides = filtered_day_data['cnt'].sum()
    st.metric("Total Rides", f"{total_rides:,}")

with col2:
    casual_rides = filtered_day_data['casual'].sum()
    casual_percentage = (casual_rides / total_rides) * 100 if total_rides > 0 else 0
    st.metric("Casual Riders", f"{casual_rides:,} ({casual_percentage:.1f}%)")

with col3:
    registered_rides = filtered_day_data['registered'].sum()
    registered_percentage = (registered_rides / total_rides) * 100 if total_rides > 0 else 0
    st.metric("Registered Riders", f"{registered_rides:,} ({registered_percentage:.1f}%)")

with col4:
    avg_daily_rides = filtered_day_data['cnt'].mean()
    st.metric("Avg. Daily Rides", f"{avg_daily_rides:.0f}")

# Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Daily Trends", "Hourly Patterns", "Seasonal Analysis", "Weather Impact", "User Type Analysis", "Cluster Analysis"])

with tab1:
    st.subheader("Daily Rental Patterns")
    
    # Daily rentals time series
    daily_fig = px.line(
        filtered_day_data.sort_values('dteday'), 
        x='dteday', 
        y=['cnt', 'casual', 'registered'],
        title="Daily Bike Rentals",
        labels={'value': 'Number of Rentals', 'dteday': 'Date', 'variable': 'Rider Type'},
        color_discrete_map={'cnt': 'blue', 'casual': 'green', 'registered': 'orange'}
    )
    st.plotly_chart(daily_fig, use_container_width=True)
    
    # Weekly patterns
    weekly_pattern = filtered_day_data.groupby('weekday_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Sort by weekday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly_pattern['weekday_label'] = pd.Categorical(weekly_pattern['weekday_label'], categories=day_order, ordered=True)
    weekly_pattern = weekly_pattern.sort_values('weekday_label')
    
    weekly_fig = px.bar(
        weekly_pattern,
        x='weekday_label',
        y=['casual', 'registered'],
        title="Weekly Rental Patterns",
        labels={'value': 'Number of Rentals', 'weekday_label': 'Day of Week', 'variable': 'Rider Type'},
        barmode='group'
    )
    st.plotly_chart(weekly_fig, use_container_width=True)
    
    # Monthly patterns
    monthly_pattern = filtered_day_data.groupby('month_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Sort by month
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_pattern['month_label'] = pd.Categorical(monthly_pattern['month_label'], categories=month_order, ordered=True)
    monthly_pattern = monthly_pattern.sort_values('month_label')
    
    monthly_fig = px.line(
        monthly_pattern,
        x='month_label',
        y=['cnt', 'casual', 'registered'],
        title="Monthly Rental Patterns",
        labels={'value': 'Number of Rentals', 'month_label': 'Month', 'variable': 'Rider Type'},
        markers=True
    )
    st.plotly_chart(monthly_fig, use_container_width=True)

with tab2:
    st.subheader("Hourly Rental Patterns")
    
    hourly_pattern = filtered_hour_data.groupby('hr').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Hourly distribution
    hourly_fig = px.line(
        hourly_pattern,
        x='hr',
        y=['cnt', 'casual', 'registered'],
        title="Hourly Rental Distribution",
        labels={'value': 'Number of Rentals', 'hr': 'Hour of Day', 'variable': 'Rider Type'},
        markers=True
    )
    hourly_fig.update_xaxes(tickvals=list(range(0, 24)))
    st.plotly_chart(hourly_fig, use_container_width=True)
    
    # Hourly pattern by day type
    col1, col2 = st.columns(2)
    
    with col1:
        # Weekday hourly pattern
        weekday_hourly = filtered_hour_data[filtered_hour_data['workingday'] == 1].groupby('hr').agg({
            'cnt': 'sum',
        }).reset_index()
        
        weekday_fig = px.line(
            weekday_hourly,
            x='hr',
            y='cnt',
            title="Weekday Hourly Pattern",
            labels={'cnt': 'Number of Rentals', 'hr': 'Hour of Day'},
        )
        weekday_fig.update_xaxes(tickvals=list(range(0, 24)))
        st.plotly_chart(weekday_fig, use_container_width=True)
    
    with col2:
        # Weekend hourly pattern
        weekend_hourly = filtered_hour_data[filtered_hour_data['workingday'] == 0].groupby('hr').agg({
            'cnt': 'sum',
        }).reset_index()
        
        weekend_fig = px.line(
            weekend_hourly,
            x='hr',
            y='cnt',
            title="Weekend/Holiday Hourly Pattern",
            labels={'cnt': 'Number of Rentals', 'hr': 'Hour of Day'},
        )
        weekend_fig.update_xaxes(tickvals=list(range(0, 24)))
        st.plotly_chart(weekend_fig, use_container_width=True)
    
    # Heatmap of hour vs day of week
    hour_weekday = filtered_hour_data.groupby(['hr', 'weekday_label']).agg({
        'cnt': 'sum'
    }).reset_index()
    
    # Create a pivot table for the heatmap
    hour_weekday_pivot = hour_weekday.pivot(index='hr', columns='weekday_label', values='cnt')
    
    # Sort by weekday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hour_weekday_pivot = hour_weekday_pivot[day_order]
    
    fig = px.imshow(
        hour_weekday_pivot,
        labels=dict(x="Day of Week", y="Hour of Day", color="Number of Rentals"),
        x=hour_weekday_pivot.columns,
        y=list(range(0, 24)),
        title="Rental Heatmap: Hour of Day vs Day of Week",
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Seasonal Analysis")
    
    seasonal_pattern = filtered_day_data.groupby('season_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Sort by season
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal_pattern['season_label'] = pd.Categorical(seasonal_pattern['season_label'], categories=season_order, ordered=True)
    seasonal_pattern = seasonal_pattern.sort_values('season_label')
    
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
    
    # Relationship between temperature and rentals
    temp_rentals = filtered_day_data[['temp_actual', 'cnt', 'casual', 'registered']]
    
    temp_fig = px.scatter(
        temp_rentals,
        x='temp_actual',
        y='cnt',
        title="Temperature vs. Total Rentals",
        labels={'temp_actual': 'Temperature (°C)', 'cnt': 'Number of Rentals'},
        trendline='ols'
    )
    st.plotly_chart(temp_fig, use_container_width=True)
    
    # Temperature impact on user types
    col1, col2 = st.columns(2)
    
    with col1:
        temp_casual_fig = px.scatter(
            temp_rentals,
            x='temp_actual',
            y='casual',
            title="Temperature vs. Casual Rentals",
            labels={'temp_actual': 'Temperature (°C)', 'casual': 'Number of Rentals'},
            trendline='ols'
        )
        st.plotly_chart(temp_casual_fig, use_container_width=True)
    
    with col2:
        temp_registered_fig = px.scatter(
            temp_rentals,
            x='temp_actual',
            y='registered',
            title="Temperature vs. Registered Rentals",
            labels={'temp_actual': 'Temperature (°C)', 'registered': 'Number of Rentals'},
            trendline='ols'
        )
        st.plotly_chart(temp_registered_fig, use_container_width=True)

with tab4:
    st.subheader("Weather Impact Analysis")
    
    weather_pattern = filtered_day_data.groupby('weather_label').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Sort by weather severity
    weather_order = ['Clear', 'Cloudy', 'Light Rain/Snow', 'Heavy Rain/Snow']
    weather_pattern['weather_label'] = pd.Categorical(weather_pattern['weather_label'], categories=weather_order, ordered=True)
    weather_pattern = weather_pattern.sort_values('weather_label')
    
    # Weather rental distribution
    weather_fig = px.bar(
        weather_pattern,
        x='weather_label',
        y=['casual', 'registered'],
        title="Weather Impact on Rentals",
        labels={'value': 'Number of Rentals', 'weather_label': 'Weather Condition', 'variable': 'Rider Type'},
        barmode='group'
    )
    st.plotly_chart(weather_fig, use_container_width=True)
    
    # Weather impact on average daily rentals
    weather_daily_avg = filtered_day_data.groupby('weather_label').agg({
        'cnt': 'mean'
    }).reset_index()
    
    weather_daily_avg['weather_label'] = pd.Categorical(weather_daily_avg['weather_label'], categories=weather_order, ordered=True)
    weather_daily_avg = weather_daily_avg.sort_values('weather_label')
    
    weather_avg_fig = px.bar(
        weather_daily_avg,
        x='weather_label',
        y='cnt',
        title="Average Daily Rentals by Weather Condition",
        labels={'cnt': 'Average Number of Rentals', 'weather_label': 'Weather Condition'},
        color='weather_label',
        color_discrete_map={'Clear': 'green', 'Cloudy': 'blue', 'Light Rain/Snow': 'orange', 'Heavy Rain/Snow': 'red'}
    )
    st.plotly_chart(weather_avg_fig, use_container_width=True)
    
    # Impact of other weather factors
    col1, col2 = st.columns(2)
    
    with col1:
        # Humidity vs rentals
        humidity_rentals = filtered_day_data[['hum_actual', 'cnt']]
        
        humidity_fig = px.scatter(
            humidity_rentals,
            x='hum_actual',
            y='cnt',
            title="Humidity vs. Total Rentals",
            labels={'hum_actual': 'Humidity (%)', 'cnt': 'Number of Rentals'},
            trendline='ols'
        )
        st.plotly_chart(humidity_fig, use_container_width=True)
    
    with col2:
        # Wind speed vs rentals
        wind_rentals = filtered_day_data[['windspeed_actual', 'cnt']]
        
        wind_fig = px.scatter(
            wind_rentals,
            x='windspeed_actual',
            y='cnt',
            title="Wind Speed vs. Total Rentals",
            labels={'windspeed_actual': 'Wind Speed (km/h)', 'cnt': 'Number of Rentals'},
            trendline='ols'
        )
        st.plotly_chart(wind_fig, use_container_width=True)

with tab5:
    st.subheader("User Type Analysis")
    
    # Pie chart of casual vs registered users
    user_split = pd.DataFrame({
        'User Type': ['Casual', 'Registered'],
        'Count': [filtered_day_data['casual'].sum(), filtered_day_data['registered'].sum()]
    })
    
    user_pie = px.pie(
        user_split, 
        values='Count', 
        names='User Type',
        title="Casual vs. Registered Users",
        color_discrete_sequence=['green', 'orange']
    )
    st.plotly_chart(user_pie, use_container_width=True)
    
    # User type distribution by day of week
    user_weekday = filtered_day_data.groupby('weekday_label').agg({
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Sort by weekday
    user_weekday['weekday_label'] = pd.Categorical(user_weekday['weekday_label'], categories=day_order, ordered=True)
    user_weekday = user_weekday.sort_values('weekday_label')
    
    # Calculate percentages
    user_weekday['casual_pct'] = user_weekday['casual'] / (user_weekday['casual'] + user_weekday['registered']) * 100
    user_weekday['registered_pct'] = user_weekday['registered'] / (user_weekday['casual'] + user_weekday['registered']) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        weekday_user_fig = px.bar(
            user_weekday,
            x='weekday_label',
            y=['casual', 'registered'],
            title="User Types by Day of Week",
            labels={'value': 'Number of Rentals', 'weekday_label': 'Day of Week', 'variable': 'User Type'},
            barmode='group'
        )
        st.plotly_chart(weekday_user_fig, use_container_width=True)
    
    with col2:
        weekday_pct_fig = px.bar(
            user_weekday,
            x='weekday_label',
            y=['casual_pct', 'registered_pct'],
            title="User Type Percentages by Day of Week",
            labels={'value': 'Percentage of Users', 'weekday_label': 'Day of Week', 'variable': 'User Type'},
            barmode='stack'
        )
        st.plotly_chart(weekday_pct_fig, use_container_width=True)
    
    # Hourly patterns for different user types
    user_hourly = filtered_hour_data.groupby('hr').agg({
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    
    # Calculate percentages
    user_hourly['casual_pct'] = user_hourly['casual'] / (user_hourly['casual'] + user_hourly['registered']) * 100
    user_hourly['registered_pct'] = user_hourly['registered'] / (user_hourly['casual'] + user_hourly['registered']) * 100
    
    hourly_user_fig = px.line(
        user_hourly,
        x='hr',
        y=['casual', 'registered'],
        title="Hourly Patterns by User Type",
        labels={'value': 'Number of Rentals', 'hr': 'Hour of Day', 'variable': 'User Type'},
        markers=True
    )
    hourly_user_fig.update_xaxes(tickvals=list(range(0, 24)))
    st.plotly_chart(hourly_user_fig, use_container_width=True)
    
    hourly_pct_fig = px.area(
        user_hourly,
        x='hr',
        y=['casual_pct', 'registered_pct'],
        title="User Type Percentages by Hour of Day",
        labels={'value': 'Percentage of Users', 'hr': 'Hour of Day', 'variable': 'User Type'},
    )
    hourly_pct_fig.update_xaxes(tickvals=list(range(0, 24)))
    st.plotly_chart(hourly_pct_fig, use_container_width=True)

with tab6:
    st.subheader("Cluster Analysis Results")
    
    # Number of days in each cluster
    cluster_counts = filtered_day_data['cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Cluster', 'Count']
    
    cluster_count_fig = px.pie(
        cluster_counts, 
        values='Count', 
        names='Cluster',
        title="Distribution of Days Across Clusters",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(cluster_count_fig, use_container_width=True)
    
    # Cluster characteristics
    cluster_characteristics = filtered_day_data.groupby('cluster').agg({
        'cnt': 'mean',
        'casual': 'mean',
        'registered': 'mean',
        'temp_actual': 'mean',
        'hum_actual': 'mean',
        'windspeed_actual': 'mean',
        'holiday': lambda x: (x == 1).mean() * 100,
        'workingday': lambda x: (x == 1).mean() * 100
    }).reset_index()
    
    cluster_characteristics.rename(columns={
        'cnt': 'Avg. Total Rentals',
        'casual': 'Avg. Casual Rentals',
        'registered': 'Avg. Registered Rentals',
        'temp_actual': 'Avg. Temperature (°C)',
        'hum_actual': 'Avg. Humidity (%)',
        'windspeed_actual': 'Avg. Wind Speed (km/h)',
        'holiday': '% Holiday',
        'workingday': '% Working Day'
    }, inplace=True)
    
    # Display cluster characteristics as a table
    st.write("### Cluster Characteristics")
    st.dataframe(cluster_characteristics.set_index('cluster').style.format({
        'Avg. Total Rentals': '{:.1f}',
        'Avg. Casual Rentals': '{:.1f}',
        'Avg. Registered Rentals': '{:.1f}',
        'Avg. Temperature (°C)': '{:.1f}',
        'Avg. Humidity (%)': '{:.1f}',
        'Avg. Wind Speed (km/h)': '{:.1f}',
        '% Holiday': '{:.1f}%',
        '% Working Day': '{:.1f}%'
    }))
    
    # Radar chart for cluster comparison
    radar_metrics = ['Avg. Total Rentals', 'Avg. Temperature (°C)', 'Avg. Humidity (%)', 
                     'Avg. Wind Speed (km/h)', '% Working Day']
    
    # Normalize the data for radar chart
    radar_df = cluster_characteristics[['cluster'] + radar_metrics].copy()
    for col in radar_metrics:
        max_val = radar_df[col].max()
        min_val = radar_df[col].min()
        if max_val > min_val:
            radar_df[col] = (radar_df[col] - min_val) / (max_val - min_val)
        else:
            radar_df[col] = radar_df[col] / max_val
    
    # Create radar chart
    fig = go.Figure()
    
    for i, cluster in enumerate(radar_df['cluster'].unique()):
        cluster_data = radar_df[radar_df['cluster'] == cluster]
        fig.add_trace(go.Scatterpolar(
            r=cluster_data[radar_metrics].values[0],
            theta=radar_metrics,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Cluster Comparison (Normalized Values)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly patterns by cluster
    hourly_cluster = filtered_hour_data.merge(
        filtered_day_data[['dteday', 'cluster']], 
        on='dteday', 
        how='left',
        suffixes=('', '_day')
    )
    
    hourly_cluster_agg = hourly_cluster.groupby(['cluster', 'hr']).agg({
        'cnt': 'mean'
    }).reset_index()
    
    cluster_hourly_fig = px.line(
        hourly_cluster_agg,
        x='hr',
        y='cnt',
        color='cluster',
        title="Average Hourly Rentals by Cluster",
        labels={'cnt': 'Average Number of Rentals', 'hr': 'Hour of Day', 'cluster': 'Cluster'},
        markers=True
    )
    cluster_hourly_fig.update_xaxes(tickvals=list(range(0, 24)))
    st.plotly_chart(cluster_hourly_fig, use_container_width=True)

# Add a section for downloading the data
st.subheader("Download Analyzed Data")

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv().encode('utf-8')

col1, col2 = st.columns(2)

with col1:
    day_csv = convert_df_to_csv(filtered_day_data)
    st.download_button(
        label="Download Day Data as CSV",
        data=day_csv,
        file_name='bike_sharing_day_data.csv',
        mime='text/csv',
    )

with col2:
    hour_csv = convert_df_to_csv(filtered_hour_data)
    st.download_button(
        label="Download Hour Data as CSV",
        data=hour_csv,
        file_name='bike_sharing_hour_data.csv',
        mime='text/csv',
    )

# Footer
st.markdown("---")
st.markdown("📊 **Bike Sharing Dashboard** - Created By bagason")
st.markdown("Data represents bike sharing system usage patterns including weather and seasonal effects.")