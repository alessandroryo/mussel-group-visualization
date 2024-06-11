
# Streamlit for creating the web application interface
import streamlit as st

# Pandas for data manipulation and analysis
import pandas as pd

# Folium and streamlit_folium for interactive maps
import folium
from streamlit_folium import folium_static, st_folium

# NumPy for numerical operations
import numpy as np

# Plotly for creating interactive plots and charts
import plotly.graph_objects as go
import plotly.express as px

# Matplotlib for additional plotting utilities
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Joblib for model serialization
import joblib

# Hashlib for hashing access codes
import hashlib

# Math functions for distance calculations
from math import radians, sin, cos, sqrt, atan2

# OS for file operations (not used, can be removed if not needed)
import os


access_granted = False

st.set_page_config(layout="wide", page_title="Mussel Growth Trends Dashboard", page_icon="🐚")

def verify_access_code(input_code):
    """
    Verifies the access code provided by the user.

    This function hashes the input code using SHA256 and compares it to a pre-defined 
    hashed access code. It is used to ensure that only users with the correct access 
    code can proceed to use the application.

    Args:
        input_code (str): The access code input by the user.

    Returns:
        bool: True if the input code matches the pre-defined hashed access code, False otherwise.
    """
    # Hash the input code using SHA256
    hashed_input = hashlib.sha256(input_code.encode()).hexdigest()

    # Pre-defined hashed version of the correct access code
    # Note: To generate this hash for a new access code, use the following code snippet in a separate cell:
    # import hashlib
    # correct_code = 'your_desired_access_code'
    # print(hashlib.sha256(correct_code.encode()).hexdigest())
    hashed_access_code = 'd0601b8185961d8e3e0ea3bbbeca893e630abfa2f9617a5122a431f498bcfe6e'

    # Compare the hashed input code with the pre-defined hashed access code
    return hashed_input == hashed_access_code

def load_trained_model(file_path):
    """
    Loads the trained model from a file.

    Args:
        file_path (str): The path to the model file.

    Returns:
        model: The loaded model.
    """
    return joblib.load(file_path)

def load_data(file):
    """
    Loads the data from a CSV file.

    Args:
        file (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file)

def create_sidebar(df):
    """
    Creates a sidebar in the Streamlit app for user settings, allowing selection of years and locations.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.

    Returns:
        tuple: A tuple containing the selected year range and location selection.
    """
    # Set the title and header for the sidebar
    st.sidebar.title("Settings")
    
    # Add spacing
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Multiselect widget for selecting year(s)
    year_range = st.sidebar.multiselect(
        '📅 Select Year(s)',
        options=list(range(int(df['Year'].min()), int(df['Year'].max()) + 1)), 
        default=list(range(int(df['Year'].min()), int(df['Year'].max()) + 1))
    )
    
    # Instruction for year selection
    st.sidebar.markdown("""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; font-size: 12px; margin-top: -10px;">
    <b>Note</b>: Selecting more than one year will average the values for all selected years.
    </div>
    """, unsafe_allow_html=True)
    
    # Add spacing
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Prepare location options by combining 'System' and 'Plot Location'
    location_options = df[['System', 'Plot Location']].drop_duplicates()
    location_options['Location_System'] = location_options['System'] + ' - ' + location_options['Plot Location']
    
    # Multiselect widget for selecting locations
    location_selection = st.sidebar.multiselect(
        '📍 Select Locations',
        options=location_options['Location_System'].sort_values()
    )
    
    # Instruction for location selection
    st.sidebar.markdown("""
    <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; font-size: 12px; margin-top: -10px;">
    <b>Note</b>: If no location is selected, the data will be averaged per system.
    Selecting more than one location will display data per location.
    </div>
    """, unsafe_allow_html=True)
    
    # Return the selected year range and location selection
    return year_range, location_selection

def display_main_map(df, year_range, location_selection):
    """
    Displays an interactive map with bubbles representing mussel growth data, colored by a selected feature and animated by month.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        year_range (list): List of selected years.
        location_selection (list): List of selected locations.
    """
    # User selection for the feature to color the map bubbles
    selected_color_feature = st.selectbox('Select feature for bubble color', [
        'Precipitation', 'Sea Surface Temperature (C)', 'Chlorophyll (mg per m3)', 'Turbidity (FTU)'
    ], help="The color scale represents the selected feature, indicating its intensity.")
    
    # Add a custom information box
    st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-size: 16px; margin-bottom: 20px;">
            <strong>Note</strong>: Color scale indicates the intensity of the selected feature, and bubble size represents the magnitude of growth (g per day).
        </div>
    """, unsafe_allow_html=True)

    # Filter the dataframe based on the selected years
    df_filtered = df[df['Year'].isin(year_range)]

    # If specific locations are selected, filter the dataframe accordingly
    if location_selection:
        df_filtered['Location_System'] = df_filtered['System'] + ' - ' + df_filtered['Plot Location']
        df_filtered = df_filtered[df_filtered['Location_System'].isin(location_selection)]

    # Exclude data for Monitoring Period 0
    df_filtered = df_filtered[df_filtered['Monitoring Period'] > 0]

    # Group the filtered data by specific columns and calculate the mean for each group
    df_grouped = df_filtered.groupby([
        'lat', 'lon', 'Plot Location', 'System', 'Month', 'Monitoring Period'
    ]).mean().round(2).reset_index()

    # Rename columns for better readability in the hover information
    df_grouped.rename(columns={'lat': 'Latitude', 'lon': 'Longitude'}, inplace=True)

    # Sort the grouped data by Monitoring Period
    df_grouped.sort_values(by='Monitoring Period', inplace=True)

    # Create the scatter mapbox figure using Plotly Express
    fig = px.scatter_mapbox(
        df_grouped,
        lat="Latitude",
        lon="Longitude",
        color=selected_color_feature,
        size="Growth (g per day)",
        size_max=25,
        hover_name="Plot Location",
        hover_data={
            "System": True,
            "Latitude": False,
            "Longitude": False,
            selected_color_feature: True,
            "Growth (g per day)": True,
            "Month": True
        },
        animation_frame="Monitoring Period",
        mapbox_style="carto-positron",
        color_continuous_scale=px.colors.sequential.Viridis,
        zoom=7,
        center={"lat": df['lat'].mean(), "lon": df['lon'].mean()}
    )

    # Update the layout of the figure
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        mapbox=dict(bearing=0, pitch=0),
        coloraxis_colorbar=dict(title=selected_color_feature),
        height=800
    )

    # Display the figure using Streamlit
    st.plotly_chart(fig, use_container_width=True)
    
def display_graphs(df, year_range, location_selection):
    """
    Displays interactive graphs showing various metrics over time.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        year_range (list): List of selected years.
        location_selection (list): List of selected locations.
    """
    st.header("📈 Graphs")  # Add a header for the graphs section
    
    # Define the metrics to be plotted, ordered by importance
    metrics = ['Growth (g per day)', 'Chlorophyll (mg per m3)', 'Sea Surface Temperature (C)', 
               'Turbidity (FTU)', 'Precipitation', 'Individual Weight (g)', 
               'Depth (m)', 'Average Flow Speed (mps)', 'Maximum Flow Speed (mps)', 'Living Mussel Count']

    # Create two columns for displaying the plots
    col1, col2 = st.columns(2)
    
    # Define colors for different systems
    system_colors = {'OS': 'orange', 'WAD': 'green'}
    
    # Create a colormap for location colors
    cmap = plt.get_cmap('tab10')
    location_colors = [mcolors.to_hex(cmap(i)) for i in np.linspace(0, 1, len(location_selection))]

    # Filter the dataframe based on the selected years
    df_filtered = df[df['Year'].isin(year_range)]

    # Initialize a counter to alternate between columns
    counter = 0

    # Loop through each metric to create a plot
    for metric in metrics:
        fig = go.Figure()  # Create a new figure for each metric
        added_systems = set()  # Track systems that have been added to the plot

        # If specific locations are selected, plot each location's data
        if location_selection:
            # Create a dictionary to map locations to colors
            location_colors_dict = {location: color for location, color in zip(location_selection, location_colors)}
            
            # Loop through each selected location
            for location in location_selection:
                if ' - ' in location:
                    # Split the location string into system and plot location
                    system, plot_location = location.split(' - ', 1)
                    
                    # Filter the data for the specific system and plot location
                    df_location = df_filtered[(df_filtered['System'] == system) & 
                                              (df_filtered['Plot Location'] == plot_location)].dropna()
                    
                    # Group by 'Month' and 'Monitoring Period', and calculate the mean for the metric
                    df_location_avg = df_location.groupby(['Month', 'Monitoring Period'])[metric].mean().reset_index()
                    
                    # Sort the grouped data by 'Monitoring Period'
                    df_location_avg.sort_values(by='Monitoring Period', inplace=True)

                    # Add a trace for the location's data
                    fig.add_trace(go.Scatter(x=df_location_avg['Month'], y=df_location_avg[metric], 
                                             mode='lines+markers', 
                                             name=f'Average at {plot_location}', 
                                             line=dict(shape='spline', color=location_colors_dict[location]),
                                             marker=dict(symbol='circle')))
                    
                    # Add system average line if not already added
                    if system not in added_systems:
                        df_system_avg = df_filtered[df_filtered['System'] == system].groupby(['Month', 'Monitoring Period'])[metric].mean().reset_index()
                        df_system_avg.sort_values(by='Monitoring Period', inplace=True)
                        fig.add_trace(go.Scatter(x=df_system_avg['Month'], y=df_system_avg[metric], 
                                                 mode='lines+markers', 
                                                 line=dict(dash='dash', shape='spline', color=system_colors[system]), 
                                                 name=f'Average in {system} System',
                                                 marker=dict(symbol='circle')))
                        added_systems.add(system)
                else:
                    st.warning(f"Invalid location format: {location}")
        else:
            # If no specific locations are selected, plot each system's average data
            for system in df['System'].unique():
                df_system_avg = df_filtered[df_filtered['System'] == system].groupby(['Month', 'Monitoring Period'])[metric].mean().reset_index()
                df_system_avg.sort_values(by='Monitoring Period', inplace=True)
                fig.add_trace(go.Scatter(x=df_system_avg['Month'], y=df_system_avg[metric], 
                                         mode='lines+markers', 
                                         name=f'Average {system}', 
                                         line=dict(shape='spline', color=system_colors[system]),
                                         marker=dict(symbol='circle')))

        # Update the layout of the figure
        fig.update_layout(template="plotly_dark", title=f"{metric} Trends - {', '.join(map(str, year_range))}")
        
        # Alternate between the two columns for displaying the charts
        with (col1 if counter % 2 == 0 else col2):
            st.plotly_chart(fig, use_container_width=True)
        
        counter += 1  # Increment the counter to alternate columns
            
def display_feature_vs_target_analysis(df, year_range, location_selection):
    """
    Displays an interactive graph comparing a selected feature against Growth (g per day).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        year_range (list): List of selected years.
        location_selection (list): List of selected locations.
    """
    st.header("🔍 Growth (g per day) Feature Analysis")  # Header for the analysis section
    
    # Define the available metrics
    metrics = ['Growth (g per day)', 'Chlorophyll (mg per m3)', 'Sea Surface Temperature (C)', 
               'Turbidity (FTU)', 'Precipitation', 'Individual Weight (g)', 
               'Depth (m)', 'Average Flow Speed (mps)', 'Maximum Flow Speed (mps)', 'Living Mussel Count']
    
    # Create a list of feature options excluding 'Growth (g per day)'
    feature_options = [metric for metric in metrics if metric != 'Growth (g per day)']
    
    # Selectbox for the user to choose a feature to plot against 'Growth (g per day)'
    selected_feature = st.selectbox('Select a feature to plot against Growth (g per day)', options=feature_options)

    if selected_feature:
        # Filter the dataframe based on the selected years
        df_filtered = df[df['Year'].isin(year_range)]
        
        # If specific locations are selected, filter based on those locations
        if location_selection:
            df_filtered['Location_System'] = df_filtered['System'] + ' - ' + df_filtered['Plot Location']
            df_filtered = df_filtered[df_filtered['Location_System'].isin(location_selection)]
        
        # Create a new figure for plotting
        fig = go.Figure()

        if location_selection:
            # Loop through each selected location
            for location in location_selection:
                if ' - ' in location:
                    # Split the location string into system and plot location
                    system, plot_location = location.split(' - ', 1)
                    
                    # Filter the data for the specific system and plot location
                    df_location = df_filtered[(df_filtered['System'] == system) & 
                                              (df_filtered['Plot Location'] == plot_location)].dropna(subset=[selected_feature, 'Growth (g per day)'])
                    
                    # Group by 'Month' and 'Monitoring Period', and calculate the mean for the selected feature and target
                    df_grouped = df_location.groupby(['Month', 'Monitoring Period'])[[selected_feature, 'Growth (g per day)']].mean().reset_index()
                    
                    # Sort the grouped data by 'Monitoring Period'
                    df_grouped.sort_values(by='Monitoring Period', inplace=True)

                    # Add a trace for the selected feature
                    fig.add_trace(go.Scatter(x=df_grouped['Month'], y=df_grouped[selected_feature], mode='lines+markers', 
                                             name=f'{selected_feature} at {plot_location}', yaxis='y1', line_shape='spline'))
                    
                    # Add a trace for 'Growth (g per day)'
                    fig.add_trace(go.Scatter(x=df_grouped['Month'], y=df_grouped['Growth (g per day)'], mode='lines+markers', 
                                             name=f'Growth (g per day) at {plot_location}', line=dict(width=4, dash='dash'), yaxis='y2', line_shape='spline'))
        else:
            # If no specific locations are selected, plot the data for each system
            for system in df['System'].unique():
                df_system = df_filtered[df_filtered['System'] == system].dropna(subset=[selected_feature, 'Growth (g per day)'])
                
                # Group by 'Month' and 'Monitoring Period', and calculate the mean for the selected feature and target
                df_grouped = df_system.groupby(['Month', 'Monitoring Period'])[[selected_feature, 'Growth (g per day)']].mean().reset_index()
                
                # Sort the grouped data by 'Monitoring Period'
                df_grouped.sort_values(by='Monitoring Period', inplace=True)

                # Add a trace for the selected feature
                fig.add_trace(go.Scatter(x=df_grouped['Month'], y=df_grouped[selected_feature], mode='lines+markers', 
                                         name=f'{selected_feature} in {system} System', yaxis='y1', line_shape='spline'))
                
                # Add a trace for 'Growth (g per day)'
                fig.add_trace(go.Scatter(x=df_grouped['Month'], y=df_grouped['Growth (g per day)'], mode='lines+markers', 
                                         name=f'Growth (g per day) in {system} System', line=dict(width=4, dash='dash'), yaxis='y2', line_shape='spline'))

        # Update the layout of the figure
        fig.update_layout(
            title=f"{selected_feature} vs Growth (g per day)",
            xaxis_title='Month',
            yaxis=dict(title=f'{selected_feature} Values', side='left'),
            yaxis2=dict(title='Growth (g per day)', overlaying='y', side='right'),
            height=600,
            template="plotly_dark"
        )

        # Display the figure
        st.plotly_chart(fig, use_container_width=True)
        
def display_location_rankings(df):
    """
    Displays a ranking table of locations based on the average of selected metrics over all years.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    """
    # Select the metrics to rank locations
    metrics = ['Growth (g per day)', 'Precipitation', 'Sea Surface Temperature (C)', 'Chlorophyll (mg per m3)', 'Turbidity (FTU)']
    
    # Calculate the average of each metric for each location and system
    df_rankings = df.groupby(['Plot Location', 'System'])[metrics].mean().round(2).reset_index()
    
    # Sort by 'Growth (g per day)' in descending order
    df_rankings = df_rankings.sort_values(by='Growth (g per day)', ascending=False).reset_index(drop=True)

    # Display the rankings table with a header
    st.header("📊 Location Rankings Based on Average Metrics")
    
    # Add a custom information box
    st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-size: 16px; margin-bottom: 20px;">
            <strong>Note</strong>: The table below shows the ranking of locations based on the average of selected metrics over all years.
        </div>
    """, unsafe_allow_html=True)
    
    # Display the dataframe as a table
    st.dataframe(df_rankings, use_container_width=True)

def haversine(latitude1, longitude1, latitude2, longitude2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface given their latitude and longitude.
    
    Parameters:
    - latitude1: Latitude of the first point in decimal degrees.
    - longitude1: Longitude of the first point in decimal degrees.
    - latitude2: Latitude of the second point in decimal degrees.
    - longitude2: Longitude of the second point in decimal degrees.
    
    Returns:
    - distance: The distance between the two points in kilometers.
    """
    # Radius of the Earth in kilometers
    earth_radius_km = 6371.0

    # Convert latitude and longitude from degrees to radians
    delta_latitude = radians(latitude2 - latitude1)
    delta_longitude = radians(longitude2 - longitude1)
    
    # Apply the Haversine formula to calculate the great-circle distance
    a = sin(delta_latitude / 2)**2 + cos(radians(latitude1)) * cos(radians(latitude2)) * sin(delta_longitude / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Calculate the distance
    distance = earth_radius_km * c

    return distance

def get_nearest_environmental_data(latitude, longitude, dataframe, month):
    """
    Find the nearest environmental data point to a given latitude and longitude for a specific month.

    Parameters:
    - latitude: Latitude of the target location in decimal degrees.
    - longitude: Longitude of the target location in decimal degrees.
    - dataframe: DataFrame containing the environmental data with 'lat' and 'lon' columns.
    - month: The month for which to filter the data.

    Returns:
    - nearest_row: The row from the DataFrame that is closest to the given location for the specified month.
    """
    global df_environment  # Add this line
    
    # Filter the dataframe for the specified month
    monthly_data = dataframe[dataframe['month'] == month].copy()

    # Calculate the distance from the target location for each row in the filtered dataframe
    monthly_data['distance'] = monthly_data.apply(
        lambda row: haversine(latitude, longitude, row['lat'], row['lon']), axis=1
    )

    # Find the row with the minimum distance
    nearest_row = monthly_data.loc[monthly_data['distance'].idxmin()]
    
    return nearest_row

def calculate_average_ash_free_dry_weight(monitoring_period):
    """
    Calculate the average Ash Free Dry Weight (g) for the specified monitoring period.

    Parameters:
    - monitoring_period: The monitoring period for which to calculate the average weight.

    Returns:
    - average_weight: The average Ash Free Dry Weight (g) for the specified period.
    """
    # Filter the dataframe for the specified monitoring period
    period_data = df_modeling[df_modeling['Monitoring Period'] == monitoring_period]
    
    # Calculate the mean of 'Ash Free Dry Weight (g)_lag' for the filtered data
    average_weight = period_data['Ash Free Dry Weight (g)_lag'].mean()
    
    return average_weight

def predict_growth(latitude, longitude, initial_weight):
    """
    Predict mussel growth based on user inputs and additional environmental features.

    Parameters:
    - latitude: Latitude of the location.
    - longitude: Longitude of the location.
    - initial_weight: Initial weight of the mussel.

    Returns:
    - predictions: DataFrame with columns 'Month' and 'Growth (g per day)'.
    """
    # Get nearest environmental data for April (lag) and May (current)
    env_data_april = get_nearest_environmental_data(latitude, longitude, df_environment, 4)
    env_data_may = get_nearest_environmental_data(latitude, longitude, df_environment, 5)

    # Create input data dictionary for prediction
    input_data = {
        'Chlorophyll': [env_data_may.get('chlorophyll', 0)],
        'Water Temperature (C)': [env_data_may.get('sst', 0)],
        'Water Temperature (C)_lag': [env_data_april.get('sst', 0)],
        'Ash Free Dry Weight (g)_lag': [calculate_average_ash_free_dry_weight(0)],
        'Number of Days': [30],
        'Precipitation_lag': [env_data_april.get('precipitation', 0)],
        'Turbidity (FTU)': [env_data_may.get('turbidity', 0)],
        'Turbidity (FTU)_lag': [env_data_april.get('turbidity', 0)],
        'Monitoring Period': [0],
        'Individual Weight (g)_lag': [initial_weight],
        'Chlorophyll_lag': [env_data_april.get('chlorophyll', 0)]
    }

    # Convert input data dictionary to DataFrame
    input_df = pd.DataFrame(input_data)
    predictions = []

    # Perform predictions for each monitoring period from May to October
    for period in range(1, 7):
        # Predict growth using the loaded model
        prediction = loaded_model.predict(input_df[[
            'Chlorophyll', 'Water Temperature (C)', 'Water Temperature (C)_lag', 
            'Ash Free Dry Weight (g)_lag', 'Number of Days', 'Precipitation_lag', 
            'Turbidity (FTU)', 'Turbidity (FTU)_lag', 'Monitoring Period', 
            'Individual Weight (g)_lag', 'Chlorophyll_lag']])[0]
        predictions.append(prediction)
        
        # Update input data for the next period
        input_df['Monitoring Period'] = period
        input_df['Growth (g per day)'] = prediction
        input_df['Ash Free Dry Weight (g)_lag'] = calculate_average_ash_free_dry_weight(period)

    # Create a DataFrame for predictions with 'Month' and 'Growth (g per day)'
    months = ['May', 'June', 'July', 'August', 'September', 'October']
    prediction_df = pd.DataFrame({
        'Month': months,
        'Growth (g per day)': predictions
    })

    return prediction_df

def display_prediction_interface():
    """
    Displays the prediction interface for mussel growth. Allows users to input latitude, longitude, 
    and individual weight, then predicts growth based on these inputs.
    """
    st.header("🔮 Predict Mussel Growth")

    # Custom warning box for model disclaimer
    st.markdown("""
        <div style="background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; font-size: 16px; margin-bottom: 20px;">
            <strong>Disclaimer</strong>: The predictions provided here are based on a trained model. 
            While the model has been developed and tested with historical data, 
            the predictions are not guaranteed to be 100% accurate. 
            Please use these predictions as a guideline and not as a definitive forecast.
        </div>
    """, unsafe_allow_html=True)

    # Initial coordinates for Amsterdam
    initial_lat = 52.3676
    initial_lon = 4.9041

    # User inputs for prediction
    lat = st.number_input('Latitude', value=initial_lat, format="%.6f")
    lon = st.number_input('Longitude', value=initial_lon, format="%.6f")
    individual_weight = st.number_input('Individual Weight (g)', value=2.0, min_value=0.1)

    # Button click state initialization
    if 'predict_clicked' not in st.session_state:
        st.session_state['predict_clicked'] = False

    button_placeholder = st.empty()  # Placeholder for the predict button

    # Display the predict button if prediction has not been clicked yet
    if not st.session_state['predict_clicked']:
        if button_placeholder.button("Predict Growth", key="predict"):
            st.session_state['predict_clicked'] = True  # Update state to indicate button was clicked
            with st.spinner('Predicting...'):
                # Perform prediction
                predictions_df = predict_growth(lat, lon, individual_weight)
                st.session_state['predictions'] = predictions_df  # Store predictions in session state

    # Display predictions and plot if available
    if st.session_state['predict_clicked'] and 'predictions' in st.session_state:
        st.write(st.session_state['predictions'])  # Display predictions DataFrame

        # Create a plot of the predictions
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state['predictions']['Month'], 
            y=st.session_state['predictions']['Growth (g per day)'], 
            mode='lines+markers', 
            line_shape='spline'
        ))
        fig.update_layout(
            title="Predicted Growth (g per day) for Each Month", 
            xaxis_title="Month", 
            yaxis_title="Growth (g per day)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Reset the state for the next prediction
        st.session_state['predict_clicked'] = False
        st.session_state.pop('predictions', None)
        button_placeholder.button("Predict Growth", key="predict_new")
        
def display_data(df):
    """
    Main function to display the interactive map, graphs, and prediction interface in the Streamlit app.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
    """

    # Create the sidebar and get the user's selections
    year_range, location_selection = create_sidebar(df)

    # Apply custom styling using CSS for font family
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter&display=swap');
    h1, h2, h3, h4, h5, h6, p {
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    # Markdown block to provide an overview and instructions for the app
    st.markdown("""
    > This dashboard visualizes mussel growth data from various locations in the Netherlands. 
    > Use the sidebar to filter data by year and location.
    > The map shows mussel baskets with different colors representing different systems. 
    > Charts below display detailed growth trends. Hover over them for more information.
    """)

    # Display header for the map section
    st.header("🌍 Map")

    # Display the main interactive map
    display_main_map(df, year_range, location_selection)
    
    # Add a divider for better visual separation
    st.divider()

    # Display graphs for various metrics over time
    display_graphs(df, year_range, location_selection)

    # Add another divider
    st.divider()

    # Display analysis comparing selected features against Growth (g per day)
    display_feature_vs_target_analysis(df, year_range, location_selection)
    
    # Add another divider
    st.divider()

    # Display location rankings
    display_location_rankings(df)
    
    # Add a final divider
    st.divider()
    
    # Display the prediction interface for predicting mussel growth
    display_prediction_interface()

def main():
    """
    Main function to run the Streamlit app for Mussel Growth Trends Dashboard.
    """
    # Set the title of the Streamlit app
    st.title('🦪 Mussel Growth Trends Dashboard'.upper())

    # Check if 'access_granted' exists in the session state, if not, initialize it as False
    if "access_granted" not in st.session_state:
        st.session_state.access_granted = False

    # Access control: If access has not been granted
    if not st.session_state.access_granted:
        # Create three columns for layout, with the middle column for the access code input
        cols = st.columns([1, 2, 1])
        with cols[1]:
            # Create a password input field in the middle column for the access code
            access_code = st.text_input("Enter access code", type="password")
            # Create a submit button in the middle column
            if st.button("Submit"):
                # Verify the access code
                if verify_access_code(access_code):
                    # If the access code is correct, set 'access_granted' to True and refresh the page
                    st.session_state.access_granted = True
                    st.experimental_rerun()()  # Refresh the page to reload with granted access
                else:
                    # If the access code is incorrect, display an error message
                    st.error("Access denied. Please enter the correct access code.")
            else:
                # If the submit button is not clicked, display a message prompting for the access code
                st.markdown("""
                > This dashboard allows you to explore the growth trends of mussels across various locations and periods. 
                > Please enter the access code to continue.
                """)
    else:
        # If access is granted, proceed to load necessary files
        global loaded_model, df_environment, df_modeling  # Define as global variables

        # Load the trained model
        loaded_model = load_trained_model('./Data/final_ridge_model.pkl')
        # Load the combined data
        df_environment = load_data('./Data/combined_data.csv')
        
        # Load the main data
        df_modeling = load_data('./Data/It3 - Mussel + SatML + Weather (Lag Features).csv')

        # Drop 'Plot Location' column from the main DataFrame as it is not needed
        df_modeling.drop(columns=['Plot Location'], inplace=True)

        # Load the main dataset and ensure the 'Year' column is of type int
        df = load_data('./Data/Mussel + Satellite + Weather.csv')
        df['Year'] = df['Year'].astype(int)

        # Display the data and visualizations in the app
        display_data(df)

if __name__ == '__main__':
    main()
