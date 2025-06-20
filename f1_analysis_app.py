import streamlit as st
import pandas as pd
import fastf1 as ff1
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
st.set_page_config(page_title="F1 Analysis Dashboard", layout="wide")

# Set up cache directory
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')
ff1.Cache.enable_cache('f1_cache')

# Helper functions
@st.cache_data
def load_session_data(year, event, session):
    session = ff1.get_session(year, event, session)
    session.load()
    return session

@st.cache_data
def load_ergast_data():
    url = "http://ergast.com/api/f1/2023/results.json?limit=1000"
    response = requests.get(url)
    data = response.json()
    races = pd.json_normalize(data['MRData']['RaceTable']['Races'])
    return races

# Sidebar controls
st.sidebar.header("F1 Data Selection")
selected_year = st.sidebar.selectbox("Select Year", [2023, 2022, 2021])
selected_event = st.sidebar.selectbox("Select Event", ['Monaco', 'Bahrain', 'Spain'])
selected_session = st.sidebar.selectbox("Select Session", ['Q', 'R', 'FP1', 'FP2'])

# Main app
st.title("Formula 1 Analysis Dashboard")

# Data Loading Section
st.header("Session Data Loading")
try:
    session = load_session_data(selected_year, selected_event, selected_session)
    st.success("Session data loaded successfully!")
except Exception as e:
    st.error(f"Error loading session data: {str(e)}")

if 'session' in locals():
    # Laps Data
    st.subheader("Laps Data")
    laps = session.laps
    st.dataframe(laps.head(), height=200)
    
    # Telemetry Data
    st.subheader("Driver Telemetry")
    selected_driver = st.selectbox("Select Driver", laps['Driver'].unique())
    driver_lap = laps.pick_driver(selected_driver).pick_fastest()
    telemetry = driver_lap.get_telemetry()
    st.dataframe(telemetry[['Speed', 'Throttle', 'Brake', 'nGear']].head())
    
    # Weather Data
    st.subheader("Weather Data")
    st.dataframe(session.weather_data)

# Race Results Section
st.header("Historical Race Results")
if st.button("Load Ergast API Data"):
    races = load_ergast_data()
    st.dataframe(races[['raceName', 'date', 'Circuit.circuitName']])
    
    # Winners Data
    st.subheader("2023 Race Winners")
    winners = pd.json_normalize(
        races,
        record_path=['Results'],
        meta=['season', 'round', 'url', 'raceName', 'Circuit.circuitName']
    )
    st.dataframe(winners[['number', 'Driver.givenName', 'Driver.familyName', 'position']])

# Web Scraping Section
st.header("Wikipedia Race Data Scraping")
if st.button("Scrape Wikipedia Data"):
    try:
        url = "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='wikitable')
        df = pd.read_html(str(table))[0]
        st.dataframe(df.head())
        df.to_csv('f1_2023_results.csv', index=False)
        st.success("Data scraped and saved successfully!")
    except Exception as e:
        st.error(f"Scraping error: {str(e)}")

# Analysis Section
st.header("Data Analysis")
if st.checkbox("Show Merged Dataset Analysis"):
    try:
        merged_data = pd.read_csv('f1_merged_data.csv')
        
        # Weather Impact Analysis
        st.subheader("Weather Impact on Performance")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            x='Weather_Category',
            y='positionOrder',
            data=merged_data,
            order=['Dry', 'Damp', 'Wet', 'Torrential', 'Unknown'],
            ax=ax
        )
        plt.title('Race Performance by Weather Condition')
        plt.ylabel('Finishing Position (lower = better)')
        plt.xlabel('Weather Conditions')
        st.pyplot(fig)
        
        # Team Comparison
        st.subheader("Team Performance")
        selected_team = st.selectbox("Select Team", merged_data['Team'].unique())
        team_data = merged_data[merged_data['Team'] == selected_team]
        st.write(f"### {selected_team} Results")
        st.dataframe(team_data[['raceName', 'positionOrder', 'points']])
        
    except Exception as e:
        st.warning(f"Couldn't load merged data: {str(e)}")

# Modeling Section
st.header("Predictive Modeling")
if st.checkbox("Show Modeling Options"):
    st.write("""
    ### Model Configuration
    (Placeholder for modeling interface)
    """)
    
    target = st.selectbox("Select Target Variable", ['positionOrder', 'points', 'win'])
    features = st.multiselect("Select Features", ['grid', 'pit_stop_count', 'temperature', 'rainfall'])
    
    if st.button("Run Model"):
        st.warning("Modeling functionality not implemented in this demo")

# Footer
st.markdown("---")
st.markdown("F1 Analysis Dashboard - Created using Streamlit")