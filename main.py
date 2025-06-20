#!/usr/bin/env python
# coding: utf-8

# In[2]:


import fastf1 as ff1
import os
# Create cache directory if it doesn't exist
if not os.path.exists('f1_cache'):
    os.makedirs('f1_cache')

ff1.Cache.enable_cache('f1_cache')  # Stores data in a folder


# In[4]:


session = ff1.get_session(2023, 'Monaco', 'Q')  # 2023 Monaco GP Qualifying
session.load()  # Load telemetry, weather, and lap data


# Lap Times

# In[5]:


laps = session.laps
print(laps.head())  # Shows fastest laps, drivers, compounds


# Telemetry (Speed, Throttle, Gear, etc.)

# In[6]:


driver_lap = laps.pick_driver('VER').pick_fastest()  # Max Verstappen's fastest lap
telemetry = driver_lap.get_telemetry()
print(telemetry[['Speed', 'Throttle', 'Brake', 'nGear']].head())


# Weather

# In[7]:


print(session.weather_data)  # Air temp, humidity, rainfall


# Make http request

# In[8]:


import requests
import pandas as pd

url = "http://ergast.com/api/f1/2023/results.json?limit=1000"
response = requests.get(url)
data = response.json()

# Convert to DataFrame
races = pd.json_normalize(data['MRData']['RaceTable']['Races'])
print(races[['raceName', 'date', 'Circuit.circuitName']])


# All 2023 Race Winners

# In[9]:


winners = pd.json_normalize(
    data['MRData']['RaceTable']['Races'],
    record_path=['Results']
)
print(winners[['number', 'Driver.givenName', 'Driver.familyName', 'position']])


# In[13]:


import pandas as pd

races = pd.read_csv('Formula 1 World Championship (1950 - 2024)/races.csv')  
results = pd.read_csv('Formula 1 World Championship (1950 - 2024)/results.csv')  
drivers = pd.read_csv('Formula 1 World Championship (1950 - 2024)/drivers.csv')  


# Example: Get 2023 Race Winners

# In[14]:


winners_2023 = results[results['positionOrder'] == 1].merge(
    races[races['year'] == 2023],
    on='raceId'
)
print(winners_2023[['name', 'date', 'driverId']])


# In[15]:


# pip install requests beautifulsoup4 pandas


# Scrape Race Results

# In[2]:


# Install lxml (run this cell first if not already installed)
# get_ipython().system('pip install lxml')

# Import required libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import io

# Fetch the Wikipedia page
url = "https://en.wikipedia.org/wiki/2023_Formula_One_World_Championship"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the results table (adjust the selector or index if needed)
# Note: Inspect the page to confirm the correct table (e.g., class 'wikitable')
table = soup.find('table', class_='wikitable')  # Target tables with class 'wikitable'

if table:
    # Convert the table to a string and wrap in StringIO
    html_table = str(table)
    df = pd.read_html(io.StringIO(html_table))[0]  # Convert to DataFrame, take first table
else:
    print("No table found with the specified selector. Please adjust the 'find' criteria.")

# Display the DataFrame
print(df.head())


# Save to CSV

# In[9]:


df.to_csv('f1_2023_results.csv', index=False)


# Run Data Sanity Checks

# In[12]:


print(df.isnull().sum())  # Check missing values
print(df.describe())      # Stats for numerical fields


# Optimize Merges

# In[19]:


# More efficient pit stop aggregation
# Load pit stops data first
pit_stops = pd.read_csv('Formula 1 World Championship (1950 - 2024)/pit_stops.csv')

# More efficient pit stop aggregation
pit_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
    pit_stop_count=('stop', 'count'),
    total_pit_time=('milliseconds', 'sum')
).reset_index()
results = results.merge(pit_agg, on=['raceId', 'driverId'], how='left')


# In[18]:


import pandas as pd

# Check if results is already defined, if not load it
if 'results' not in globals() or 'pit_stop_count' not in results.columns:
    # Need to reload data and recreate the merge
    # Import pandas if not already imported
    results = pd.read_csv('Formula 1 World Championship (1950 - 2024)/results.csv')

    # Reload pit stops data if needed
    if 'pit_stops' not in globals():
        pit_stops = pd.read_csv('Formula 1 World Championship (1950 - 2024)/pit_stops.csv')

    # Recreate the pit stop aggregation
    pit_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
        pit_stop_count=('stop', 'count'),
        total_pit_time=('milliseconds', 'sum')
    ).reset_index()

    # Merge pit stop data with results
    results = results.merge(pit_agg, on=['raceId', 'driverId'], how='left')

# Save the processed results    
results.to_csv('f1_processed_data.csv', index=False)
print(f"Data saved successfully with {results.shape[0]} rows and {results.shape[1]} columns")


# ## Step 2: Data Preprocessing & Feature Engineering

# In[17]:


import numpy as np

# =============================================
# STEP 2: DATA PREPROCESSING & FEATURE ENGINEERING
# =============================================

# Load preprocessed data from Step 1
data = pd.read_csv('F1-Weather/F1 Weather(2023-2018).csv')

# Convert date fields (if not already done)
# Create a date column using Year and Round Number
# Assuming each round corresponds roughly to a month (starting March for round 1)
data['date'] = pd.to_datetime(data['Year'].astype(str) + '-' + 
                             ((data['Round Number'] + 2) % 12 + 1).astype(str) + '-1')

# If Time contains timestamp information, convert it properly
if 'Time' in data.columns and data['Time'].dtype == 'object':
    data['Time'] = pd.to_timedelta(data['Time'])

# -------------------------------------------------------------------
# 1. HANDLE MISSING DATA
# -------------------------------------------------------------------
# Check which columns exist in the dataset
expected_columns = ['pit_stop_count', 'total_pit_time_ms', 'temperature', 'grid', 'positionOrder', 
                    'circuitId', 'driverId', 'constructorId']
missing_columns = [col for col in expected_columns if col not in data.columns]

if missing_columns:
    print(f"Missing columns: {missing_columns}")
    # Add missing columns with default values
    for col in missing_columns:
        if col in ['pit_stop_count', 'total_pit_time_ms', 'rainfall']:
            data[col] = 0
        elif col == 'temperature':
            data[col] = data['AirTemp']  # Use AirTemp instead
        elif col == 'rainfall':
            data[col] = data['Rainfall'].astype(int)
        elif col == 'grid':
            data[col] = 20  # Default to back of grid
        else:
            data[col] = None

# Now process each column safely
# Weather data
data['temperature'] = data['temperature'].fillna(data['temperature'].median())
# Check if 'rainfall' column exists, if not create it from 'Rainfall'
if 'rainfall' not in data.columns and 'Rainfall' in data.columns:
    data['rainfall'] = data['Rainfall'].astype(int)
# Now fill any NA values
data['rainfall'] = data['rainfall'].fillna(0)

# Pit stop data (if available)
if 'pit_stop_count' in data.columns:
    data['pit_stop_count'] = data['pit_stop_count'].fillna(0)
    data['total_pit_time_ms'] = data['total_pit_time_ms'].fillna(0)

# Qualifying position
if 'grid' in data.columns:
    data['grid'] = data['grid'].fillna(20)  # Assume back of grid if no qualifying

# Rest of the feature engineering code only if required columns exist
if 'positionOrder' in data.columns:
    # Target variable: Win (1 if winner, 0 otherwise)
    data['win'] = (data['positionOrder'] == 1).astype(int)

    # Pit stop efficiency (ms per stop)
    if 'pit_stop_count' in data.columns and 'total_pit_time_ms' in data.columns:
        data['avg_pit_time_ms'] = np.where(
            data['pit_stop_count'] > 0,
            data['total_pit_time_ms'] / data['pit_stop_count'],
            0  # For drivers with no pit stops
        )

# Weather impact features
if 'temperature' in data.columns:
    data['is_wet'] = (data['rainfall'] > 0).astype(int)
    data['temp_category'] = pd.cut(data['temperature'],
                                  bins=[-np.inf, 15, 25, np.inf],
                                  labels=['cold', 'moderate', 'hot'])

# Continue with other feature engineering that matches your data schema
print("\n=== Missing Values Check ===")
print(data.isnull().sum())


# In[16]:


# Check for missing values
print("Missing Values:\n", data.isnull().sum())

# Verify feature ranges
print("\nFeature Ranges:")
print("- Win rate:", data['win'].mean())  # Should be ~1/20 drivers win
print("- Avg pit stops:", data['pit_stop_count'].mean())  # Typically 1-3
print("- Wet races:", data['is_wet'].sum())  # Count of wet races


print("\nSample Engineered Features:")
# Show existing columns that have data
available_cols = ['win', 'grid', 'is_wet', 'temperature', 'avg_pit_time_ms']
print(data[available_cols].sample(5))

# Show data types and non-null counts for key columns
print("\nColumn Info:")
for col in ['driverId', 'constructorId', 'positionOrder']:
    print(f"- {col}: {data[col].dtype}, {data[col].count()} non-null values")


# Merging weather in merged
# 

# In[83]:


import pandas as pd

# Load your merged dataset
merged_data = pd.read_csv('f1_merged_data.csv')

# Check available weather-related columns
print("Columns containing weather data:")
print([col for col in merged_data.columns if 'weather' in col.lower() or 'temp' in col.lower() or 'rain' in col.lower()])


# Standardize column names (adjust based on your actual columns)
weather_cols = {
    'Rainfall': 'rainfall',
    'Precipitation': 'rainfall',
    'Temp': 'temperature',
    'AirTemp': 'temperature',
    'IsRain': 'is_wet',
    'WetRace': 'is_wet'
}

# Rename columns if they exist in your data
for old_name, new_name in weather_cols.items():
    if old_name in merged_data.columns:
        merged_data.rename(columns={old_name: new_name}, inplace=True)

# Create missing columns with default values if needed
if 'rainfall' not in merged_data:
    merged_data['rainfall'] = 0  # Default to dry conditions

if 'temperature' not in merged_data:
    merged_data['temperature'] = 20  # Default temperate (20°C)

if 'is_wet' not in merged_data:
    # Create binary flag based on rainfall
    merged_data['is_wet'] = (merged_data['rainfall'] > 0).astype(int)

    # Create composite weather score
merged_data['Weather_Impact'] = (
    merged_data['rainfall'] * 0.7 +  # Weight rainfall more heavily
    merged_data['temperature'] * 0.3 * merged_data['is_wet']  # Temp only matters when wet
)

# Bin into categories
weather_bins = [-1, 0, 2, 5, 10]  # Adjust thresholds based on your data distribution
merged_data['Weather_Category'] = pd.cut(
    merged_data['Weather_Impact'],
    bins=weather_bins,
    labels=['Dry', 'Damp', 'Wet', 'Torrential'],
    right=False
)

# Fill any NA values (if bins don't cover all cases)
merged_data['Weather_Category'] = merged_data['Weather_Category'].cat.add_categories(['Unknown']).fillna('Unknown')


# Check distribution
print(merged_data['Weather_Category'].value_counts())

# Visualize impact on performance
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.boxplot(
    x='Weather_Category',
    y='positionOrder',
    data=merged_data,
    order=['Dry', 'Damp', 'Wet', 'Torrential', 'Unknown']
)
plt.title('Race Performance by Weather Condition')
plt.ylabel('Finishing Position (lower = better)')
plt.xlabel('Weather Conditions')
plt.show()

# Save back to CSV if needed
merged_data.to_csv('f1_merged_data_with_weather.csv', index=False)
merged_data.to_csv('f1_merged_data.csv', index=False)


# Modeling: Use win as target for classification, positionOrder for regression.

# In[6]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Create a boxplot comparing a valid column with the win status
# Ensure data is loaded
if 'data' not in globals():
    # Reload data from cell 26
    data = pd.read_csv('F1-Weather/F1 Weather(2023-2018).csv')

    # Minimal preprocessing needed for this visualization
    if 'positionOrder' in data.columns:
        data['win'] = (data['positionOrder'] == 1).astype(int)

# Check if win column exists, if not we need to create it or use another approach
if 'win' not in data.columns:
    # Since we don't have positionOrder in the data to determine winners
    # Let's create a different visualization using available columns
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Rainfall', y='AirTemp', data=data)
    plt.title('Air Temperature by Weather Condition')
    plt.ylabel('Air Temperature (°C)')
    plt.xlabel('Rainfall (True/False)')
else:
    # Original visualization if win column exists
    sns.boxplot(x='win', y='grid', data=data)
plt.title('Starting Position by Race Outcome')
plt.ylabel('Grid Position')
plt.xlabel('Win (1=Yes, 0=No)')
# Only execute this section if the data is properly prepared
# This section needs both 'win' and 'driver_form' columns which aren't in the dataset
# Instead, visualize the relationship between existing columns
plt.figure(figsize=(10, 6))
sns.boxplot(x='Rainfall', y='AirTemp', data=data)
plt.title('Air Temperature Distribution by Rainfall Condition')
plt.ylabel('Air Temperature (°C)')
plt.xlabel('Rainfall Present')


# To verify the 2nd step

# In[15]:


import pandas as pd
# Load the existing weather data that we've been using
data = pd.read_csv('F1-Weather/F1 Weather(2023-2018).csv')

# Save it as the expected file for future use
data.to_csv('f1_engineered_features.csv', index=False)

# Check structure
print(f"Shape: {data.shape} (rows, cols)")
print("\nMissing values:\n", data.isnull().sum())

# Verify key ranges
print("\nValue Ranges:")
# Check if win column exists before accessing it
if 'win' in data.columns:
    print("- Wins:", data['win'].value_counts())  # Should be ~5% wins (1) per race
else:
    print("- No 'win' column found in dataset")
# Check if grid column exists before accessing it
if 'grid' in data.columns:
    print("- Grid positions:", data['grid'].describe())  # Usually 1-20
else:
    print("- No 'grid' column found in dataset")
# Check if pit_stop_count column exists before accessing it
if 'pit_stop_count' in data.columns:
    print("- Pit stops:", data['pit_stop_count'].max())  # Rarely >4
else:
    print("- No 'pit_stop_count' column found in dataset")


# In[14]:


import pandas as pd

# Set the correct directory path
data_dir = 'Formula 1 World Championship (1950 - 2024)'

# Load ALL required files with correct paths
races = pd.read_csv(f'{data_dir}/races.csv')
results = pd.read_csv(f'{data_dir}/results.csv')
pit_stops = pd.read_csv(f'{data_dir}/pit_stops.csv')
drivers = pd.read_csv(f'{data_dir}/drivers.csv')
constructors = pd.read_csv(f'{data_dir}/constructors.csv')

# Merge core data (results + races)
data = pd.merge(
    results,
    races[['raceId', 'year', 'circuitId', 'name', 'date']],
    on='raceId',
    how='left'
)

# Merge pit stops (aggregated per driver/race)
pit_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
    pit_stop_count=('stop', 'count'),
    total_pit_time=('milliseconds', 'sum')
).reset_index()
data = pd.merge(data, pit_agg, on=['raceId', 'driverId'], how='left')

# Merge driver & constructor names
data = pd.merge(data, drivers[['driverId', 'driverRef']], on='driverId')
data = pd.merge(data, constructors[['constructorId', 'name']], on='constructorId')

# Save intermediate file
data.to_csv('f1_merged_data.csv', index=False)


# In[16]:


# Load merged data
data = pd.read_csv('f1_merged_data.csv')

# Add target variable
data['win'] = (data['positionOrder'] == 1).astype(int)

# Handle missing pit stops
data['pit_stop_count'] = data['pit_stop_count'].fillna(0)

# Add position gain
data['position_gain'] = data['grid'] - data['positionOrder']

# Save final output
data.to_csv('f1_engineered_features.csv', index=False)


# In[17]:


print(data[['year', 'name', 'driverRef', 'grid', 'positionOrder', 'win', 'pit_stop_count']].head(3))


# In[25]:


# Select and rename columns for clarity
output_data = data[[
    'year', 
    'name_x',          # Race name (from races.csv)
    'driverRef',       # Driver identifier
    'grid',            # Starting position
    'positionOrder',   # Finishing position
    'win',             # 1 if winner, 0 otherwise
    'pit_stop_count',  # Number of pit stops
    'position_gain'    # Positions gained (grid - positionOrder)
]].rename(columns={'name_x': 'raceName'})

# Display the first 3 rows
print(output_data.head(3))


data = data.rename(columns={'name_y': 'constructorName'})

# Rename both columns for consistency
data = data.rename(columns={'name': 'raceName', 'name_y': 'constructorName'})
print(data[['raceName', 'driverRef', 'constructorName']].sample(5))

# Merge weather data if available
if 'temperature' not in data.columns:
    weather = pd.read_csv('weather.csv')
    data = pd.merge(data, weather, on='raceId', how='left')


# ## Checking

# In[64]:


import pandas as pd
import numpy as np

# Set the correct directory path
data_dir = 'Formula 1 World Championship (1950 - 2024)'

# 1. LOAD ALL REQUIRED FILES
try:
    races = pd.read_csv(f'{data_dir}/races.csv')
    results = pd.read_csv(f'{data_dir}/results.csv')
    pit_stops = pd.read_csv(f'{data_dir}/pit_stops.csv')
    drivers = pd.read_csv(f'{data_dir}/drivers.csv')
    constructors = pd.read_csv(f'{data_dir}/constructors.csv')

    # Try to load weather data if available
    try:
        weather = pd.read_csv(f'{data_dir}/weather.csv')
        has_weather = True
    except FileNotFoundError:
        print("Weather data not found - continuing without it")
        has_weather = False

except FileNotFoundError as e:
    print(f"Missing file: {e}")
    raise

# 2. MERGE CORE DATA
# Merge races + results
data = pd.merge(
    results,
    races[['raceId', 'year', 'name', 'circuitId', 'date']],
    on='raceId',
    how='left',
    suffixes=('', '_race')
)

# Merge pit stop data (aggregated)
pit_agg = pit_stops.groupby(['raceId', 'driverId']).agg(
    pit_stop_count=('stop', 'count'),
    total_pit_time_ms=('milliseconds', 'sum')
).reset_index()
data = pd.merge(data, pit_agg, on=['raceId', 'driverId'], how='left')

# Merge driver and constructor names
data = pd.merge(
    data,
    drivers[['driverId', 'driverRef']],
    on='driverId',
    how='left'
)
data = pd.merge(
    data,
    constructors[['constructorId', 'name']],
    on='constructorId',
    how='left',
    suffixes=('', '_constructor')
)

# 3. FEATURE ENGINEERING
# Target variable
data['win'] = (data['positionOrder'] == 1).astype(int)

# Position gain
data['position_gain'] = data['grid'] - data['positionOrder']

# Pit stop efficiency
data['avg_pit_time_ms'] = np.where(
    data['pit_stop_count'] > 0,
    data['total_pit_time_ms'] / data['pit_stop_count'],
    0
)

# Weather impact (if weather data exists)
if has_weather and 'Rainfall' in weather.columns:
    # Merge weather data if available
    data = pd.merge(data, weather, on='raceId', how='left')
    data['is_wet'] = (data['Rainfall'] > 0).astype(int)

# 4. FINAL DATA VALIDATION
# Check for missing values
print("\n=== Missing Values ===")
print(data.isnull().sum())

# Verify critical columns exist
required_columns = ['year', 'name', 'driverRef', 'grid', 'positionOrder', 'win', 'pit_stop_count']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    print(f"\nWARNING: Missing columns {missing_cols}")
else:
    print("\nAll required columns present")

# 5. SAMPLE OUTPUT
print("\n=== Sample Data ===")
sample_cols = ['year', 'name', 'driverRef', 'grid', 'positionOrder', 'win', 'pit_stop_count']
print(data[sample_cols].head(3))

# 6. SAVE FINAL DATA
output_path = 'f1_processed_final.csv'
data.to_csv(output_path, index=False)
print(f"\nData successfully saved to {output_path}")


# ## Step 3: Exploratory Data Analysis (EDA)

# In[27]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load preprocessed data with error handling
try:
    data = pd.read_csv('f1_processed_final.csv')

    # Ensure required columns exist
    required_cols = ['grid', 'positionOrder', 'win']
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# =============================================
# 1. CORRELATION ANALYSIS (UPDATED)
# =============================================
plt.figure(figsize=(12,8))
numeric_cols = ['grid', 'positionOrder', 'win']
if 'pit_stop_count' in data.columns:
    numeric_cols.append('pit_stop_count')
if 'avg_pit_time_ms' in data.columns:
    numeric_cols.append('avg_pit_time_ms')
if 'is_wet' in data.columns:
    numeric_cols.append('is_wet')

corr_matrix = data[numeric_cols].corr()

# Filter out non-numeric columns that might have sneaked in
corr_matrix = corr_matrix.select_dtypes(include=[np.number])

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()  # Use close() instead of show() to prevent display issues

# =============================================
# 2. QUALIFYING VS RACE PERFORMANCE (SAFER VERSION)
# =============================================
plt.figure(figsize=(10,6))

# Ensure we have enough data points
plot_data = data.dropna(subset=['grid', 'positionOrder']).copy()
if len(plot_data) > 10000:
    plot_data = plot_data.sample(10000)

sns.scatterplot(
    x='grid', 
    y='positionOrder',
    hue='win',
    alpha=0.6,
    palette={0:'blue', 1:'red'},
    data=plot_data
)
plt.plot([1,20], [1,20], 'k--')
plt.xlabel("Qualifying Position (Grid)")
plt.ylabel("Race Finish Position")
plt.title("Qualifying vs Race Performance")
plt.savefig('qualifying_vs_race.png', dpi=300, bbox_inches='tight')
plt.close()

# =============================================
# 3. PIT STOP ANALYSIS (WITH MORE ROBUST CHECKS)
# =============================================
if all(col in data.columns for col in ['pit_stop_count', 'positionOrder', 'avg_pit_time_ms']):
    plt.figure(figsize=(12,6))

    # Subplot 1: Pit stop count impact
    plt.subplot(1,2,1)
    pit_data = data[data['pit_stop_count'] <= 4].dropna(subset=['pit_stop_count', 'positionOrder'])
    sns.boxplot(
        x='pit_stop_count', 
        y='positionOrder', 
        data=pit_data,
        showfliers=False  # Remove outliers for clearer visualization
    )
    plt.title("Pit Stop Count vs Finish Position")
    plt.xlabel("Number of Pit Stops")
    plt.ylabel("Race Position")

    # Subplot 2: Pit stop time impact
    plt.subplot(1,2,2)
    time_data = data[
        (data['avg_pit_time_ms'] > 0) & 
        (data['avg_pit_time_ms'] < 60000)
    ].dropna(subset=['avg_pit_time_ms', 'position_gain'])

    if len(time_data) > 5000:
        time_data = time_data.sample(5000)

    sns.scatterplot(
        x='avg_pit_time_ms', 
        y='position_gain',
        hue='pit_stop_count',
        palette='viridis',
        alpha=0.6,
        data=time_data
    )
    plt.title("Pit Stop Efficiency")
    plt.xlabel("Average Pit Time (ms)")
    plt.ylabel("Positions Gained")
    plt.tight_layout()
    plt.savefig('pit_stop_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# =============================================
# 4. WINNER ANALYSIS (WITH DATA VALIDATION)
# =============================================
if 'driverRef' in data.columns and 'win' in data.columns:
    plt.figure(figsize=(10,6))
    winners = data[data['win'] == 1].copy()

    if not winners.empty:
        top_winners = winners['driverRef'].value_counts().head(10)
        if not top_winners.empty:
            sns.barplot(
                x=top_winners.values, 
                y=top_winners.index, 
                palette='Reds_r'
            )
            plt.title("Top 10 Winning Drivers (All Time)")
            plt.xlabel("Number of Wins")
            plt.savefig('top_winners.png', dpi=300, bbox_inches='tight')
            plt.close()

# =============================================
# 5. WEATHER IMPACT (MORE ROBUST)
# =============================================
if 'is_wet' in data.columns and 'position_gain' in data.columns:
    plt.figure(figsize=(8,5))
    weather_data = data.dropna(subset=['is_wet', 'position_gain'])
    if not weather_data.empty:
        wet_vs_dry = weather_data.groupby('is_wet')['position_gain'].mean()
        sns.barplot(
            x=wet_vs_dry.index, 
            y=wet_vs_dry.values,
            palette=['skyblue', 'darkblue']
        )
        plt.xticks([0,1], ['Dry', 'Wet'])
        plt.ylabel("Average Position Gain")
        plt.title("Performance in Wet vs Dry Conditions")
        plt.savefig('weather_impact.png', dpi=300, bbox_inches='tight')
        plt.close()

# =============================================
# 6. POSITION CHANGE ANALYSIS (SAFER)
# =============================================
if 'position_gain' in data.columns:
    plt.figure(figsize=(10,6))
    gain_data = data['position_gain'].dropna()
    if not gain_data.empty:
        sns.histplot(
            gain_data, 
            bins=30, 
            kde=True,
            color='purple'
        )
        plt.axvline(0, color='red', linestyle='--')
        plt.xlabel("Positions Gained (Grid → Finish)")
        plt.title("Distribution of Position Changes")
        plt.savefig('position_changes.png', dpi=300, bbox_inches='tight')
        plt.close()

# =============================================
# 7. ERA COMPARISON (UPDATED)
# =============================================
if 'year' in data.columns and 'pit_stop_count' in data.columns:
    plt.figure(figsize=(10,6))

    # Create era categories
    data['era'] = pd.cut(
        data['year'],
        bins=[1950, 1980, 1990, 2000, 2010, 2020, data['year'].max()],
        labels=['1950-1980', '1981-1990', '1991-2000', '2001-2010', '2011-2020', '2021-Present']
    )

    era_data = data.dropna(subset=['era', 'pit_stop_count'])
    if not era_data.empty:
        sns.boxplot(
            x='era', 
            y='pit_stop_count', 
            data=era_data,
            showfliers=False
        )
        plt.xticks(rotation=45)
        plt.title("Pit Stop Strategy by Era")
        plt.ylabel("Number of Pit Stops")
        plt.savefig('era_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

print("All visualizations generated successfully!")


# ## Step 4: Feature Engineering

# In[34]:


from sklearn.preprocessing import StandardScaler

# 1. Define your features (replace with actual column names from your data)
features = ['grid', 'positionOrder', 'pit_stop_count', 'avg_pit_time_ms']  # Example features

# 2. Use your actual DataFrame name (from your EDA code it's 'data')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])


# Qualifying-to-Race Delta

# In[39]:


# Calculate position difference (negative = gained positions, positive = lost positions)
data['Quali_Race_Delta'] = data['positionOrder'] - data['grid']

# Alternative: Normalized delta (accounts for starting position)
data['Quali_Race_Delta_Norm'] = (data['positionOrder'] - data['grid']) / data['grid']

# Visualize the impact
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x='win', y='Quali_Race_Delta', data=data)
plt.title("Position Changes for Winners vs Non-Winners")
plt.show()


# checking
# 

# In[38]:


from sklearn.model_selection import train_test_split

# Select the features and handle missing values
data_clean = data[features].copy()
data_clean = data_clean.dropna()  # Remove rows with missing values

# Scale the features
scaled_features = scaler.fit_transform(data_clean)

# Create a DataFrame with the scaled features, maintaining the original index
X = pd.DataFrame(scaled_features, index=data_clean.index, columns=features)
y = data['win'].iloc[data_clean.index]  # Align target with cleaned features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the scaled_df from X (which already contains the scaled features)
scaled_df = X
print("Means:", scaled_df.mean())
print("Std Devs:", scaled_df.std())
# Should show ~0 mean and 1 std dev for each feature


# 
#  Tire Strategy Impact
# 
# In[41]:


if not pit_stops.empty:
    # Calculate stint lengths (difference between pit stops per driver per race)
    stint_lengths = pit_stops.groupby(['raceId', 'driverId'])['lap'].diff().fillna(pit_stops['lap'])

    # Compute average stint length
    avg_stints = stint_lengths.groupby([pit_stops['raceId'], pit_stops['driverId']]).mean().reset_index()
    avg_stints.columns = ['raceId', 'driverId', 'Avg_Stint_Length']

    # Merge into main data (formerly 'merged_data', now use 'data')
    data = pd.merge(data, avg_stints, on=['raceId', 'driverId'], how='left')
    data['Avg_Stint_Length'] = data['Avg_Stint_Length'].fillna(data['laps'])  # If no stops, stint = full race

# Optional: view to verify
print(data[['raceId', 'driverId', 'Avg_Stint_Length']].head())


# Track-Specific Performance

# In[44]:


# Create track difficulty tiers (customize based on research)
high_downforce_tracks = ['monaco', 'hungaroring', 'singapore']
medium_tracks = ['spa', 'silverstone', 'interlagos']
low_downforce_tracks = ['monza', 'baku', 'americas']

# Assign difficulty tier numerically
data['Track_Difficulty'] = data['circuitId'].apply(
    lambda x: 3 if x in high_downforce_tracks
    else 2 if x in medium_tracks
    else 1  # low downforce
)

# Alternative: Use average Quali-Race position delta per track
track_difficulty = data.groupby('circuitId')['Quali_Race_Delta'].mean().to_dict()
data['Track_Difficulty_Score'] = data['circuitId'].map(track_difficulty)

data = pd.read_csv("f1_merged_data.csv")  # or however you're defining your merged dataset



# 4. Weather Impact Score
# 

# In[49]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame and necessary columns exist

# Create 'Weather_Impact' column
data['Weather_Impact'] = (
    data['rainfall'] * 0.7 +
    data['temperature'] * 0.3 * data['is_wet']
)

# Create 'Weather_Category' column
data['Weather_Category'] = pd.cut(
    data['Weather_Impact'],
    bins=[-1, 0, 2, 5, 10],
    labels=['Dry', 'Damp', 'Wet', 'Torrential']
)

# Convert 'positionOrder' to numeric, coercing errors to NaN
data['positionOrder'] = pd.to_numeric(data['positionOrder'], errors='coerce')

# Drop rows with NaN in 'positionOrder' or 'Weather_Category'
data_clean = data.dropna(subset=['positionOrder', 'Weather_Category'])

# Visual correlation with position
sns.barplot(x='Weather_Category', y='positionOrder', data=data_clean)
plt.title("Average Finish Position by Weather Condition")
plt.show()


# In[52]:


print(data.columns)


# Run this in one cell to ensure merged_data is defined before scaling.

# In[61]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and merge data (from your previous steps)
data = pd.read_csv('f1_processed_final.csv')

# 1. Convert '\N' to NaN for numeric columns
numeric_cols = ['grid', 'positionOrder', 'pit_stop_count', 'total_pit_time_ms', 'position_gain']
for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Converts '\N' to NaN

# 2. Select features and handle missing data
features = ['grid', 'positionOrder', 'pit_stop_count', 'total_pit_time_ms']
X = data[features].copy()

# Fill missing values (using median which is more robust than mean)
X = X.fillna(X.median())

# 3. Verify all values are numeric before scaling
print("Data types before scaling:")
print(X.dtypes)

print("\nMissing values after imputation:")
print(X.isnull().sum())

# 4. Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Convert back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=[f'scaled_{col}' for col in features])

# 6. Merge scaled features back with original data
final_data = pd.concat([data, X_scaled_df], axis=1)

# Save results
final_data.to_csv('f1_data_scaled.csv', index=False)
print("\nScaling completed successfully. Data saved to f1_data_scaled.csv")


# In[85]:


# Create composite weather score
merged_data['Weather_Impact'] = (
    merged_data['rainfall'] * 0.7 +  # Weight rainfall more
    merged_data['temperature'] * 0.3 * merged_data['is_wet']  # Temp matters more in wet conditions
)

# Bin into categories
merged_data['Weather_Category'] = pd.cut(
    merged_data['Weather_Impact'],
    bins=[-1, 0, 2, 5, 10],
    labels=['Dry', 'Damp', 'Wet', 'Torrential']
)

# Visual correlation with position
sns.barplot(x='Weather_Category', y='positionOrder', data=merged_data)
plt.title("Average Finish Position by Weather Condition")
plt.show()


# In[86]:


import pandas as pd

# Load weather data
weather_file = 'F1-Weather/F1 Weather(2023-2018).csv'
try:
    weather = pd.read_csv(weather_file)
    print(f"Weather data loaded successfully from {weather_file}")
except FileNotFoundError:
    print(f"Weather file not found: {weather_file}")
    weather = None

# Ensure merged_data is defined
if 'merged_data' not in globals():
    print("Error: 'merged_data' is not defined. Ensure you have merged your core datasets first.")
else:
    # Merge weather data if available
    if weather is not None:
        merged_data = pd.merge(merged_data, weather, on=['raceId', 'year'], how='left')
        print("Weather data merged successfully.")
    else:
        print("Skipping weather data merge due to missing file.")


# Additional useful features

# In[90]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
merged_data = pd.read_csv('f1_merged_data.csv')

# 1. Create Target Variable 'win'
merged_data['win'] = (merged_data['positionOrder'] == 1).astype(int)

# 2. Create Quali_Race_Delta
merged_data['Quali_Race_Delta'] = merged_data['positionOrder'] - merged_data['grid']

# 3. Create Avg_Stint_Length (if pit stop data exists)
if 'pit_stop_count' in merged_data:
    merged_data['pit_stop_count'] = merged_data['pit_stop_count'].fillna(0)
    merged_data['Avg_Stint_Length'] = merged_data['laps'] / (merged_data['pit_stop_count'] + 1)
    merged_data['Avg_Stint_Length'] = merged_data['Avg_Stint_Length'].fillna(merged_data['laps'])
else:
    print("Warning: pit_stop_count not found - skipping Avg_Stint_Length")

# 4. Create Track_Difficulty
track_difficulty_map = {
    'monaco': 3, 'singapore': 3, 'hungaroring': 3,
    'spa': 2, 'silverstone': 2, 'interlagos': 2,
    'monza': 1, 'americas': 1, 'bahrain': 1
}
merged_data['Track_Difficulty'] = merged_data['circuitId'].map(track_difficulty_map).fillna(1)

# 5. Create Weather Features
# First ensure required columns exist
weather_defaults = {
    'rainfall': 0,
    'temperature': 20,
    'is_wet': 0
}
for col, default in weather_defaults.items():
    if col not in merged_data:
        merged_data[col] = default
        print(f"Warning: {col} not found - using default value {default}")

# Then create composite score
merged_data['Weather_Impact'] = (
    merged_data['rainfall'] * 0.7 + 
    merged_data['temperature'] * 0.3 * merged_data['is_wet']
)

# 6. Driver Consistency (rolling std dev)
merged_data = merged_data.sort_values(['driverId', 'date'])
merged_data['Driver_Consistency'] = (
    merged_data.groupby('driverId')['positionOrder']
    .rolling(5, min_periods=1).std()
    .reset_index(level=0, drop=True)
)
merged_data['Driver_Consistency'] = merged_data['Driver_Consistency'].fillna(
    merged_data['Driver_Consistency'].median()
)

# 7. Constructor Form (rolling avg points)
merged_data['Constructor_Form'] = (
    merged_data.groupby('constructorId')['points']
    .rolling(3, min_periods=1).mean()
    .reset_index(level=0, drop=True)
)

# 8. Final Data Check
print("\nMissing values after processing:")
print(merged_data.isnull().sum())

# 9. Correlation Analysis
correlation_cols = ['win', 'Quali_Race_Delta', 'Track_Difficulty', 'Weather_Impact']
if 'Avg_Stint_Length' in merged_data:
    correlation_cols.append('Avg_Stint_Length')

plt.figure(figsize=(10, 8))
sns.heatmap(
    merged_data[correlation_cols].corr(),
    annot=True,
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Feature Correlation Matrix", pad=20)
plt.tight_layout()
plt.show()

# 10. Save Enhanced Data
merged_data.to_csv('f1_data_enhanced.csv', index=False)
print("\nEnhanced data saved to f1_data_enhanced.csv")


# 
# ## Train MODEL

# Prepare the Data

# In[4]:


# Select features (using only those we created earlier)
features = [
    'grid', 
    'Quali_Race_Delta',
    'Track_Difficulty',
    'Weather_Impact',
    'Driver_Consistency',
    'Constructor_Form'
]
if 'Avg_Stint_Length' in merged_data.columns:
    features.append('Avg_Stint_Length')

X = merged_data[features]
y = merged_data['win']

# Check class balance
print("Class distribution:\n", y.value_counts(normalize=True))


# In[3]:


import pandas as pd

# Example dummy data
merged_data = pd.DataFrame({
    'grid': [1, 2, 3],
    'Quali_Race_Delta': [0.2, -0.1, 0.3],
    'Track_Difficulty': [5, 7, 6],
    'Weather_Impact': [2, 3, 1],
    'Driver_Consistency': [0.9, 0.8, 0.85],
    'Constructor_Form': [0.7, 0.6, 0.65],
    'Avg_Stint_Length': [15.3, 14.7, 16.1],
    'win': [1, 0, 0]
})


# 2. Train-Test Split

# In[7]:


from sklearn.model_selection import train_test_split

# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")


# In[6]:


print(y.value_counts())


# 3. Feature Scaling

# In[8]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. Baseline Model (Logistic Regression)

# In[10]:


# Run this before training:

print("Train label distribution:\n", y_train.value_counts())
print("Test label distribution:\n", y_test.value_counts())


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Train the model safely
unique_classes = np.unique(y_train)

if len(unique_classes) == 1:
    print("⚠️ Only one class in training data. Injecting a synthetic opposite-class sample.")

    # Create model (no class_weight in this case)
    lr = LogisticRegression(max_iter=1000)

    # Add synthetic training sample (ONLY to train set!)
    X_synthetic = np.zeros((1, X_train_scaled.shape[1]))
    y_synthetic = [1 - unique_classes[0]]

    X_train_aug = np.vstack([X_train_scaled, X_synthetic])
    y_train_aug = np.append(y_train.values, y_synthetic)

    lr.fit(X_train_aug, y_train_aug)

else:
    # Normal balanced training
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)

# 🧪 Test set must NOT be altered — just predict on it
y_pred_lr = lr.predict(X_test_scaled)
y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# ✅ Confirm same shape
assert len(y_pred_lr) == len(y_test), "Mismatch between prediction and test set sizes!"

# Evaluate
print("\n📊 Logistic Regression Performance:")
print(classification_report(y_test, y_pred_lr, zero_division=0))
print(f"🔍 ROC AUC: {roc_auc_score(y_test, y_proba_lr):.3f}")


# In[14]:


# Separate the two classes
class_0 = merged_data[merged_data['win'] == 0]
class_1 = merged_data[merged_data['win'] == 1]

# Upsample class_1 if needed (here to match class_0 size)
from sklearn.utils import resample

class_1_upsampled = resample(class_1, 
                             replace=True, 
                             n_samples=len(class_0), 
                             random_state=42)

# Combine the balanced dataset
balanced_data = pd.concat([class_0, class_1_upsampled])

# Shuffle
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Now split into features and target
X = balanced_data[features]
y = balanced_data['win']

# Then split into train/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42,
    stratify=y  # now stratify will work fine
)

# Confirm
print("Train label distribution:\n", y_train.value_counts())
print("Test label distribution:\n", y_test.value_counts())


# In[17]:


print("Train shape:", X_train_scaled.shape, y_train.shape)
print("Test shape:", X_test_scaled.shape, y_test.shape)


# 5. Advanced Model (Random Forest)

# In[20]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd
import warnings

# Suppress undefined metric warnings due to small test size
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced_subsample',
    random_state=42
)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\n🌲 Random Forest Performance:")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print(f"📈 ROC AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")

# Feature importances
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n🔍 Feature Importances:")
print(importance_df)


# 6. Gradient Boosting (XGBoost)

# In[4]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load your dataset
try:
    # First try the enhanced dataset
    data = pd.read_csv('f1_data_enhanced.csv')
    print("Successfully loaded f1_data_enhanced.csv")
except FileNotFoundError:
    try:
        # Fall back to merged dataset if enhanced not found
        data = pd.read_csv('f1_merged_data.csv')
        print("Successfully loaded f1_merged_data.csv")
    except FileNotFoundError:
        # Last resort - processed final data
        data = pd.read_csv('f1_processed_final.csv')
        print("Successfully loaded f1_processed_final.csv")
    except FileNotFoundError:
        raise FileNotFoundError("None of the specified datasets were found. Please check the file paths.")

# Ensure 'win' column exists, create it if needed
if 'win' not in data.columns:
    if 'positionOrder' in data.columns:
        data['win'] = (data['positionOrder'] == 1).astype(int)
        print("Created 'win' column based on positionOrder")
    else:
        raise ValueError("Cannot create 'win' column: 'positionOrder' not found in dataset.")

# Define features (X) and target (y)
# Use 'win' as the target column
y = data['win']
# Drop non-feature columns (adjust as needed based on your dataset)
columns_to_drop = ['win', 'positionOrder']  # Add other non-feature columns if necessary
X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Preprocess the data
# Handle categorical columns (e.g., driver names, teams)
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
else:
    print("No categorical columns detected.")

# Handle missing values (fill with mean for numeric columns)
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
if X[numeric_cols].isnull().any().any():
    print("Filling missing values in numeric columns with their mean.")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
else:
    print("No missing values in numeric columns.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # Handle class imbalance
    random_state=42
)
xgb.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

print("\nXGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_xgb):.3f}")


# In[7]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Function to plot Precision-Recall curve
def plot_pr_curve(y_true, y_proba, label):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{label} (AUC = {pr_auc:.2f})')

# Load your dataset
try:
    data = pd.read_csv('f1_data_enhanced.csv')
    print("Successfully loaded f1_data_enhanced.csv")
except FileNotFoundError:
    try:
        data = pd.read_csv('f1_merged_data.csv')
        print("Successfully loaded f1_merged_data.csv")
    except FileNotFoundError:
        try:
            data = pd.read_csv('f1_processed_final.csv')
            print("Successfully loaded f1_processed_final.csv")
        except FileNotFoundError:
            raise FileNotFoundError("None of the specified datasets were found. Please check the file paths.")

# Ensure 'win' column exists, create it if needed
if 'win' not in data.columns:
    if 'positionOrder' in data.columns:
        data['win'] = (data['positionOrder'] == 1).astype(int)
        print("Created 'win' column based on positionOrder")
    else:
        raise ValueError("Cannot create 'win' column: 'positionOrder' not found in dataset.")

# Define features (X) and target (y)
y = data['win']
columns_to_drop = ['win', 'positionOrder']  # Adjust based on your dataset
X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Preprocess the data
# Handle categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
else:
    print("No categorical columns detected.")

# Handle missing values
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
if X[numeric_cols].isnull().any().any():
    print("Filling missing values in numeric columns with their mean.")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
else:
    print("No missing values in numeric columns.")

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print("Applied StandardScaler to numeric columns.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# XGBoost
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    random_state=42
)
xgb.fit(X_train, y_train)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# Evaluate XGBoost (as in your original code)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_xgb):.3f}")

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
plot_pr_curve(y_test, y_proba_lr, "Logistic Regression")
plot_pr_curve(y_test, y_proba_rf, "Random Forest")
plot_pr_curve(y_test, y_proba_xgb, "XGBoost")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()


# 8. Save the Best Model

# In[11]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# Function to plot Precision-Recall curve
def plot_pr_curve(y_true, y_proba, label):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{label} (AUC = {pr_auc:.2f})')

# Load your dataset
try:
    data = pd.read_csv('f1_data_enhanced.csv')
    print("Successfully loaded f1_data_enhanced.csv")
except FileNotFoundError:
    try:
        data = pd.read_csv('f1_merged_data.csv')
        print("Successfully loaded f1_merged_data.csv")
    except FileNotFoundError:
        try:
            data = pd.read_csv('f1_processed_final.csv')
            print("Successfully loaded f1_processed_final.csv")
        except FileNotFoundError:
            raise FileNotFoundError("None of the specified datasets were found. Please check the file paths.")

# Ensure 'win' column exists, create it if needed
if 'win' not in data.columns:
    if 'positionOrder' in data.columns:
        data['win'] = (data['positionOrder'] == 1).astype(int)
        print("Created 'win' column based on positionOrder")
    else:
        raise ValueError("Cannot create 'win' column: 'positionOrder' not found in dataset.")

# Define features (X) and target (y)
y = data['win']
columns_to_drop = ['win', 'positionOrder']  # Adjust based on your dataset
X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

# Preprocess the data
# Handle categorical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
else:
    print("No categorical columns detected.")

# Handle missing values
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
if X[numeric_cols].isnull().any().any():
    print("Filling missing values in numeric columns with their mean.")
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
else:
    print("No missing values in numeric columns.")

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
print("Applied StandardScaler to numeric columns.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train models
# Logistic Regression
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
y_proba_lr = lr.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# XGBoost
xgb = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
    random_state=42
)
xgb.fit(X_train, y_train)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]

# Evaluate XGBoost
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost Performance:")
print(classification_report(y_test, y_pred_xgb))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_xgb):.3f}")

# Evaluate Random Forest with additional metrics
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Performance:")
print(classification_report(y_test, y_pred_rf))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.3f}")

# Accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.3f}")

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nRandom Forest Confusion Matrix:")
print(cm_rf)

# Feature Importance
importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
print("\nRandom Forest Feature Importance:")
print(feature_importance_df)

# Plot Precision-Recall curves
plt.figure(figsize=(10, 6))
plot_pr_curve(y_test, y_proba_lr, "Logistic Regression")
plot_pr_curve(y_test, y_proba_rf, "Random Forest")
plot_pr_curve(y_test, y_proba_xgb, "XGBoost")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Save XGBoost model
joblib.dump(xgb, 'f1_winner_predictor_xgb.pkl')
print("Saved XGBoost model to 'f1_winner_predictor_xgb.pkl'.")

# Save Random Forest model
joblib.dump(rf, 'f1_winner_predictor_rf.pkl')
print("Saved Random Forest model to 'f1_winner_predictor_rf.pkl'.")

# Save scaler
joblib.dump(scaler, 'feature_scaler.pkl')
print("Saved scaler to 'feature_scaler.pkl'.")


# ## Step 7: Predictions & Insights

# In[ ]:


import pandas as pd
from datetime import datetime

# ────────── 1. Your prebuilt calendar of upcoming races ──────────
# Make sure this DataFrame has one row per future race, with:
#  • a 'race_date' column of pd.Timestamp,
#  • exactly the same feature columns you trained on.
upcoming_races = pd.DataFrame([
    {
        'race_date': pd.Timestamp('2025-04-27'),
        'grid': 5,
        'Quali_Race_Delta': 0.1,
        'Track_Difficulty': 7,
        'Weather_Impact': 2,
        'Driver_Consistency': 0.85,
        'Constructor_Form': 0.9,
        'Avg_Stint_Length': 14.8
    },
    # … add more future races here …
])
features = ['grid', 'Quali_Race_Delta', 'Track_Difficulty',
            'Weather_Impact', 'Driver_Consistency',
            'Constructor_Form', 'Avg_Stint_Length']

# ────────── 2. Your trained RandomForest model ──────────
# (Make sure `rf` is already fitted on your full training set.)
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(...).fit(X_train, y_train)

def predict_finish_by_date(date_str: str):
    """Lookup the row for date_str, run rf.predict & rf.predict_proba."""
    dt = pd.to_datetime(date_str)
    row = upcoming_races.loc[upcoming_races['race_date'] == dt]
    if row.empty:
        raise ValueError(f"No upcoming race found on {date_str}")
    X_new = row[features]
    pred_label = rf.predict(X_new)[0]
    pred_prob  = rf.predict_proba(X_new)[0, 1]
    return pred_label, pred_prob

# ────────── 3. Interactive prompt ──────────
if __name__ == "__main__":
    date_input = input("Enter next race date (YYYY‑MM‑DD): ")
    try:
        win_label, win_prob = predict_finish_by_date(date_input)
        print(f"On {date_input} → predicted win = {win_label} "
              f"(probability = {win_prob:.2%})")
    except Exception as e:
        print(f"❌ {e}")


# In[2]:


import joblib

# Determine the best model based on ROC AUC score
try:
    # Compare models - use the ones already trained in previous cells
    model_scores = {
        'Logistic Regression': roc_auc_score(y_test, y_proba_lr) if 'y_proba_lr' in globals() else 0,
        'Random Forest': roc_auc_score(y_test, y_proba_rf) if 'y_proba_rf' in globals() else 0,
        'XGBoost': roc_auc_score(y_test, y_proba_xgb) if 'y_proba_xgb' in globals() else 0
    }

    best_model_name = max(model_scores, key=model_scores.get)
    print(f"Best model: {best_model_name} (ROC AUC: {model_scores[best_model_name]:.3f})")

    # Set the best model
    if best_model_name == 'XGBoost' and 'xgb' in globals():
        best_model = xgb
    elif best_model_name == 'Random Forest' and 'rf' in globals():
        best_model = rf
    elif best_model_name == 'Logistic Regression' and 'lr' in globals():
        best_model = lr
    else:
        print("Warning: Best model not found in current session")
        best_model = None

    # Create results dataframe
    results_df = pd.DataFrame({
        'Model': list(model_scores.keys()),
        'ROC_AUC': list(model_scores.values())
    })

    # Use feature importance if available
    if 'feature_importance_df' in globals():
        feature_importance = feature_importance_df
    else:
        # Create from best model if it's RF or XGB
        if best_model_name == 'Random Forest' and 'rf' in globals():
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf.feature_importances_
            }).sort_values('Importance', ascending=False)
        elif best_model_name == 'XGBoost' and 'xgb' in globals():
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': xgb.feature_importances_
            }).sort_values('Importance', ascending=False)
        else:
            feature_importance = pd.DataFrame(columns=['Feature', 'Importance'])

    # Save results
    if best_model is not None:
        joblib.dump(best_model, 'race_outcome_model.pkl')
        print("Best model saved to 'race_outcome_model.pkl'")

    results_df.to_csv('model_results.csv', index=False)
    print("Model comparison results saved to 'model_results.csv'")

    feature_importance.to_csv('feature_importance.csv', index=False)
    print("Feature importance saved to 'feature_importance.csv'")

except Exception as e:
    print(f"Error saving model results: {e}")


# In[6]:


# Step 7: Prediction (Add this code cell)
import subprocess

# Install required packages
subprocess.run(['pip', 'install', 'scikit-learn', 'xgboost', 'tensorflow'], check=True)

# Correct imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# ────────── 1. Your prebuilt calendar of upcoming races ──────────
# Make sure this DataFrame has one row per future race, with:
#  • a 'race_date' column of pd.Timestamp,
#  • exactly the same feature columns you trained on.
upcoming_races = pd.DataFrame([
    {
        'race_date': pd.Timestamp('2025-04-27'),
        'grid': 5,
        'Quali_Race_Delta': 0.1,
        'Track_Difficulty': 7,
        'Weather_Impact': 2,
        'Driver_Consistency': 0.85,
        'Constructor_Form': 0.9,
        'Avg_Stint_Length': 14.8
    },
    # … add more future races here …
])
features = ['grid', 'Quali_Race_Delta', 'Track_Difficulty',
            'Weather_Impact', 'Driver_Consistency',
            'Constructor_Form', 'Avg_Stint_Length']

# ────────── 2. Your trained RandomForest model ──────────
# (Make sure `rf` is already fitted on your full training set.)
# from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(...).fit(X_train, y_train)

def predict_finish_by_date(date_str: str):
    """Lookup the row for date_str, run rf.predict & rf.predict_proba."""
    dt = pd.to_datetime(date_str)
    row = upcoming_races.loc[upcoming_races['race_date'] == dt]
    if row.empty:
        raise ValueError(f"No upcoming race found on {date_str}")
    X_new = row[features]
    pred_label = rf.predict(X_new)[0]
    pred_prob  = rf.predict_proba(X_new)[0, 1]
    return pred_label, pred_prob

# ────────── 3. Interactive prompt ──────────
if __name__ == "__main__":
    date_input = input("Enter next race date (YYYY‑MM‑DD): ")
    try:
        win_label, win_prob = predict_finish_by_date(date_input)
        print(f"On {date_input} → predicted win = {win_label} "
              f"(probability = {win_prob:.2%})")
    except Exception as e:
        print(f"❌ {e}")




