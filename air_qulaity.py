import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def load_air_quality_data():
    """
    Load air quality dataset
    """
   
    np.random.seed(42)
    
    cities_data = {
        'Delhi': {'country': 'India', 'pm25_range': (80, 250), 'pm10_range': (120, 400)},
        'Mumbai': {'country': 'India', 'pm25_range': (60, 180), 'pm10_range': (90, 300)},
        'Beijing': {'country': 'China', 'pm25_range': (70, 220), 'pm10_range': (100, 350)},
        'New York': {'country': 'USA', 'pm25_range': (8, 35), 'pm10_range': (15, 60)},
        'London': {'country': 'UK', 'pm25_range': (10, 40), 'pm10_range': (18, 70)},
        'Tokyo': {'country': 'Japan', 'pm25_range': (12, 45), 'pm10_range': (20, 75)},
        'Paris': {'country': 'France', 'pm25_range': (15, 50), 'pm10_range': (25, 80)},
        'Berlin': {'country': 'Germany', 'pm25_range': (12, 42), 'pm10_range': (22, 75)},
        'Singapore': {'country': 'Singapore', 'pm25_range': (20, 60), 'pm10_range': (30, 90)},
        'Dubai': {'country': 'UAE', 'pm25_range': (40, 120), 'pm10_range': (60, 200)}
    }
    
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    data = []
    

    start_date = datetime(2023, 1, 1)
    for day in range(365):  
        current_date = start_date + timedelta(days=day)
        
        for city, info in cities_data.items():
            for pollutant in pollutants:
               
                seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * day / 365)  
                
                if pollutant == 'PM2.5':
                    value = np.random.uniform(*info['pm25_range']) * seasonal_factor
                elif pollutant == 'PM10':
                    value = np.random.uniform(*info['pm10_range']) * seasonal_factor
                elif pollutant == 'NO2':
                    value = np.random.uniform(10, 80) * seasonal_factor
                elif pollutant == 'SO2':
                    value = np.random.uniform(5, 40) * seasonal_factor
                elif pollutant == 'CO':
                    value = np.random.uniform(0.5, 3.0)  
                else:  
                    value = np.random.uniform(20, 100) * seasonal_factor
                
               
                value = max(0, value + np.random.normal(0, value * 0.1))
                
               
                if pollutant in ['PM2.5', 'PM10']:
                    if value <= 50:
                        aqi_category = 'Good'
                    elif value <= 100:
                        aqi_category = 'Moderate'
                    elif value <= 200:
                        aqi_category = 'Poor'
                    else:
                        aqi_category = 'Very Poor'
                else:
                    aqi_category = np.random.choice(['Good', 'Moderate', 'Poor', 'Very Poor'], 
                                                  p=[0.3, 0.4, 0.2, 0.1])
                
                data.append({
                    'Date': current_date,
                    'City': city,
                    'Country': info['country'],
                    'Pollutant': pollutant,
                    'Value': round(value, 2),
                    'Unit': 'µg/m³' if pollutant != 'CO' else 'mg/m³',
                    'AQI_Category': aqi_category,
                    'Season': 'Winter' if day < 90 else 'Spring' if day < 180 else 'Summer' if day < 270 else 'Autumn'
                })
    
    df = pd.DataFrame(data)
    print(f" Created  air quality dataset with {len(df)} records!")
    return df

# Load the data
print("Loading Air Quality Data...")
df = load_air_quality_data()


print(f"📊 Dataset shape: {df.shape}")
print("\n🔍 First 8 rows:")
print(df.head(8))
print("\n📋 Basic info:")
print(df.info())
print("\n🏙️ Cities in dataset:")
print(df['City'].unique())
print("\n🌫️ Pollutants measured:")
print(df['Pollutant'].unique())
print("\n📈 Basic statistics:")
print(df['Value'].describe())


print("🎨 Creating Visualizations...")


plt.style.use('default')
sns.set_palette("husl")

# BAR PLOT: Average Pollution by City for PM2.5
plt.figure(figsize=(12, 6))
pm25_data = df[df['Pollutant'] == 'PM2.5']
city_avg = pm25_data.groupby('City')['Value'].mean().sort_values(ascending=False)

plt.subplot(2, 3, 1)
sns.barplot(x=city_avg.values, y=city_avg.index)
plt.title('Average PM2.5 Levels by City')
plt.xlabel('PM2.5 (µg/m³)')
plt.tight_layout()

# LINE PLOT: Pollution Trends Over Time for Top Cities
plt.subplot(2, 3, 2)
top_cities = city_avg.head(3).index
for city in top_cities:
    city_data = pm25_data[pm25_data['City'] == city]
    monthly_avg = city_data.groupby(city_data['Date'].dt.to_period('M'))['Value'].mean()
    monthly_avg.index = monthly_avg.index.astype(str)
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label=city, linewidth=2)

plt.title('Monthly PM2.5 Trends - Top Cities')
plt.xlabel('Month')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend()
plt.xticks(rotation=45)

# BOX PLOT: Pollution Distribution by City
plt.subplot(2, 3, 3)
sns.boxplot(data=pm25_data, x='Value', y='City')
plt.title('PM2.5 Distribution by City')
plt.xlabel('PM2.5 (µg/m³)')

# HEATMAP: Correlation between Pollutants
plt.subplot(2, 3, 4)
pivot_data = df.pivot_table(index=['City', 'Date'], columns='Pollutant', values='Value')
correlation_matrix = pivot_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Pollutant Correlation Heatmap')

# PIE CHART: AQI Category Distribution
plt.subplot(2, 3, 5)
aqi_counts = df['AQI_Category'].value_counts()
plt.pie(aqi_counts.values, labels=aqi_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('AQI Category Distribution')

plt.subplot(2, 3, 6)
scatter_data = df.pivot_table(index=['City', 'Date'], columns='Pollutant', values='Value').reset_index()
plt.scatter(scatter_data['PM2.5'], scatter_data['PM10'], alpha=0.6)
plt.xlabel('PM2.5 (µg/m³)')
plt.ylabel('PM10 (µg/m³)')
plt.title('PM2.5 vs PM10 Correlation')

plt.tight_layout()
plt.show()

# 7. SEASONAL ANALYSIS
print("\n🌤️ Creating Seasonal Analysis...")
plt.figure(figsize=(15, 10))

seasonal_pm25 = df[df['Pollutant'] == 'PM2.5'].groupby(['City', 'Season'])['Value'].mean().reset_index()

plt.subplot(2, 2, 1)
sns.barplot(data=seasonal_pm25, x='Season', y='Value', hue='City')
plt.title('Seasonal PM2.5 Levels by City')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

#  POLLUTANT COMPARISON ACROSS CITIES
plt.subplot(2, 2, 2)
pollutant_avg = df.groupby(['City', 'Pollutant'])['Value'].mean().reset_index()
sns.barplot(data=pollutant_avg, x='City', y='Value', hue='Pollutant')
plt.title('Average Pollutant Levels by City')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

#  TIME SERIES FOR ALL CITIES
plt.subplot(2, 2, 3)
for city in df['City'].unique():
    city_pm25 = df[(df['City'] == city) & (df['Pollutant'] == 'PM2.5')]
    monthly = city_pm25.groupby(city_pm25['Date'].dt.to_period('M'))['Value'].mean()
    plt.plot(monthly.index.astype(str), monthly.values, label=city, alpha=0.7, linewidth=1)

plt.title('PM2.5 Trends - All Cities')
plt.xlabel('Month')
plt.ylabel('PM2.5 (µg/m³)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)


plt.subplot(2, 2, 4)
sns.violinplot(data=df[df['Pollutant'] == 'PM2.5'], x='Season', y='Value')
plt.title('PM2.5 Distribution by Season')

plt.tight_layout()
plt.show()


print("\n🔄 Creating Interactive Visualizations...")

#  Scatter Plot
fig = px.scatter(df[df['Pollutant'] == 'PM2.5'], 
                 x='Date', y='Value', color='City',
                 title='Interactive PM2.5 Levels Over Time',
                 labels={'Value': 'PM2.5 (µg/m³)', 'Date': 'Date'})
fig.show()

# Bar Chart - City Comparison
city_pollutant_avg = df.groupby(['City', 'Pollutant'])['Value'].mean().reset_index()
fig2 = px.bar(city_pollutant_avg, 
              x='City', y='Value', color='Pollutant',
              title='Interactive Pollutant Levels by City',
              barmode='group')
fig2.show()

print("\n KEY INSIGHTS:")
print(f"• Most polluted city (PM2.5): {pm25_data.groupby('City')['Value'].mean().idxmax()}")
print(f"• Least polluted city (PM2.5): {pm25_data.groupby('City')['Value'].mean().idxmin()}")
print(f"• Most common AQI category: {df['AQI_Category'].value_counts().idxmax()}")
print(f"• Strongest pollutant correlation: PM2.5 vs PM10 = {correlation_matrix.loc['PM2.5', 'PM10']:.2f}")

print("\n All visualizations created successfully!")