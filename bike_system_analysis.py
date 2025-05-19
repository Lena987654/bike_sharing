import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import folium
from folium import plugins
from tqdm import tqdm

class BikeSystemAnalyzer:
    """
    Class for analyzing bike journey data from multiple cities
    """
    def __init__(self, london_data=None, helsinki_data=None):
        self.london_data = london_data
        self.helsinki_data = helsinki_data
        
        if london_data is not None:
            self.london_data['hour'] = pd.to_datetime(london_data['Start date']).dt.hour
            self.london_data['day_of_week'] = pd.to_datetime(london_data['Start date']).dt.day_name()
        
        if helsinki_data is not None:
            self.helsinki_data['hour'] = pd.to_datetime(helsinki_data['departure']).dt.hour
            self.helsinki_data['day_of_week'] = pd.to_datetime(helsinki_data['departure']).dt.day_name()

    def visualize_distribution(self, data, x_col, title, xlabel, ylabel, plot_type='line', city='both'):
        """
        Unified function for visualizing distributions
        plot_type: 'line', 'bar', 'scatter', 'kde'
        """
        if city in ['london', 'both'] and self.london_data is not None:
            plt.figure(figsize=(12, 6))
            if plot_type == 'line':
                plt.plot(data.index, data.values, marker='o', label='London')
            elif plot_type == 'bar':
                plt.bar(data.index, data.values, label='London')
            elif plot_type == 'scatter':
                plt.scatter(data.index, data.values, alpha=0.5, label='London')
            elif plot_type == 'kde':
                sns.kdeplot(data=data.values, label='London')
            
            plt.title(f'{title} (London)')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.savefig(f'london_{x_col}_distribution.png')
            plt.close()

        if city in ['helsinki', 'both'] and self.helsinki_data is not None:
            plt.figure(figsize=(12, 6))
            if plot_type == 'line':
                plt.plot(data.index, data.values, marker='o', label='Helsinki')
            elif plot_type == 'bar':
                plt.bar(data.index, data.values, label='Helsinki')
            elif plot_type == 'scatter':
                plt.scatter(data.index, data.values, alpha=0.5, label='Helsinki')
            elif plot_type == 'kde':
                sns.kdeplot(data=data.values, label='Helsinki')
            
            plt.title(f'{title} (Helsinki)')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.legend()
            plt.savefig(f'helsinki_{x_col}_distribution.png')
            plt.close()

    def analyze_hourly_distribution(self, city='both'):
        """Analyze and visualize hourly distribution of rides"""
        if city in ['london', 'both'] and self.london_data is not None:
            london_hourly = self.london_data['hour'].value_counts().sort_index()
            self.visualize_distribution(
                london_hourly, 'hourly',
                'Distribution of Rides by Hour',
                'Hour of Day',
                'Number of Rides',
                'line',
                'london'
            )

        if city in ['helsinki', 'both'] and self.helsinki_data is not None:
            helsinki_hourly = self.helsinki_data['hour'].value_counts().sort_index()
            self.visualize_distribution(
                helsinki_hourly, 'hourly',
                'Distribution of Rides by Hour',
                'Hour of Day',
                'Number of Rides',
                'line',
                'helsinki'
            )

    def analyze_daily_distribution(self, city='both'):
        """Analyze and visualize daily distribution of rides"""
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        if city in ['london', 'both'] and self.london_data is not None:
            london_daily = self.london_data['day_of_week'].value_counts().reindex(days_order)
            self.visualize_distribution(
                london_daily, 'daily',
                'Distribution of Rides by Day of Week',
                'Day of Week',
                'Number of Rides',
                'bar',
                'london'
            )

        if city in ['helsinki', 'both'] and self.helsinki_data is not None:
            helsinki_daily = self.helsinki_data['day_of_week'].value_counts().reindex(days_order)
            self.visualize_distribution(
                helsinki_daily, 'daily',
                'Distribution of Rides by Day of Week',
                'Day of Week',
                'Number of Rides',
                'bar',
                'helsinki'
            )

    def analyze_duration(self, city='both'):
        """Analyze and visualize ride duration patterns"""
        if city in ['london', 'both'] and self.london_data is not None:
            london_duration = self.london_data.groupby('hour')['Total duration (ms)'].mean() / 1000
            self.visualize_distribution(
                london_duration, 'duration',
                'Average Ride Duration by Hour',
                'Hour of Day',
                'Average Duration (seconds)',
                'line',
                'london'
            )

        if city in ['helsinki', 'both'] and self.helsinki_data is not None:
            helsinki_duration = self.helsinki_data.groupby('hour')['duration (sec.)'].mean()
            self.visualize_distribution(
                helsinki_duration, 'duration',
                'Average Ride Duration by Hour',
                'Hour of Day',
                'Average Duration (seconds)',
                'line',
                'helsinki'
            )

    def analyze_temperature_impact(self):
        """Analyze and visualize impact of temperature on ride patterns (Helsinki only)"""
        if self.helsinki_data is not None:
            temp_counts = self.helsinki_data.groupby('Air temperature (degC)').size()
            self.visualize_distribution(
                temp_counts, 'temperature',
                'Number of Rides vs Temperature',
                'Temperature (°C)',
                'Number of Rides',
                'scatter',
                'helsinki'
            )
            
            temp_correlation = self.helsinki_data.groupby('Air temperature (degC)').size().reset_index()
            temp_correlation.columns = ['temperature', 'rides']
            correlation = temp_correlation['temperature'].corr(temp_correlation['rides'])
            
            return temp_counts, correlation
        return None, None

    def analyze_speed_patterns(self):
        """Analyze and visualize speed patterns (Helsinki only)"""
        if self.helsinki_data is not None:
            speed_by_hour = self.helsinki_data.groupby('hour')['avg_speed (km/h)'].mean()
            self.visualize_distribution(
                speed_by_hour, 'speed',
                'Average Speed by Hour',
                'Hour of Day',
                'Average Speed (km/h)',
                'line',
                'helsinki'
            )
            return speed_by_hour
        return None

    def get_basic_stats(self, city='both'):
        """
        Get basic statistics about the dataset(s)
        """
        stats = {}
        
        if city in ['london', 'both'] and self.london_data is not None:
            stats['london'] = {
                'total_rides': len(self.london_data),
                'avg_duration': self.london_data['Total duration (ms)'].mean() / 1000,
                'peak_hour': self.london_data['hour'].value_counts().idxmax(),
                'peak_day': self.london_data['day_of_week'].value_counts().idxmax()
            }
        
        if city in ['helsinki', 'both'] and self.helsinki_data is not None:
            stats['helsinki'] = {
                'total_rides': len(self.helsinki_data),
                'avg_duration': self.helsinki_data['duration (sec.)'].mean(),
                'peak_hour': self.helsinki_data['hour'].value_counts().idxmax(),
                'peak_day': self.helsinki_data['day_of_week'].value_counts().idxmax(),
                'avg_temperature': self.helsinki_data['Air temperature (degC)'].mean(),
                'min_temperature': self.helsinki_data['Air temperature (degC)'].min(),
                'max_temperature': self.helsinki_data['Air temperature (degC)'].max(),
                'avg_speed': self.helsinki_data['avg_speed (km/h)'].mean(),
                'avg_distance': self.helsinki_data['distance (m)'].mean() / 1000  # Convert to km
            }
        
        return stats

def load_data():
    print("Loading datasets...")
    london_df = pd.read_csv('London_August_2023_normalized.csv')
    helsinki_2019_df = pd.read_csv('Helsinki_August_2019_cleaned.csv')
    helsinki_2020_df = pd.read_csv('Helsinki_August_2020_cleaned.csv')
    
    # Convert date columns to datetime
    for df in [london_df]:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
    
    for df in [helsinki_2019_df, helsinki_2020_df]:
        df['departure'] = pd.to_datetime(df['departure'])
        df['return'] = pd.to_datetime(df['return'])
    
    return london_df, helsinki_2019_df, helsinki_2020_df

def classify_trips_and_stations(df, city):
    """Combined function for classifying both trips and stations"""
    print(f"\nClassifying trips and stations for {city}...")
    
    # Prepare features for classification
    if city == 'London':
        df['hour'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['duration'] = df['duration_seconds']
        station_cols = ['start_station', 'end_station']
    else:
        df['hour'] = df['departure'].dt.hour
        df['day_of_week'] = df['departure'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['duration'] = df['duration (sec.)']
        station_cols = ['departure_name', 'return_name']
    
    # Trip classification
    trip_features = pd.DataFrame({
        'hour': df['hour'],
        'is_weekend': df['is_weekend'].astype(int),
        'duration': df['duration'],
        'day_of_week': df['day_of_week']
    })
    
    scaler = StandardScaler()
    scaled_trip_features = scaler.fit_transform(trip_features)
    
    kmeans_trips = KMeans(n_clusters=3, random_state=42)
    df['trip_type'] = kmeans_trips.fit_predict(scaled_trip_features)
    
    # Map trip types
    trip_types = ['work', 'leisure', 'tourist']
    df['trip_type'] = df['trip_type'].map(dict(zip(range(3), trip_types)))
    
    # Station classification
    all_stations = pd.concat([df[col].unique() for col in station_cols]).unique()
    station_patterns = pd.DataFrame(index=all_stations)
    
    # Calculate station patterns
    for col in station_cols:
        station_patterns[f'{col}_hour_mean'] = df.groupby(col)['hour'].mean()
        station_patterns[f'{col}_hour_std'] = df.groupby(col)['hour'].std()
        station_patterns[f'{col}_weekend_ratio'] = df.groupby(col)['is_weekend'].mean()
        station_patterns[f'{col}_duration_mean'] = df.groupby(col)['duration'].mean()
        station_patterns[f'{col}_duration_std'] = df.groupby(col)['duration'].std()
    
    # Fill NaN values
    station_patterns = station_patterns.fillna(station_patterns.mean())
    
    # Scale station features
    station_features = station_patterns.values
    scaled_station_features = scaler.fit_transform(station_features)
    
    # Cluster stations
    kmeans_stations = KMeans(n_clusters=4, random_state=42)
    station_patterns['station_type'] = kmeans_stations.fit_predict(scaled_station_features)
    
    # Map station types
    station_types = ['work', 'leisure', 'balanced', 'seasonal']
    station_patterns['station_type'] = station_patterns['station_type'].map(dict(zip(range(4), station_types)))
    
    return df, station_patterns

def visualize_analysis(df, stations, city):
    """Combined visualization function for all analyses"""
    print(f"\nVisualizing analysis for {city}...")
    
    # Create interaction matrix
    interaction_matrix = pd.crosstab(
        df['trip_type'],
        stations.loc[df[df.columns[0]].drop_duplicates(), 'station_type']
    )
    
    # Normalize matrix
    interaction_matrix_norm = interaction_matrix.div(interaction_matrix.sum(axis=1), axis=0)
    
    # Visualize interaction matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix_norm, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title(f'Trip-Station Interaction Matrix ({city})')
    plt.tight_layout()
    plt.savefig(f'interaction_matrix_{city.lower()}.png')
    plt.close()
    
    # Visualize flows
    for trip_type in df['trip_type'].unique():
        type_trips = df[df['trip_type'] == trip_type]
        flow_matrix = pd.crosstab(
            type_trips['start_station_type'],
            type_trips['end_station_type']
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(flow_matrix, annot=True, fmt='d', cmap='YlOrRd')
        plt.title(f'Flow Matrix for {trip_type} Trips ({city})')
        plt.tight_layout()
        plt.savefig(f'flow_matrix_{trip_type}_{city.lower()}.png')
        plt.close()
    
    # Visualize clustering
    features_to_scale = [col for col in stations.columns if col != 'station_type']
    scaler = StandardScaler()
    scaled_patterns = scaler.fit_transform(stations[features_to_scale])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_patterns)
    
    plt.figure(figsize=(15, 10))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=stations['station_type'].map({
                             'work': 0, 'leisure': 1, 'balanced': 2, 'seasonal': 3
                         }),
                         cmap='viridis',
                         alpha=0.7,
                         s=100)
    
    plt.title(f'Station Clustering Visualization ({city})\n'
              f'Explained variance: {pca.explained_variance_ratio_[0]:.1%} and {pca.explained_variance_ratio_[1]:.1%}')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'clustering_visualization_{city.lower()}.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    print("Starting analysis...")
    
    # Load data
    london_df, helsinki_2019_df, helsinki_2020_df = load_data()
    
    # Process each dataset
    datasets = [
        (london_df, 'London'),
        (helsinki_2019_df, 'Helsinki_2019'),
        (helsinki_2020_df, 'Helsinki_2020')
    ]
    
    for df, city in tqdm(datasets, desc="Processing datasets"):
        print(f"\nProcessing {city}...")
        
        # Classify trips and stations
        df, stations = classify_trips_and_stations(df, city)
        
        # Visualize analysis
        visualize_analysis(df, stations, city)
    
    print("\nAnalysis completed. Check the generated visualizations for detailed patterns.")

if __name__ == "__main__":
    main() 