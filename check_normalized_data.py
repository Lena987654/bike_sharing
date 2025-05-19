import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_data_quality(df, city_name):
    print(f"\n{'='*50}")
    print(f"Checking {city_name} normalized dataset:")
    print(f"{'='*50}")
    
    # Basic info
    print(f"\nTotal records: {len(df):,}")
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values:")
        for col, count in missing.items():
            if count > 0:
                print(f"{col}: {count:,} ({count/len(df)*100:.2f}%)")
    else:
        print("\nNo missing values found")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate records: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    # Check date ranges
    print("\nDate ranges:")
    print(f"Start time: {df['start_time'].min()} to {df['start_time'].max()}")
    print(f"End time: {df['end_time'].min()} to {df['end_time'].max()}")
    
    # Check duration validity
    print("\nDuration statistics (seconds):")
    print(df['duration_seconds'].describe())
    
    # Check for negative durations
    negative_durations = (df['duration_seconds'] < 0).sum()
    print(f"\nNegative durations: {negative_durations:,}")
    
    # Check for zero durations
    zero_durations = (df['duration_seconds'] == 0).sum()
    print(f"Zero durations: {zero_durations:,}")
    
    # Check for very long durations (> 24 hours)
    long_durations = (df['duration_seconds'] > 86400).sum()
    print(f"Durations > 24 hours: {long_durations:,}")
    
    # Check station statistics
    print("\nStation statistics:")
    print(f"Unique start stations: {df['start_station'].nunique():,}")
    print(f"Unique end stations: {df['end_station'].nunique():,}")
    
    return df

def compare_datasets(london_df, helsinki_df):
    print("\n" + "="*50)
    print("Datasets Comparison:")
    print("="*50)
    
    # Compare total rides
    print(f"\nTotal rides:")
    print(f"London: {len(london_df):,}")
    print(f"Helsinki: {len(helsinki_df):,}")
    print(f"Difference: {len(london_df) - len(helsinki_df):,}")
    
    # Compare duration statistics
    print("\nDuration comparison (seconds):")
    print("London:")
    print(london_df['duration_seconds'].describe())
    print("\nHelsinki:")
    print(helsinki_df['duration_seconds'].describe())
    
    # Compare station counts
    print("\nStation counts:")
    print("London:")
    print(f"Start stations: {london_df['start_station'].nunique():,}")
    print(f"End stations: {london_df['end_station'].nunique():,}")
    print("\nHelsinki:")
    print(f"Start stations: {helsinki_df['start_station'].nunique():,}")
    print(f"End stations: {helsinki_df['end_station'].nunique():,}")
    
    # Compare daily patterns
    print("\nDaily ride counts:")
    london_daily = london_df['start_time'].dt.date.value_counts().sort_index()
    helsinki_daily = helsinki_df['start_time'].dt.date.value_counts().sort_index()
    
    print("\nLondon daily rides:")
    print(london_daily)
    print("\nHelsinki daily rides:")
    print(helsinki_daily)

def main():
    print("Loading normalized datasets...")
    
    # Load normalized datasets
    london_df = pd.read_csv('London_August_2023_normalized.csv')
    helsinki_df = pd.read_csv('Helsinki_August_2019_normalized.csv')
    
    # Convert date columns to datetime
    london_df['start_time'] = pd.to_datetime(london_df['start_time'])
    london_df['end_time'] = pd.to_datetime(london_df['end_time'])
    helsinki_df['start_time'] = pd.to_datetime(helsinki_df['start_time'])
    helsinki_df['end_time'] = pd.to_datetime(helsinki_df['end_time'])
    
    # Check quality of each dataset
    london_df = check_data_quality(london_df, 'London')
    helsinki_df = check_data_quality(helsinki_df, 'Helsinki')
    
    # Compare datasets
    compare_datasets(london_df, helsinki_df)

if __name__ == "__main__":
    main() 