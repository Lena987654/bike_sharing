import pandas as pd
import numpy as np
from datetime import datetime

def normalize_london_data(df):
    # Convert duration from milliseconds to seconds
    df['duration_seconds'] = df['Total duration (ms)'] / 1000
    
    # Select and rename columns
    normalized = pd.DataFrame({
        'start_time': pd.to_datetime(df['Start date']),
        'end_time': pd.to_datetime(df['End date']),
        'start_station': df['Start station'],
        'end_station': df['End station'],
        'duration_seconds': df['duration_seconds']
    })
    
    return normalized

def normalize_helsinki_data(input_file, output_file):
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    
    # Convert date columns to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # Calculate duration in seconds
    df['duration_seconds'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    
    # Remove rows with negative or zero duration
    df = df[df['duration_seconds'] > 0]
    
    # Remove rows with duration > 24 hours
    df = df[df['duration_seconds'] <= 24 * 3600]
    
    # Save normalized data
    print(f"Saving normalized data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

def main():
    print("Loading datasets...")
    
    # Load original datasets
    london_df = pd.read_csv('LondonBikeJourneyAug2023.csv')
    
    print("\nNormalizing datasets...")
    
    # Normalize datasets
    london_normalized = normalize_london_data(london_df)
    
    # Save normalized datasets
    london_normalized.to_csv('London_August_2023_normalized.csv', index=False)
    
    # Print summary
    print("\nNormalized datasets summary:")
    print("\nLondon dataset:")
    print(f"Records: {len(london_normalized):,}")
    print("Columns:", london_normalized.columns.tolist())
    print("\nDuration statistics (seconds):")
    print(london_normalized['duration_seconds'].describe())
    
    # Normalize Helsinki 2019 data
    normalize_helsinki_data('Helsinki_August_2019_cleaned.csv', 'Helsinki_August_2019_normalized.csv')
    
    # Normalize Helsinki 2020 data
    normalize_helsinki_data('Helsinki_August_2020_cleaned.csv', 'Helsinki_August_2020_normalized.csv')

if __name__ == "__main__":
    main() 