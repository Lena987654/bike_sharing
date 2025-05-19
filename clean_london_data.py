import pandas as pd
import numpy as np

def clean_london_data():
    # Load data
    print("Loading data...")
    df = pd.read_csv('LondonBikeJourneyAug2023.csv')
    
    # Print initial statistics
    print(f"\nInitial number of rides: {len(df)}")
    print(f"Rides longer than 24h: {(df['Total duration (ms)'] > 24*3600*1000).sum()}")
    print(f"Rides with same start and end station: {(df['Start station number'] == df['End station number']).sum()}")
    
    # Remove trips longer than 24 hours
    df = df[df['Total duration (ms)'] <= 24*3600*1000]
    
    # Convert duration to seconds for easier analysis
    df['Duration (sec)'] = df['Total duration (ms)'] / 1000
    
    # Print final statistics
    print(f"\nFinal number of rides: {len(df)}")
    print("\nDuration Statistics (in seconds):")
    print(df['Duration (sec)'].describe())
    
    # Save cleaned dataset
    df.to_csv('LondonBikeJourneyAug2023_cleaned.csv', index=False)
    print("\nCleaned data saved to 'LondonBikeJourneyAug2023_cleaned.csv'")
    
    return df

if __name__ == "__main__":
    clean_london_data() 