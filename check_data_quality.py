import pandas as pd
import numpy as np

def check_dataset(file_path, city_name):
    print(f"\n{'='*50}")
    print(f"Checking {city_name} dataset:")
    print(f"{'='*50}")
    
    # Load data
    print(f"\nLoading {city_name} data...")
    df = pd.read_csv(file_path)
    
    # Basic info
    print(f"\nTotal records: {len(df):,}")
    print("\nColumns:")
    for col in df.columns:
        print(f"- {col}")
    
    # Check missing values
    print("\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"{col}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Check data types
    print("\nData types:")
    print(df.dtypes)
    
    # Basic statistics for numeric columns
    print("\nBasic statistics for numeric columns:")
    print(df.describe())
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"\nDuplicate records: {duplicates:,} ({duplicates/len(df)*100:.2f}%)")
    
    # Check date range
    if 'Start date' in df.columns:
        df['Start date'] = pd.to_datetime(df['Start date'])
        print(f"\nDate range: {df['Start date'].min()} to {df['Start date'].max()}")
    elif 'departure' in df.columns:
        df['departure'] = pd.to_datetime(df['departure'])
        print(f"\nDate range: {df['departure'].min()} to {df['departure'].max()}")
    
    return df

def main():
    # Check London data
    london_df = check_dataset('LondonBikeJourneyAug2023.csv', 'London')
    
    # Check Helsinki data
    helsinki_df = check_dataset('Helsinki_August_2019_cleaned.csv', 'Helsinki')
    
    # Compare key metrics
    print("\n" + "="*50)
    print("Key Metrics Comparison:")
    print("="*50)
    
    # Compare total rides
    print(f"\nTotal rides:")
    print(f"London: {len(london_df):,}")
    print(f"Helsinki: {len(helsinki_df):,}")
    
    # Compare date ranges
    print("\nDate ranges:")
    print(f"London: {london_df['Start date'].min()} to {london_df['Start date'].max()}")
    print(f"Helsinki: {helsinki_df['departure'].min()} to {helsinki_df['departure'].max()}")
    
    # Compare missing values
    print("\nMissing values comparison:")
    print("London:")
    london_missing = london_df.isnull().sum()
    for col, count in london_missing.items():
        if count > 0:
            print(f"{col}: {count:,} ({count/len(london_df)*100:.2f}%)")
    
    print("\nHelsinki:")
    helsinki_missing = helsinki_df.isnull().sum()
    for col, count in helsinki_missing.items():
        if count > 0:
            print(f"{col}: {count:,} ({count/len(helsinki_df)*100:.2f}%)")

if __name__ == "__main__":
    main() 