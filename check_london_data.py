import pandas as pd

# Load data
df = pd.read_csv('LondonBikeJourneyAug2023.csv')

# Basic information
print("\nDataset Info:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicates
print("\nNumber of duplicates:", df.duplicated().sum())

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Check for negative durations
print("\nNegative durations:", (df['Total duration (ms)'] < 0).sum())

# Check for zero durations
print("\nZero durations:", (df['Total duration (ms)'] == 0).sum())

# Check for extreme durations (more than 24 hours)
print("\nExtreme durations (>24h):", (df['Total duration (ms)'] > 24*3600*1000).sum())

# Check for same start and end station
print("\nTrips with same start and end station:", 
      (df['Start station number'] == df['End station number']).sum())

# Check for missing station IDs
print("\nMissing station IDs:", df['Start station number'].isnull().sum())
print("Missing station names:", df['Start station'].isnull().sum())

# Check for missing dates
print("\nMissing dates:", df['Start date'].isnull().sum())

# Convert duration to seconds for easier analysis
df['Duration (sec)'] = df['Total duration (ms)'] / 1000

# Print duration statistics in seconds
print("\nDuration Statistics (in seconds):")
print(df['Duration (sec)'].describe()) 