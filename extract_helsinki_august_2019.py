import pandas as pd
import numpy as np

print("Loading Helsinki data...")
df = pd.read_csv('Helsinki.csv')

print("Converting dates...")
df['departure'] = pd.to_datetime(df['departure'])
df['return'] = pd.to_datetime(df['return'])

# Filter for August 2019
print("Filtering for August 2019...")
august_2019 = df[
    (df['departure'].dt.year == 2019) & 
    (df['departure'].dt.month == 8)
].copy()

print("\nInitial data quality check:")
print(f"Total records: {len(august_2019):,}")

# Check for missing values
print("\nMissing values:")
print(august_2019.isnull().sum())

# Check for invalid values
print("\nChecking for invalid values...")
print("Duration statistics (seconds):")
print(august_2019['duration (sec.)'].describe())

print("\nDistance statistics (meters):")
print(august_2019['distance (m)'].describe())

print("\nSpeed statistics (km/h):")
print(august_2019['avg_speed (km/h)'].describe())

# Basic data cleaning
print("\nCleaning data...")

# 1. Remove rides with invalid duration (less than 10 seconds or more than 24 hours)
valid_duration = (august_2019['duration (sec.)'] >= 10) & (august_2019['duration (sec.)'] <= 86400)
invalid_duration_count = (~valid_duration).sum()
print(f"Rides with invalid duration: {invalid_duration_count:,}")

# 2. Remove rides with invalid distance (less than 10 meters or more than 100km)
valid_distance = (august_2019['distance (m)'] >= 10) & (august_2019['distance (m)'] <= 100000)
invalid_distance_count = (~valid_distance).sum()
print(f"Rides with invalid distance: {invalid_distance_count:,}")

# 3. Remove rides with unrealistic speeds (more than 50 km/h)
valid_speed = (august_2019['avg_speed (km/h)'] > 0) & (august_2019['avg_speed (km/h)'] <= 50)
invalid_speed_count = (~valid_speed).sum()
print(f"Rides with invalid speed: {invalid_speed_count:,}")

# Apply all filters
clean_data = august_2019[valid_duration & valid_distance & valid_speed]

print(f"\nRecords after cleaning: {len(clean_data):,}")
print(f"Removed {len(august_2019) - len(clean_data):,} invalid records")

# Save cleaned dataset
output_file = 'Helsinki_August_2019_cleaned.csv'
clean_data.to_csv(output_file, index=False)
print(f"\nCleaned data saved to {output_file}")

# Print final statistics
print("\nFinal dataset statistics:")
print("\nDuration statistics (seconds):")
print(clean_data['duration (sec.)'].describe())
print("\nDistance statistics (meters):")
print(clean_data['distance (m)'].describe())
print("\nSpeed statistics (km/h):")
print(clean_data['avg_speed (km/h)'].describe()) 