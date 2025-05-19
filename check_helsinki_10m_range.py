import pandas as pd

# Load only the departure column from Helsinki data
print("Loading Helsinki data (departure times only)...")
helsinki_data = pd.read_csv('Helsinki.csv', 
                          usecols=['departure'],
                          engine='python')

# Convert dates to datetime
print("Converting dates...")
helsinki_data['departure'] = pd.to_datetime(helsinki_data['departure'])

# Get date range
start_date = helsinki_data['departure'].min()
end_date = helsinki_data['departure'].max()

print(f"\nDate range in Helsinki dataset:")
print(f"Start date: {start_date}")
print(f"End date: {end_date}")

# Get monthly distribution
print("\nMonthly distribution:")
monthly_counts = helsinki_data['departure'].dt.to_period('M').value_counts().sort_index()
for month, count in monthly_counts.items():
    print(f"{month}: {count:,} rides") 