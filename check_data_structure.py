import pandas as pd

def check_file_structure(file_path):
    print(f"\nChecking structure of {file_path}:")
    df = pd.read_csv(file_path, nrows=5)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df)

if __name__ == "__main__":
    check_file_structure('Helsinki_August_2019_cleaned.csv')
    check_file_structure('Helsinki_August_2020_cleaned.csv') 