import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    """
    Class responsible for loading and preprocessing datasets
    """
    def __init__(self):
        self.london_data = None
        self.helsinki_data = None
        self.london_stations = None

    def load_all_data(self):
        """
        Load all datasets and perform initial preprocessing
        """
        print("Loading London data...")
        self.london_data = pd.read_csv('LondonBikeJourneyAug2023.csv')
        
        print("Loading Helsinki data...")
        self.helsinki_data = pd.read_csv('Helsinki_August_2020_cleaned.csv')
        
        # Convert dates to datetime
        self.london_data['Start date'] = pd.to_datetime(self.london_data['Start date'])
        self.helsinki_data['departure'] = pd.to_datetime(self.helsinki_data['departure'])
        
        print("\nData Summary:")
        print(f"London records: {len(self.london_data):,}")
        print(f"Helsinki records: {len(self.helsinki_data):,}")
        
        print("\nDate ranges:")
        print(f"London: {self.london_data['Start date'].min()} to {self.london_data['Start date'].max()}")
        print(f"Helsinki: {self.helsinki_data['departure'].min()} to {self.helsinki_data['departure'].max()}")
        
        return self.london_data, self.helsinki_data, None 

    def get_london_data(self):
        return self.london_data
    
    def get_helsinki_data(self):
        return self.helsinki_data 