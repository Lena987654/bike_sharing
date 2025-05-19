import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_data():
    print("Loading normalized datasets...")
    london_df = pd.read_csv('London_August_2023_normalized.csv')
    helsinki_2019_df = pd.read_csv('Helsinki_August_2019_normalized.csv')
    helsinki_2020_df = pd.read_csv('Helsinki_August_2020_normalized.csv')
    
    # Convert date columns to datetime
    for df in [london_df, helsinki_2019_df, helsinki_2020_df]:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
    
    return london_df, helsinki_2019_df, helsinki_2020_df

def test_duration_hypothesis(london_df, helsinki_df):
    print("\n" + "="*50)
    print("Testing Duration Hypothesis")
    print("="*50)
    
    # Perform Mann-Whitney U test (non-parametric test for independent samples)
    statistic, p_value = stats.mannwhitneyu(london_df['duration_seconds'], 
                                          helsinki_df['duration_seconds'],
                                          alternative='two-sided')
    
    print("\nMann-Whitney U test results:")
    print(f"Statistic: {statistic:,.2f}")
    print(f"P-value: {p_value:.10f}")
    
    # Calculate effect size (Cohen's d)
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    effect_size = cohens_d(london_df['duration_seconds'], helsinki_df['duration_seconds'])
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot duration distributions
    plt.subplot(1, 2, 1)
    data = pd.DataFrame({
        'City': ['London'] * len(london_df) + ['Helsinki'] * len(helsinki_df),
        'Duration': pd.concat([london_df['duration_seconds'], helsinki_df['duration_seconds']])
    })
    sns.boxplot(x='City', y='Duration', data=data)
    plt.title('Duration Distribution Comparison')
    plt.ylabel('Duration (seconds)')
    
    # Plot duration histograms
    plt.subplot(1, 2, 2)
    plt.hist(london_df['duration_seconds'], bins=50, alpha=0.5, label='London')
    plt.hist(helsinki_df['duration_seconds'], bins=50, alpha=0.5, label='Helsinki')
    plt.title('Duration Distribution Histograms')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('duration_comparison.png')
    plt.close()

def test_weekday_patterns(london_df, helsinki_df):
    print("\n" + "="*50)
    print("Testing Weekday Patterns Hypothesis")
    print("="*50)
    
    # Extract day of week
    for df in [london_df, helsinki_df]:
        df['day_of_week'] = df['start_time'].dt.day_name()
    
    # Calculate daily counts
    london_daily = london_df['day_of_week'].value_counts()
    helsinki_daily = helsinki_df['day_of_week'].value_counts()
    
    # Perform chi-square test
    contingency_table = pd.DataFrame({
        'London': london_daily,
        'Helsinki': helsinki_daily
    })
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nChi-square test results:")
    print(f"Chi2 statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.10f}")
    print(f"Degrees of freedom: {dof}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot daily patterns
    plt.subplot(1, 2, 1)
    london_daily.plot(kind='bar')
    plt.title('London Daily Distribution')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Rides')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    helsinki_daily.plot(kind='bar')
    plt.title('Helsinki Daily Distribution')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Rides')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('daily_patterns_comparison.png')
    plt.close()
    
    # Create overlaid visualization
    plt.figure(figsize=(12, 6))
    
    # Calculate percentages for better comparison
    london_percentages = (london_daily / london_daily.sum()) * 100
    helsinki_percentages = (helsinki_daily / helsinki_daily.sum()) * 100
    
    # Plot overlaid daily patterns
    plt.plot(range(len(london_daily)), london_percentages.values, 'b-o', label='London 2023')
    plt.plot(range(len(helsinki_daily)), helsinki_percentages.values, 'r-o', label='Helsinki 2019')
    
    plt.title('Daily Distribution Comparison (% of total rides)')
    plt.xlabel('Day of Week')
    plt.ylabel('Percentage of Total Rides')
    plt.xticks(range(len(london_daily)), london_daily.index, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('daily_distribution_overlay.png')
    plt.close()

def test_hourly_patterns(london_df, helsinki_df):
    print("\n" + "="*50)
    print("Testing Hourly Patterns Hypothesis")
    print("="*50)
    
    # Extract hour
    for df in [london_df, helsinki_df]:
        df['hour'] = df['start_time'].dt.hour
    
    # Calculate hourly counts
    london_hourly = london_df['hour'].value_counts().sort_index()
    helsinki_hourly = helsinki_df['hour'].value_counts().sort_index()
    
    # Perform chi-square test
    contingency_table = pd.DataFrame({
        'London': london_hourly,
        'Helsinki': helsinki_hourly
    })
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nChi-square test results:")
    print(f"Chi2 statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.10f}")
    print(f"Degrees of freedom: {dof}")
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot hourly patterns
    plt.plot(london_hourly.index, london_hourly.values, 'b-o', label='London')
    plt.plot(helsinki_hourly.index, helsinki_hourly.values, 'r-o', label='Helsinki')
    plt.title('Hourly Distribution Comparison')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Rides')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hourly_patterns_comparison.png')
    plt.close()

def analyze_weekend_vs_weekday(london_df, helsinki_df):
    print("\n" + "="*50)
    print("Analyzing Weekend vs Weekday Patterns")
    print("="*50)
    
    for df in [london_df, helsinki_df]:
        df['is_weekend'] = df['start_time'].dt.dayofweek.isin([5, 6])
    
    # Calculate weekend vs weekday statistics
    def get_weekend_stats(df, city):
        weekend = df[df['is_weekend']]
        weekday = df[~df['is_weekend']]
        
        stats_dict = {
            'City': city,
            'Weekend_Count': len(weekend),
            'Weekday_Count': len(weekday),
            'Weekend_Avg_Duration': weekend['duration_seconds'].mean(),
            'Weekday_Avg_Duration': weekday['duration_seconds'].mean()
        }
        return stats_dict
    
    stats_list = [
        get_weekend_stats(london_df, 'London'),
        get_weekend_stats(helsinki_df, 'Helsinki')
    ]
    
    stats_df = pd.DataFrame(stats_list)
    print("\nWeekend vs Weekday Statistics:")
    print(stats_df)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Ride counts
    plt.subplot(1, 2, 1)
    weekend_counts = [stats['Weekend_Count'] for stats in stats_list]
    weekday_counts = [stats['Weekday_Count'] for stats in stats_list]
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, weekday_counts, width, label='Weekday')
    plt.bar(x + width/2, weekend_counts, width, label='Weekend')
    plt.xticks(x, ['London', 'Helsinki'])
    plt.title('Ride Counts: Weekend vs Weekday')
    plt.legend()
    
    # Plot 2: Average duration
    plt.subplot(1, 2, 2)
    weekend_duration = [stats['Weekend_Avg_Duration'] for stats in stats_list]
    weekday_duration = [stats['Weekday_Avg_Duration'] for stats in stats_list]
    
    plt.bar(x - width/2, weekday_duration, width, label='Weekday')
    plt.bar(x + width/2, weekend_duration, width, label='Weekend')
    plt.xticks(x, ['London', 'Helsinki'])
    plt.title('Average Duration: Weekend vs Weekday')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('weekend_weekday_comparison.png')
    plt.close()

def analyze_peak_hours(london_df, helsinki_df):
    print("\n" + "="*50)
    print("Analyzing Peak Hours Patterns")
    print("="*50)
    
    # Define peak hours
    morning_peak = (6, 10)  # 6:00-10:00
    evening_peak = (16, 20)  # 16:00-20:00
    
    def get_peak_stats(df, city):
        df['hour'] = df['start_time'].dt.hour
        df['is_weekday'] = ~df['start_time'].dt.dayofweek.isin([5, 6])
        
        # Filter for weekdays only
        weekday_df = df[df['is_weekday']]
        
        # Morning peak
        morning_peak_rides = weekday_df[
            (weekday_df['hour'] >= morning_peak[0]) & 
            (weekday_df['hour'] < morning_peak[1])
        ]
        
        # Evening peak
        evening_peak_rides = weekday_df[
            (weekday_df['hour'] >= evening_peak[0]) & 
            (weekday_df['hour'] < evening_peak[1])
        ]
        
        # Off-peak
        off_peak_rides = weekday_df[
            ~((weekday_df['hour'] >= morning_peak[0]) & (weekday_df['hour'] < morning_peak[1])) &
            ~((weekday_df['hour'] >= evening_peak[0]) & (weekday_df['hour'] < evening_peak[1]))
        ]
        
        stats_dict = {
            'City': city,
            'Morning_Peak_Count': len(morning_peak_rides),
            'Evening_Peak_Count': len(evening_peak_rides),
            'Off_Peak_Count': len(off_peak_rides),
            'Morning_Peak_Avg_Duration': morning_peak_rides['duration_seconds'].mean(),
            'Evening_Peak_Avg_Duration': evening_peak_rides['duration_seconds'].mean(),
            'Off_Peak_Avg_Duration': off_peak_rides['duration_seconds'].mean()
        }
        return stats_dict
    
    stats_list = [
        get_peak_stats(london_df, 'London'),
        get_peak_stats(helsinki_df, 'Helsinki')
    ]
    
    stats_df = pd.DataFrame(stats_list)
    print("\nPeak Hours Statistics (Weekdays Only):")
    print(stats_df)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Ride counts during different periods
    plt.subplot(1, 2, 1)
    x = np.arange(2)
    width = 0.25
    
    morning_counts = [stats['Morning_Peak_Count'] for stats in stats_list]
    evening_counts = [stats['Evening_Peak_Count'] for stats in stats_list]
    off_peak_counts = [stats['Off_Peak_Count'] for stats in stats_list]
    
    plt.bar(x - width, morning_counts, width, label='Morning Peak')
    plt.bar(x, evening_counts, width, label='Evening Peak')
    plt.bar(x + width, off_peak_counts, width, label='Off Peak')
    plt.xticks(x, ['London', 'Helsinki'])
    plt.title('Ride Counts by Period (Weekdays)')
    plt.legend()
    
    # Plot 2: Average duration during different periods
    plt.subplot(1, 2, 2)
    morning_duration = [stats['Morning_Peak_Avg_Duration'] for stats in stats_list]
    evening_duration = [stats['Evening_Peak_Avg_Duration'] for stats in stats_list]
    off_peak_duration = [stats['Off_Peak_Avg_Duration'] for stats in stats_list]
    
    plt.bar(x - width, morning_duration, width, label='Morning Peak')
    plt.bar(x, evening_duration, width, label='Evening Peak')
    plt.bar(x + width, off_peak_duration, width, label='Off Peak')
    plt.xticks(x, ['London', 'Helsinki'])
    plt.title('Average Duration by Period (Weekdays)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('peak_hours_comparison.png')
    plt.close()

def compare_helsinki_years(helsinki_2019_df, helsinki_2020_df):
    print("\n" + "="*50)
    print("Comparing Helsinki Data: 2019 vs 2020")
    print("="*50)
    
    # Test duration difference
    statistic, p_value = stats.mannwhitneyu(helsinki_2019_df['duration_seconds'], 
                                          helsinki_2020_df['duration_seconds'],
                                          alternative='two-sided')
    
    print("\nDuration Comparison (Mann-Whitney U test):")
    print(f"Statistic: {statistic:,.2f}")
    print(f"P-value: {p_value:.10f}")
    
    # Calculate effect size
    def cohens_d(group1, group2):
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_se
    
    effect_size = cohens_d(helsinki_2019_df['duration_seconds'], helsinki_2020_df['duration_seconds'])
    print(f"Effect size (Cohen's d): {effect_size:.3f}")
    
    # Compare daily patterns
    for df in [helsinki_2019_df, helsinki_2020_df]:
        df['day_of_week'] = df['start_time'].dt.day_name()
    
    helsinki_2019_daily = helsinki_2019_df['day_of_week'].value_counts()
    helsinki_2020_daily = helsinki_2020_df['day_of_week'].value_counts()
    
    # Perform chi-square test for daily patterns
    contingency_table = pd.DataFrame({
        '2019': helsinki_2019_daily,
        '2020': helsinki_2020_daily
    })
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nDaily Patterns Comparison (Chi-square test):")
    print(f"Chi2 statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.10f}")
    print(f"Degrees of freedom: {dof}")
    
    # Compare hourly patterns
    for df in [helsinki_2019_df, helsinki_2020_df]:
        df['hour'] = df['start_time'].dt.hour
    
    helsinki_2019_hourly = helsinki_2019_df['hour'].value_counts().sort_index()
    helsinki_2020_hourly = helsinki_2020_df['hour'].value_counts().sort_index()
    
    # Perform chi-square test for hourly patterns
    contingency_table = pd.DataFrame({
        '2019': helsinki_2019_hourly,
        '2020': helsinki_2020_hourly
    })
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("\nHourly Patterns Comparison (Chi-square test):")
    print(f"Chi2 statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.10f}")
    print(f"Degrees of freedom: {dof}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Duration distributions
    plt.subplot(2, 2, 1)
    data = pd.DataFrame({
        'Year': ['2019'] * len(helsinki_2019_df) + ['2020'] * len(helsinki_2020_df),
        'Duration': pd.concat([helsinki_2019_df['duration_seconds'], helsinki_2020_df['duration_seconds']])
    })
    sns.boxplot(x='Year', y='Duration', data=data)
    plt.title('Duration Distribution Comparison')
    plt.ylabel('Duration (seconds)')
    
    # Plot 2: Daily patterns
    plt.subplot(2, 2, 2)
    helsinki_2019_daily.plot(kind='bar', label='2019')
    helsinki_2020_daily.plot(kind='bar', label='2020')
    plt.title('Daily Distribution Comparison')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Rides')
    plt.xticks(rotation=45)
    plt.legend()
    
    # Plot 3: Hourly patterns
    plt.subplot(2, 2, 3)
    plt.plot(helsinki_2019_hourly.index, helsinki_2019_hourly.values, 'b-o', label='2019')
    plt.plot(helsinki_2020_hourly.index, helsinki_2020_hourly.values, 'r-o', label='2020')
    plt.title('Hourly Distribution Comparison')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Rides')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 4: Weekend vs Weekday comparison
    plt.subplot(2, 2, 4)
    for df, year in [(helsinki_2019_df, '2019'), (helsinki_2020_df, '2020')]:
        df['is_weekend'] = df['start_time'].dt.dayofweek.isin([5, 6])
        weekend = df[df['is_weekend']]
        weekday = df[~df['is_weekend']]
        
        x = np.arange(2)
        width = 0.35
        offset = 0.35 if year == '2020' else 0
        
        plt.bar(x[0] + offset, len(weekday), width, label=f'{year} Weekday')
        plt.bar(x[1] + offset, len(weekend), width, label=f'{year} Weekend')
    
    plt.xticks(x + width/2, ['Weekday', 'Weekend'])
    plt.title('Weekend vs Weekday Comparison')
    plt.ylabel('Number of Rides')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('helsinki_years_comparison.png')
    plt.close()

def visualize_patterns(london_df, helsinki_df):
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparison of Bike Usage Patterns in London and Helsinki', fontsize=16)

    # 1. Daily distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    london_daily = london_df['day_of_week'].value_counts().reindex(day_order)
    helsinki_daily = helsinki_df['day_of_week'].value_counts().reindex(day_order)

    x = np.arange(len(day_order))
    width = 0.35

    axes[0, 0].bar(x - width/2, london_daily, width, label='London', color='skyblue')
    axes[0, 0].bar(x + width/2, helsinki_daily, width, label='Helsinki', color='lightcoral')
    axes[0, 0].set_title('Distribution of Rides by Day of Week')
    axes[0, 0].set_xlabel('Day of Week')
    axes[0, 0].set_ylabel('Number of Rides')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(day_order, rotation=45)
    axes[0, 0].legend()

    # 2. Hourly distribution
    london_hourly = london_df['hour'].value_counts().sort_index()
    helsinki_hourly = helsinki_df['hour'].value_counts().sort_index()

    axes[0, 1].plot(london_hourly.index, london_hourly.values, label='London', color='skyblue')
    axes[0, 1].plot(helsinki_hourly.index, helsinki_hourly.values, label='Helsinki', color='lightcoral')
    axes[0, 1].set_title('Distribution of Rides by Hour')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Number of Rides')
    axes[0, 1].legend()

    # 3. Trip duration distribution
    axes[1, 0].hist(london_df['duration_seconds'], bins=50, alpha=0.5, label='London', color='skyblue')
    axes[1, 0].hist(helsinki_df['duration_seconds'], bins=50, alpha=0.5, label='Helsinki', color='lightcoral')
    axes[1, 0].set_title('Distribution of Trip Duration')
    axes[1, 0].set_xlabel('Duration (seconds)')
    axes[1, 0].set_ylabel('Number of Rides')
    axes[1, 0].legend()

    # 4. Weekend vs Weekday comparison
    london_weekend = london_df['is_weekend'].value_counts()
    helsinki_weekend = helsinki_df['is_weekend'].value_counts()

    x = np.arange(2)
    axes[1, 1].bar(x - width/2, [london_weekend[False], london_weekend[True]], width, label='London', color='skyblue')
    axes[1, 1].bar(x + width/2, [helsinki_weekend[False], helsinki_weekend[True]], width, label='Helsinki', color='lightcoral')
    axes[1, 1].set_title('Weekend vs Weekday Comparison')
    axes[1, 1].set_xlabel('Day Type')
    axes[1, 1].set_ylabel('Number of Rides')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Weekday', 'Weekend'])
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('hypothesis_patterns.png')
    plt.close()

def main():
    # Load data
    london_df, helsinki_2019_df, helsinki_2020_df = load_data()
    
    # Test hypotheses
    test_duration_hypothesis(london_df, helsinki_2019_df)
    test_weekday_patterns(london_df, helsinki_2019_df)
    test_hourly_patterns(london_df, helsinki_2019_df)
    
    # Additional analyses
    analyze_weekend_vs_weekday(london_df, helsinki_2019_df)
    analyze_peak_hours(london_df, helsinki_2019_df)
    
    # Compare Helsinki years
    compare_helsinki_years(helsinki_2019_df, helsinki_2020_df)
    
    # Visualize patterns
    visualize_patterns(london_df, helsinki_2019_df)
    
    print("\nAnalysis completed. Check the generated visualizations for detailed patterns.")

if __name__ == "__main__":
    main() 