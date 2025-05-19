from data_loader import DataLoader
from daily_patterns_analyzer import DailyPatternsAnalyzer

def main():
    # Load data
    loader = DataLoader()
    london_data, helsinki_data, _ = loader.load_all_data()
    
    # Analyze daily patterns
    patterns_analyzer = DailyPatternsAnalyzer(london_data, helsinki_data)
    
    # Analyze daily distribution
    london_daily, helsinki_daily = patterns_analyzer.analyze_daily_distribution()
    
    # Analyze weekday vs weekend usage
    weekday_weekend_stats = patterns_analyzer.analyze_weekday_vs_weekend()
    
    # Perform chi-square test
    chi_square_results = patterns_analyzer.perform_chi_square_test()
    
    # Get basic statistics
    basic_stats = patterns_analyzer.get_basic_stats()
    
    # Print results
    print("\nDaily Patterns Analysis:")
    print("\nLondon Statistics:")
    for key, value in basic_stats['London'].items():
        print(f"{key}: {value}")
        
    print("\nHelsinki Statistics:")
    for key, value in basic_stats['Helsinki'].items():
        print(f"{key}: {value}")
        
    print("\nWeekday vs Weekend Usage:")
    for city, stats in weekday_weekend_stats.items():
        print(f"\n{city}:")
        print(f"Weekday rides: {stats['weekday']}")
        print(f"Weekend rides: {stats['weekend']}")
        print(f"Weekend percentage: {(stats['weekend'] / (stats['weekday'] + stats['weekend']) * 100):.2f}%")
        
    print("\nChi-square Test Results:")
    print(f"Chi-square statistic: {chi_square_results['chi2']:.2f}")
    print(f"Degrees of freedom: {chi_square_results['dof']}")
    print(f"p-value: {chi_square_results['p_value']:.4f}")
    
    # Interpret p-value
    alpha = 0.05
    if chi_square_results['p_value'] < alpha:
        print("\nReject H0: There are significant differences in daily patterns between cities")
    else:
        print("\nFail to reject H0: No significant differences in daily patterns between cities")

if __name__ == "__main__":
    main() 