import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Настройка стиля графиков
plt.style.use('seaborn')
sns.set_palette("husl")

# Читаем данные
print("Загрузка данных...")
london_data = pd.read_csv('LondonBikeJourneyAug2023.csv')
helsinki_data = pd.read_csv('Helsinki_1M.csv')

# Преобразуем временные столбцы
london_data['Start date'] = pd.to_datetime(london_data['Start date'])
london_data['End date'] = pd.to_datetime(london_data['End date'])

helsinki_data['departure'] = pd.to_datetime(helsinki_data['departure'])
helsinki_data['return'] = pd.to_datetime(helsinki_data['return'])

print("\nРазмер датасетов:")
print(f"Лондон: {london_data.shape}")
print(f"Хельсинки: {helsinki_data.shape}")

# 1. Анализ распределения поездок по часам
print("\n1. Анализ распределения поездок по часам")
london_data['hour'] = london_data['Start date'].dt.hour
helsinki_data['hour'] = helsinki_data['departure'].dt.hour

plt.figure(figsize=(12, 6))
london_hourly = london_data['hour'].value_counts().sort_index()
helsinki_hourly = helsinki_data['hour'].value_counts().sort_index()

# Нормализуем данные для сравнения
london_hourly_norm = london_hourly / london_hourly.max() * 100
helsinki_hourly_norm = helsinki_hourly / helsinki_hourly.max() * 100

plt.plot(london_hourly_norm.index, london_hourly_norm.values, label='Лондон', marker='o')
plt.plot(helsinki_hourly_norm.index, helsinki_hourly_norm.values, label='Хельсинки', marker='o')

plt.title('Распределение поездок по часам (нормализованные данные)')
plt.xlabel('Час дня')
plt.ylabel('Процент от максимального количества поездок')
plt.legend()
plt.grid(True)
plt.savefig('hourly_distribution.png')
plt.close()

print(f"\nПиковые часы:")
print(f"Лондон: {london_hourly.idxmax()}:00 ({london_hourly.max()} поездок)")
print(f"Хельсинки: {helsinki_hourly.idxmax()}:00 ({helsinki_hourly.max()} поездок)")

# 2. Анализ распределения поездок по дням недели
print("\n2. Анализ распределения поездок по дням недели")
london_data['day_of_week'] = london_data['Start date'].dt.day_name()
helsinki_data['day_of_week'] = helsinki_data['departure'].dt.day_name()

plt.figure(figsize=(12, 6))
london_daily = london_data['day_of_week'].value_counts()
helsinki_daily = helsinki_data['day_of_week'].value_counts()

days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
london_daily = london_daily.reindex(days_order)
helsinki_daily = helsinki_daily.reindex(days_order)

# Нормализуем данные
london_daily_norm = london_daily / london_daily.max() * 100
helsinki_daily_norm = helsinki_daily / helsinki_daily.max() * 100

x = np.arange(len(days_order))
width = 0.35

plt.bar(x - width/2, london_daily_norm.values, width, label='Лондон')
plt.bar(x + width/2, helsinki_daily_norm.values, width, label='Хельсинки')

plt.title('Распределение поездок по дням недели (нормализованные данные)')
plt.xlabel('День недели')
plt.ylabel('Процент от максимального количества поездок')
plt.xticks(x, ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'], rotation=45)
plt.legend()
plt.grid(True)
plt.savefig('daily_distribution.png')
plt.close()

print(f"\nСамые популярные дни:")
print(f"Лондон: {london_daily.idxmax()} ({london_daily.max()} поездок)")
print(f"Хельсинки: {helsinki_daily.idxmax()} ({helsinki_daily.max()} поездок)")

# 3. Анализ длительности поездок
print("\n3. Анализ длительности поездок")
plt.figure(figsize=(12, 6))
london_duration = london_data.groupby('hour')['Total duration (ms)'].mean() / 1000
helsinki_duration = helsinki_data.groupby('hour')['duration (sec.)'].mean()

plt.plot(london_duration.index, london_duration.values, label='Лондон', marker='o')
plt.plot(helsinki_duration.index, helsinki_duration.values, label='Хельсинки', marker='o')

plt.title('Средняя длительность поездок по часам')
plt.xlabel('Час дня')
plt.ylabel('Средняя длительность (секунды)')
plt.legend()
plt.grid(True)
plt.savefig('duration_analysis.png')
plt.close()

print(f"\nСредняя длительность поездок:")
print(f"Лондон: {london_data['Total duration (ms)'].mean() / 1000:.2f} секунд")
print(f"Хельсинки: {helsinki_data['duration (sec.)'].mean():.2f} секунд")

# 4. Влияние температуры на использование велосипедов (Хельсинки)
print("\n4. Анализ влияния температуры (Хельсинки)")
plt.figure(figsize=(12, 6))
helsinki_temp = helsinki_data.groupby('Air temperature (degC)').size()

plt.scatter(helsinki_temp.index, helsinki_temp.values, alpha=0.5)
plt.title('Количество поездок в зависимости от температуры воздуха')
plt.xlabel('Температура (°C)')
plt.ylabel('Количество поездок')
plt.grid(True)
plt.savefig('temperature_analysis.png')
plt.close()

print(f"\nСтатистика по температуре:")
print(f"Средняя температура: {helsinki_data['Air temperature (degC)'].mean():.2f}°C")
print(f"Минимальная температура: {helsinki_data['Air temperature (degC)'].min():.2f}°C")
print(f"Максимальная температура: {helsinki_data['Air temperature (degC)'].max():.2f}°C") 