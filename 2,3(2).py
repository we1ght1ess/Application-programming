import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, r2_score

# Функции для аппроксимации
def logarithmic(x, a, b):
    return a * np.log(x) + b

def hyperbolic(x, a, b):
    return a / x + b

def power(x, a, b):
    return a * np.power(x, b)

def exponential(x, a, b):
    return a * np.exp(b * x)

# Загрузка данных
data = pd.read_excel('Файл_1.xlsx')

# Выбор независимой переменной (X) и зависимой переменной (Y)
X = data['X']
y = data['Y']

# Аппроксимация данных с помощью логарифмической регрессии
params_logarithmic, _ = curve_fit(logarithmic, X, y)
y_logarithmic = logarithmic(X, *params_logarithmic)
mse_logarithmic = mean_squared_error(y, y_logarithmic)

# Аппроксимация данных с помощью гиперболической регрессии
params_hyperbolic, _ = curve_fit(hyperbolic, X, y)
y_hyperbolic = hyperbolic(X, *params_hyperbolic)
mse_hyperbolic = mean_squared_error(y, y_hyperbolic)

# Аппроксимация данных с помощью степенной регрессии
params_power, _ = curve_fit(power, X, y)
y_power = power(X, *params_power)
mse_power = mean_squared_error(y, y_power)

# Аппроксимация данных с помощью показательной регрессии
params_exponential, _ = curve_fit(exponential, X, y)
y_exponential = exponential(X, *params_exponential)
mse_exponential = mean_squared_error(y, y_exponential)

# Выбор лучшей модели по средней ошибке аппроксимации
errors = {
    'Логарифмическая': mse_logarithmic,
    'Гиперболическая': mse_hyperbolic,
    'Степенная': mse_power,
    'Показательная': mse_exponential
}

best_model = min(errors, key=errors.get)
best_params = {
    'Логарифмическая': params_logarithmic,
    'Гиперболическая': params_hyperbolic,
    'Степенная': params_power,
    'Показательная': params_exponential
}

# Вычисление коэффициента детерминации для логарифмической регрессии
r2_logarithmic = r2_score(y, y_logarithmic)

# Вычисление коэффициента детерминации для гиперболической регрессии
r2_hyperbolic = r2_score(y, y_hyperbolic)

# Вычисление коэффициента детерминации для степенной регрессии
r2_power = r2_score(y, y_power)

# Вычисление коэффициента детерминации для показательной регрессии
r2_exponential = r2_score(y, y_exponential)

print("Коэффициент детерминации для логарифмической регрессии:", r2_logarithmic)
print("Коэффициент детерминации для гиперболической регрессии:", r2_hyperbolic)
print("Коэффициент детерминации для степенной регрессии:", r2_power)
print("Коэффициент детерминации для показательной регрессии:", r2_exponential)

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Исходные данные')
plt.plot(X, y_logarithmic, color='red', linestyle='--', label='Логарифмическая регрессия')
plt.plot(X, y_hyperbolic, color='green', linestyle='--', label='Гиперболическая регрессия')
plt.plot(X, y_power, color='blue', linestyle='--', label='Степенная регрессия')
plt.plot(X, y_exponential, color='orange', linestyle='--', label='Показательная регрессия')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Визуализация регрессий')
plt.legend()
plt.grid(True)
plt.show()