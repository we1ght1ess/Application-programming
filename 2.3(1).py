import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных из файла Excel
data = pd.read_excel('Файл_1.xlsx')

# Разделение данных на признаки (X) и целевую переменную (y)
X = data[['X']]
y = data['Y']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели линейной регрессии
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = model.predict(X_test)

# Оценка качества модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Среднеквадратичная ошибка (MSE):", mse)
print("Коэффициент детерминации (R^2):", r2)

# Вывод коэффициентов регрессии
print("Уравнение регрессии:")
print("Y =", model.coef_[0], "* X +", model.intercept_)

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Тестовые данные')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Линейная регрессия')
plt.title('Линейная регрессия')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()