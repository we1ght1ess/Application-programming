import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных из Excel-файла
data = pd.read_excel('Преступления.xlsx')  

# Разделение данных на признаки (X) и целевую переменную (y)
X = data[['Убийства', 'Покушения', 'Gini']]
y = data['Регион']  

y = y.apply(lambda x: 1 if x == 'Криминальный' else 0)
# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели случайного леса
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Предсказание классов на тестовом наборе
y_pred = clf.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print("Точность модели:", accuracy)

# Визуализация разделения регионов на классы
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Убийства', y='Покушения',hue='Регион', data=data, palette='Set2', alpha=0.5)
plt.title('Разделение регионов на классы')
plt.xlabel('Убийства')
plt.ylabel('Покушения')
plt.legend(title='Регион')
plt.show()