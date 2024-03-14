import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Загрузка данных из Excel-файла
data = pd.read_excel('Преступления.xlsx') 

# Выбор признаков для кластеризации
X = data[['Убийства', 'Покушения', 'Gini']]

# Инициализация и обучение модели K-means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)

# Получение меток кластеров
labels = kmeans.labels_

# Добавление меток кластеров в DataFrame
data['cluster'] = labels

#Создание словаря для замены меток кластеров
cluster_labels = {0: 'Не криминальные', 1: 'Криминальные'}

# Замена числовых меток на соответствующие значения
data['cluster'] = data['cluster'].map(cluster_labels)

# Визуализация результата
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Убийства', y='Покушения', hue='cluster', data=data, palette='viridis')
for i in range(len(data)):
    plt.text(data['Убийства'][i], data['Покушения'][i], data['Регион'][i], fontsize=8)
plt.xlabel('Число убийств')
plt.ylabel('Число покушений на убийство')
plt.title('Кластеризация регионов')
plt.legend(title='Классы')
plt.show()
