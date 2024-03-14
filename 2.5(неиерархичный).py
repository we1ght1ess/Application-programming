import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Создание DataFrame с данными о холодильниках
data = {
    'Название продукта': ['Холодильник 1', 'Холодильник 2', 'Холодильник 3', 'Холодильник 4', 'Холодильник 5'],
    'Бренд': ['Samsung', 'LG', 'Bosch', 'Whirlpool', 'Electrolux'],
    'Модель': ['RT32K4000S8', 'GA-B459SLGL', 'KGN39VWEA', 'WRB322DMBM', 'EN3851AOX'],
    'Год выпуска': [2019, 2020, 2018, 2021, 2017],
    'Средняя цена': [25000, 27000, 30000, 22000, 28000],
    'Мощность': ['150 Вт', '180 Вт', '200 Вт', '170 Вт', '160 Вт'],
    'Объем, литры': [320, 460, 350, 400, 380],
    'Размеры (ШxВxГ), см': ['60x170x70', '70x180x75', '65x185x72', '65x175x68', '63x175x70'],
    'Описание': ['Двухкамерный холодильник с системой No Frost.', 
                 'Холодильник с линейкой Smart Inverter и системой Multi Air Flow.', 
                 'Трехкамерный холодильник с технологией VitaFresh.', 
                 'Холодильник с системой температурного контроля Accu-Chill.', 
                 'Холодильник с технологией IonAir для сохранения свежести продуктов.']
}

df = pd.DataFrame(data)

X = df[['Средняя цена', 'Объем, литры']]

# Стандартизация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Создание модели K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Добавление меток кластеров в DataFrame
df['Cluster'] = kmeans.labels_

print(df)

segment_map = {0: 'Средний', 1: 'Дорогой', 2: 'Бюджетный'}
# Визуализация результатов кластеризации методом k-средних
plt.figure(figsize=(8, 6))

# Построение точек для каждого сегмента
for cluster_id, segment_name in segment_map.items():
    cluster_data = X[kmeans.labels_ == cluster_id]
    plt.scatter(cluster_data['Средняя цена'], cluster_data['Объем, литры'], label=segment_name, alpha=0.8, s=50)

# Добавление названий холодильников
for i, txt in enumerate(df['Название продукта']):
    plt.annotate(txt, (X['Средняя цена'][i], X['Объем, литры'][i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('Средняя цена')
plt.ylabel('Объем, литры')
plt.title('Неиерархическая кластеризация')
plt.legend()
plt.grid(True)
plt.show()