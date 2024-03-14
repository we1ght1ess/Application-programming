import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

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


# Выбираем признаки для кластеризации
X = df[['Средняя цена', 'Объем, литры']]

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Вычисляем матрицу расстояний и строим дендрограмму
linkage_matrix = linkage(X_scaled, method='ward')

# Построение дендрограммы
plt.figure(figsize=(10, 5))
dendrogram(linkage_matrix, labels=df['Название продукта'].values, leaf_rotation=90)
plt.title('Иерархическая кластеризация')
plt.xlabel('Холодильники')
plt.ylabel('Расстояние между средними значениями')
plt.show()

# Создаем модель иерархической кластеризации
cluster = AgglomerativeClustering(n_clusters=2)

# Проводим кластеризацию
df['Cluster'] = cluster.fit_predict(X_scaled)

print(df)