import pandas as pd 

data = pd.read_csv("data_week2.csv", encoding='cp949') 

cols = ['num', 'date_time', 'power', 'temp', 'wind','hum' ,'rain', 'sun', 'non_elec', 'solar']
data.columns = cols

# 시간 관련 변수들 생성
date = pd.to_datetime(data.date_time)
# data['year'] = data.dt.year
data['month'] = date.dt.month
data['day'] = date.dt.day
data['hour'] = date.dt.hour
data['week'] = date.dt.weekday

# row['week']가 5(토요일) 또는 6(일요일)인 경우, 즉 주말이면 1을 반환
def is_holiday(row):
    if row['week'] in [5, 6]:
        return 1
    elif (row['month'] == 6 and row['day'] == 6) or \
         (row['month'] == 8 and row['day'] in [15, 17]):
        return 1
    else:
        return 0


data['holiday'] = data.apply(is_holiday, axis=1)

print(data.columns)
print(data.shape)


import matplotlib.pyplot as plt
import seaborn as sns
# 시각화(x축은 전력사용량, y축은 전력 사용량 구간에 해당하는 데이터의 빈도.)
plt.figure(figsize=(10,6))
sns.histplot(data=data, x='power')

import matplotlib.dates as mdates
# 시간변화에 따른 전력사용량 분석(x축은 날짜, y축은 전력사용량)

data['date_time']= pd.to_datetime(data['date_time'])


plt.figure(figsize=(12, 4))
sns.lineplot(data=data, x='date_time', y='power', errorbar=None)

plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.xticks(rotation=45)
plt.show()


# 건물 번호 별로 전력 사용량 3개월 추세 확인
for bn in data['num'].unique():
  plt.figure(figsize=(12, 2))
  plt.title(f'Building {bn}')
  b= data[data['num']== bn]
  sns.lineplot(data=b, x='date_time', y='power', errorbar=None)


# dtw 해보자
!pip install dtw-python
!pip install dtaidistance

from dtw import *  # dtw-python 라이브러리
import numpy as np

results = []

# 건물별로 전력 사용량 비교
for domestic_building in data['num'].unique():
   
    domestic_power_data = data[data['num'] == domestic_building]['power'].values

    # 각 건물 간의 유사도 계산
    for international_building in data['num'].unique():
        if domestic_building != international_building:  # 자기 자신과 비교하지 않음
            # 비교 대상 건물의 전력 사용량 데이터
            international_power_data = data[data['num'] == international_building]['power'].values

            alignment = dtw(domestic_power_data, international_power_data, keep_internals=True)
            dtw_distance = alignment.distance

            # 결과 저장
            results.append({
                'domestic_building': domestic_building,
                'international_building': international_building,
                'metric': 'power_usage',
                'distance': dtw_distance
            })

# 결과 출력
for result in results:
    print(result)



## 거리 계산한 결과 기반 kmeans 군집화
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_results = pd.DataFrame(results)


distance_matrix = df_results.pivot(index='domestic_building', columns='international_building', values='distance')


distance_matrix = distance_matrix.fillna(0)


kmeans = KMeans(n_clusters=3, random_state=0).fit(distance_matrix)


building_clusters = pd.DataFrame({
    'domestic_building': distance_matrix.index,
    'cluster': kmeans.labels_
})


df_results = df_results.merge(building_clusters, on='domestic_building')


print(df_results[['domestic_building', 'international_building', 'distance', 'cluster']])


# 군집화 제대로 됐는지 확인해보기
unique_clusters = df_results['cluster'].unique()
print("Unique clusters:", unique_clusters)



## 임의로 n_clusters 지정했는데, 데이터에 맞는 클러스트 개수 정해보기.\
inertia_values = []

for n in range(1, 10):
    kmeans = KMeans(n_clusters=n, random_state=0).fit(distance_matrix)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), inertia_values, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()



# 클러스터 별 건물 번호 확인하기

clustered_buildings = df_results[['domestic_building', 'cluster']].drop_duplicates().groupby('cluster')['domestic_building'].apply(list)

print(clustered_buildings)
len(clustered_buildings[0])
len(clustered_buildings[1])
len(clustered_buildings[2])


# DTW 거리 분포 확인
plt.figure(figsize=(10, 6))
sns.histplot(df_results['distance'], bins=30, kde=True)
plt.title('Distribution of DTW Distances')
plt.xlabel('DTW Distance')
plt.ylabel('Frequency')
plt.show()

### k-거리 그래프는 각 데이터 포인트에 대해 k번째 가까운 이웃과의 거리를 계산
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
# k-NN 알고리즘을 통해 각 데이터 포인트의 거리를 계산 (k=5 사용)

neighbors = NearestNeighbors(n_neighbors=5, metric='precomputed')
neighbors_fit = neighbors.fit(distance_matrix)
distances, indices = neighbors_fit.kneighbors(distance_matrix)

# k-거리 (5번째 이웃까지의 거리) 오름차순 정렬
distances = np.sort(distances[:, 4], axis=0)

# k-거리 그래프 시각화
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title('k-distance graph (for k=5)')
plt.xlabel('Data points (sorted)')
plt.ylabel('5th Nearest Neighbor Distance')
plt.grid(True)
plt.show()


###  kmeans 가 적절하지 않음 - > 밀도 기반 클러스터링 DBSCAN

from sklearn.cluster import DBSCAN

# 4. DBSCAN 클러스터링 적용
# eps와 min_samples 값을 조정하여 최적의 클러스터링을 찾습니다.
dbscan = DBSCAN(eps=0.7e6, min_samples=2, metric='precomputed').fit(distance_matrix)

# 5. 클러스터 라벨을 각 건물에 추가
building_clusters = pd.DataFrame({
    'domestic_building': distance_matrix.index,
    'cluster': dbscan.labels_
})

# 6. 병합하여 각 건물의 클러스터 확인
df_results2 = df_results.merge(building_clusters, on='domestic_building', how='left')

# 7. 클러스터 결과 출력
# cluster_y를 cluster로 이름 변경하고 cluster_x는 제거
df_results2 = df_results2.drop(columns=['cluster_x']).rename(columns={'cluster_y': 'cluster'})

# 클러스터 결과 출력
print(df_results2[['domestic_building', 'international_building', 'distance', 'cluster']])

df_results2['cluster'].unique()
(df_results2['cluster']== -1).sum()
(df_results2['cluster']== 0).sum()
(df_results2['cluster']== 1).sum()


# 각 클러스터의 데이터 수 확인
print(df_results2['cluster'].value_counts())

# PCA로 클러스터 시각화
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(distance_matrix)

df_pca = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
df_pca['cluster'] = dbscan.labels_

# 클러스터 시각화
plt.figure(figsize=(10, 7))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='Set1', s=100, marker='o')
plt.title('DBSCAN Clustering of Buildings Based on DTW Distances')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster', loc='upper right')
plt.grid(True)
plt.show()
