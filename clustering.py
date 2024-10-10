
# dtw 해보자
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

colors = ['blue', 'green', 'pink']
plt.figure(figsize=(10, 6))
df_results.groupby('cluster')['distance'].count().plot(kind='bar', color=[colors[i] for i in range(3)])

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
dbscan = DBSCAN(eps=1719346.4, min_samples=2, metric='precomputed').fit(distance_matrix)

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


colors = ['skyblue', 'lightgreen', 'salmon']
plt.figure(figsize=(10, 6))
df_results2.groupby('cluster')['distance'].count().plot(kind='bar', color=[colors[i] for i in range(3)])



## 적절한 eps값
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# DBSCAN을 사용하기 위한 eps 찾기
nearest_neighbors = NearestNeighbors(n_neighbors=2)
nearest_neighbors.fit(distance_matrix)
distances, indices = nearest_neighbors.kneighbors(distance_matrix)

# 거리 오름차순 정렬 후 그래프 그리기
distances = np.sort(distances[:, 1], axis=0)
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel('Epsilon value (Distance to nearest neighbors)')
plt.title('k-distance Graph for Epsilon Selection')
plt.show()

# 거리행렬
median_distance = np.median(distance_matrix.values)
print(f"Median distance: {median_distance}")

