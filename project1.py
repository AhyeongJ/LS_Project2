import pandas as pd 

data = pd.read_csv("data_week2.csv", encoding='cp949') 

cols = ['num', 'date_time', 'power', 'temp', 'wind','hum' ,'rain', 'sun', 'cooler', 'solar']
data.columns = cols

print(data.columns)
print(data.shape)

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

fig = plt.figure(figsize = (15, 40))
for num in range(1,61):
    ax = plt.subplot(12, 5, num)
    energy = data.loc[data.num == num, 'power'].values
    mean = energy.mean().round(3)
    std = energy.std().round(3)
    skew = (3*(mean - np.median(energy))/energy.std()).round(3)
    if skew >= 1.5:
        plt.hist(energy, alpha = 0.7, bins = 50, color = 'red')
    elif skew <= -1.5:
        plt.hist(energy, alpha = 0.7, bins = 50, color = 'blue')
    else:
        plt.hist(energy, alpha = 0.7, bins = 50, color = 'gray')
    plt.title(f'building{num}')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.text(0.35, 0.9, f'mean : {mean}',  ha='left', va='center', transform=ax.transAxes)
    plt.text(0.35, 0.8, f'std : {std}',  ha='left', va='center', transform=ax.transAxes)
    plt.text(0.35, 0.7, f'skew : {skew}',  ha='left', va='center', transform=ax.transAxes)

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
#!pip install dtw-python
#!pip install dtaidistance

from dtw import *  # dtw-python 라이브러리
import numpy as np

# DTW 거리 계산
results = []
buildings = data['num'].unique()

# DTW 계산 최적화: 메모리와 시간 절약을 위해 결과를 캐싱
dtw_cache = {}
for domestic_building in buildings:
    domestic_power_data = data[data['num'] == domestic_building]['power'].values
    
    for international_building in buildings:
        if domestic_building != international_building:
            if (international_building, domestic_building) in dtw_cache:
                dtw_distance = dtw_cache[(international_building, domestic_building)]
            else:
                international_power_data = data[data['num'] == international_building]['power'].values
                alignment = dtw(domestic_power_data, international_power_data, keep_internals=True)
                dtw_distance = alignment.distance
                dtw_cache[(domestic_building, international_building)] = dtw_distance

            results.append({
                'domestic_building': domestic_building,
                'international_building': international_building,
                'metric': 'power_usage',
                'distance': dtw_distance
            })

# DataFrame으로 변환
df_results = pd.DataFrame(results)

# 거리 행렬 생성
distance_matrix = df_results.pivot(index='domestic_building', columns='international_building', values='distance').fillna(0)
distance_matrix

## 거리 계산한 결과 기반 kmeans 군집화
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 최적의 클러스터 수 결정
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


# KMeans 클러스터링
optimal_clusters = 3  # 엘보우 메서드를 통해 결정된 클러스터 수
kmeans = KMeans(n_clusters=optimal_clusters, random_state=0).fit(distance_matrix)
building_clusters = pd.DataFrame({'domestic_building': distance_matrix.index, 'cluster': kmeans.labels_})
df_results = df_results.merge(building_clusters, on='domestic_building')

# KMeans 결과 시각화
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(distance_matrix)

plt.figure(figsize=(10, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.title('KMeans Clustering Results')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

# DBSCAN 클러스터링
dbscan = DBSCAN(eps=0.7e6, min_samples=2, metric='precomputed').fit(distance_matrix) #np.log(60)
building_clusters_dbscan = pd.DataFrame({'domestic_building': distance_matrix.index, 'cluster': dbscan.labels_})

# 클러스터 결과 병합, 접미사 추가
df_results_dbscan = df_results.merge(building_clusters_dbscan, on='domestic_building', how='left', suffixes=('', '_dbscan'))

# 클러스터 결과 출력
print(df_results_dbscan[['domestic_building', 'international_building', 'distance', 'cluster_dbscan']])

# 클러스터 결과 시각화
plt.figure(figsize=(10, 6))
sns.countplot(data=df_results_dbscan, x='cluster_dbscan', palette='viridis')
plt.title('Cluster Sizes After DBSCAN')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.show()



import numpy as np
import pandas as pd
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.utils import AirPassengersDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 데이터 로드
data['date_time'] = pd.to_datetime(data['date_time'])
data.set_index('date_time', inplace=True)

# 종속변수 선택
series = data['power'].values  # 예측할 종속변수

# 데이터 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
# 예를 들어, data는 여러 열을 가진 데이터프레임이라고 가정
data = data[['date_time', 'power', 'temp', 'wind', 'hum']]  # 원하는 열만 선택
scaled_data = scaler.fit_transform(data)
scaled_data = scaler.fit_transform(series.reshape(-1, 1))

# 데이터셋 생성
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])  # 모든 열을 포함
        y.append(data[i + time_step, 0])  # 종속 변수(y)만
    return np.array(X), np.array(y)



# time_step 설정 및 X, y 생성
time_step = 10
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2])  # 3D 배열로 변환


time_step = 10  # 시퀀스 길이
X, y = create_dataset(scaled_data, time_step)

# LSTM 입력 형식으로 변환
X = X.reshape(X.shape[0], X.shape[1], 1)

# 모델 구축
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
model.fit(X, y, epochs=100, batch_size=32)

# 예측하기 위해 데이터 준비
train_predict = model.predict(X)

# 예측 결과 스케일 복원
train_predict = scaler.inverse_transform(train_predict)

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(series, label='Actual Data', color='blue')
plt.plot(range(time_step, time_step + len(train_predict)), train_predict, label='LSTM Predictions', color='red')
plt.title('LSTM Predictions')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.show()