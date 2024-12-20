import os
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 결과 저장 폴더 생성
results_dir = './results/'
os.makedirs(results_dir, exist_ok=True)

# 데이터 로드
file = './data/export_EMSC_kr.csv'
data = pd.read_csv(file, sep=';')

# 주요 열 선택
columns = ["Date", "Latitude", "Longitude", "Depth", "Magnitude"]
names = data.loc[:, columns]

# 지도 생성
map = folium.Map(location=[37, 127], zoom_start=7, tiles="cartodb positron")
for i in range(names['Date'].count()):
    folium.Circle(
        [names['Latitude'][i], names['Longitude'][i]],
        radius=names['Magnitude'][i] ** 6.5,
        color="white",
        fill_color="Red"
    ).add_to(map)

# 지도 저장
map.save(f'{results_dir}map.html')

# 날짜 열을 datetime으로 변환
names['Date'] = pd.to_datetime(names['Date'], errors='coerce')

# 시각화 1: Magnitude 히스토그램
plt.figure(figsize=(8, 6))
sns.histplot(names['Magnitude'], kde=True, bins=30, color='blue')
plt.title("Magnitude Distribution")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.savefig(f'{results_dir}magnitude_distribution.png')  # 저장

# 상관관계 계산을 위한 숫자형 데이터 선택
numeric_columns = names.select_dtypes(include=[np.number])

# 시각화 2: 상관관계 히트맵
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_columns.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.savefig(f'{results_dir}correlation_matrix.png')  # 저장

# 시각화 3: Depth와 Magnitude 시각화
plt.figure(figsize=(8, 6))
scatter = plt.scatter(names["Depth"], names["Magnitude"],
                       c=names["Magnitude"], cmap="viridis", alpha=0.7)
plt.title("Depth vs Magnitude")
plt.xlabel("Depth (km)")
plt.ylabel("Magnitude")
plt.colorbar(scatter, label="Magnitude")  # 컬러바 추가
plt.savefig(f'{results_dir}depth_vs_magnitude.png')  # 저장

# 머신러닝 데이터 준비
features = names[["Latitude", "Longitude", "Depth"]].copy()
features['Year'] = names['Date'].dt.year
target = names['Magnitude']

# 결측치 제거
features = features.dropna()
target = target[features.index]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# 모델 학습
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 성능 평가 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.title("True vs Predicted Magnitude")
plt.xlabel("True Magnitude")
plt.ylabel("Predicted Magnitude")
plt.savefig(f'{results_dir}true_vs_predicted_depth.png')  # 저장