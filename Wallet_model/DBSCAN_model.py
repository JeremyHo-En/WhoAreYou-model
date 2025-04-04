#%%
from sklearn.preprocessing import StandardScaler
output_file ='/Users/jeremy/Desktop/code_reset/wallet_features.csv'
data = pd.read_csv(output_file)
print("原始資料筆數：", len(data))
data = data.fillna(0)
data = data.dropna()
print("刪除 NaN 後剩下：", len(data))
#%%
# 標準化處理
Feture_list = data.columns
data = data[Feture_list[1:]]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
#%%
feature_names = data.columns.tolist()
scaled_df = pd.DataFrame(scaled_data, columns=feature_names)  # 用原本特徵名稱建立 DataFrame
#%%
nan_features = scaled_df.columns[scaled_df.isna().any()].tolist()
print("含 NaN 的特徵:", nan_features)

#%%
#scaled_df=scaled_df.drop(columns=['Unnamed: 17', 'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21', 'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29', 'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32'])
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
#%%
#scaled_df = pd.read_csv('/Users/jeremy/Desktop/code_reset/wallet_features.csv')
scaled_data = scaled_df.values
# 設定 K 為 2 * 維度數
k = 2 * data.shape[1]
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)

# 畫出 K 距離圖
distances = np.sort(distances[:, -1], axis=0)
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel(f"{k}-nearest neighbor distance")
plt.title("K-Distance Graph for DBSCAN")
plt.show()
#%%
from sklearn.cluster import DBSCAN

# 設定最佳參數
dbscan = DBSCAN(eps=4, min_samples=10)
data["cluster"] = dbscan.fit_predict(scaled_df)

# 標記異常交易
data["is_anomaly"] = (data["cluster"] == -1).astype(int)

# 顯示異常交易
print(data[data["is_anomaly"] == 1])