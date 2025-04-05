#%%
# train_api/main.py
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import os
from pathlib import Path
#%%
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path("/Users/jeremy/Desktop/eth_anomaly_api_project/train_api")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR.parent / "predict_api" / "models" / "base"

# 載入資料
data_path = DATA_DIR / "Base_wallet_features.csv"
# 載入資料
data = pd.read_csv(data_path).dropna()

workingDir = os.getcwd()
# 取特徵欄位（從第2欄開始）
feature_cols = data.columns[1:]
data = data[feature_cols]

# 標準化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)

# 訓練 DBSCAN
dbscan = DBSCAN(eps=5, min_samples=30)
labels = dbscan.fit_predict(scaled_df)

# 標記異常
scaled_df["cluster"] = labels
scaled_df["is_anomaly"] = (labels == -1).astype(int)

# 取得 core samples（正常點）
core_samples = scaled_df[scaled_df["is_anomaly"] == 0][feature_cols]

# 儲存模型
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(dbscan, MODEL_DIR / "dbscan.pkl")
np.save(MODEL_DIR / "core_samples.npy", core_samples.to_numpy())

# 儲存特徵名稱
with open(MODEL_DIR / "feature_names.json", "w") as f:
    json.dump(feature_cols.tolist(), f)

print('Finish')

# %%
