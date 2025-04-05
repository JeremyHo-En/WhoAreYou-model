#%%
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

app = FastAPI()

# ==== 自動定位模型檔案路徑 ====
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ==== 載入模型與資料 ====
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
dbscan = joblib.load(MODEL_DIR / "dbscan.pkl")
core_samples = np.load(MODEL_DIR / "core_samples.npy")

with open(MODEL_DIR / "feature_names.json", "r") as f:
    feature_names = json.load(f)

core_df = pd.DataFrame(core_samples, columns=feature_names)

# ==== 輸入資料格式 ====
class InputFeatures(BaseModel):
    features: dict

# ==== 預測 API ====
@app.post("/predict")
def predict(input: InputFeatures):
    input_dict = input.features

    # 檢查缺少欄位
    missing = set(feature_names) - set(input_dict)
    if missing:
        return {"error": f"缺少欄位: {missing}"}

    # 預處理
    input_df = pd.DataFrame([input_dict])[feature_names]
    scaled_input = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_input, columns=feature_names)

    # 用 dbscan 判斷（附加一筆資料，重新群聚）
    combined = pd.concat([core_df, scaled_df], ignore_index=True)
    new_label = dbscan.fit_predict(combined)[-1]
    is_anomaly = int(new_label == -1)

    # ===== 特徵解釋邏輯 =====
    # 計算 z-score
    epsilon = 1e-8
    core_mean = core_df.mean()
    core_std = core_df.std() + epsilon  # 防止除以 0
    z_scores = (scaled_df - core_mean) / core_std

    # 依序找 z-score 超過門檻的特徵
    thresholds = [3, 2.5, 2, 1.5, 1, 2/3]
    selected_features = []
    used_threshold = None
    row = z_scores.iloc[0]

    for threshold in thresholds:
        features = row[abs(row) > threshold].sort_values(key=abs, ascending=False).index.tolist()
        if features:
            selected_features = features[:3]  # 只取最多前三個
            used_threshold = threshold
            break

    # 若找不到，回傳空清單
    return {
        "is_anomaly": is_anomaly,
        "top_3_features": selected_features,
        "features_valus":input_df[:3]
    }


# %%
