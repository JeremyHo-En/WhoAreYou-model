#%%
from fastapi import FastAPI, Request
import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI()

# ==== 自動定位模型檔案路徑 ====
BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path('/Users/jeremy/Desktop/token_predict_project')
MODEL_DIR = BASE_DIR / "model" / "base"

# ==== 載入模型與特徵順序 ====
def load_model_and_features(prefix):  # prefix: "ethereum_amplitude_model"
    model_path = MODEL_DIR / f"{prefix}.pkl"
    feature_path = MODEL_DIR / f"{prefix}_features.json"

    model = joblib.load(model_path)
    with open(feature_path, "r") as f:
        features = json.load(f)
    return model, features

# 正確載入方式
model_amp, features_amp = load_model_and_features("ethereum_amplitude_model")
model_dir, features_dir = load_model_and_features("ethereum_direction_model")
#%%
# ==== API 路由 ====
@app.post("/predict")
async def predict(request: Request):
    input_data = await request.json()

    try:
        # 整理特徵順序
        # 為了與模型特徵名稱一致，給 input_data 中的所有欄位名稱加上 '_x' 後綴
        input_data_x = {f"{key}_x" if key in ['marketCap', 'price'] else key: value for key, value in input_data.items()}

        # 使用加了 '_x' 後綴的 input_data_x 來整理特徵順序
        X_amp = pd.DataFrame([[input_data_x[f] for f in features_amp]], columns=features_amp)
        X_dir = pd.DataFrame([[input_data_x[f] for f in features_dir]], columns=features_dir)
    except KeyError as e:
        return {"error": f"Missing feature: {e}"}

    pred_amp = int(model_amp.predict(X_amp)[0])
    pred_dir = int(model_dir.predict(X_dir)[0])

    if pred_amp == 0:
        pred_amp = 'Market cap chance is less than 0.5%'
    else:
        pred_amp = 'Market cap chance is more than 0.5%'
    
    if pred_dir == 0:
        pred_dir = 'Increase'
    else:
        pred_dir = 'Decrease'

    return {
        "amplitude_class": pred_amp,
        "direction_class": pred_dir
    }

# %%
