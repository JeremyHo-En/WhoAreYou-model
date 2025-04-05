# base_token.py
from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from pathlib import Path

router = APIRouter()

# ==== 自動定位模型檔案路徑 ====
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "base_token"

# ==== 載入模型與特徵順序 ====
def load_model_and_features(prefix):  # prefix: "ethereum_amplitude_model"
    model_path = MODEL_DIR / f"{prefix}.pkl"
    feature_path = MODEL_DIR / f"{prefix}_features.json"

    model = joblib.load(model_path)
    with open(feature_path, "r") as f:
        features = json.load(f)
    return model, features

model_amp, features_amp = load_model_and_features("base_amplitude_model")
model_dir, features_dir = load_model_and_features("base_direction_model")

# ==== 輸入資料格式 ====
class TokenInput(BaseModel):
    features: dict

# ==== API ====
@router.post("/baseTokenPredict")
def predict(input: TokenInput):
    input_data = input.features
    try:
        # 加上 '_x' 後綴
        input_data_x = {
            f"{key}_x" if key in ['marketCap', 'price'] else key: value
            for key, value in input_data.items()
        }

        # 預測資料準備
        X_amp = pd.DataFrame([[input_data_x[f] for f in features_amp]], columns=features_amp)
        X_dir = pd.DataFrame([[input_data_x[f] for f in features_dir]], columns=features_dir)

    except KeyError as e:
        return {"error": f"Missing feature: {e}"}

    pred_amp = int(model_amp.predict(X_amp)[0])
    pred_dir = int(model_dir.predict(X_dir)[0])

    return {
        "amplitude_class": (
            "Market cap chance is more than 0.5%" if pred_amp else "Market cap chance is less than 0.5%"
        ),
        "direction_class": (
            "Decrease" if pred_dir else "Increase"
        )
    }
