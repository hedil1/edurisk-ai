from fastapi import FastAPI
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

rf = joblib.load("../models/rf_model.pkl")
xgb_model = joblib.load("../models/xgb_model.pkl")
ann = load_model("../models/ann_model.keras")
scaler = joblib.load("../models/scaler.pkl")

@app.post("/predict")
def predict(data: list):

    x = np.array(data).reshape(1, -1)
    x = scaler.transform(x)

    p = (
        ann.predict(x)[0][0] +
        rf.predict_proba(x)[0][1] +
        xgb_model.predict_proba(x)[0][1]
    ) / 3

    return {"risk": float(p)}