import numpy as np

def predict_model(x, model_name, rf, xgb_model, ann):

    if model_name == "RF":
        return rf.predict_proba(x)[0][1]

    if model_name == "XGB":
        return xgb_model.predict_proba(x)[0][1]

    if model_name == "ANN":
        return ann.predict(x, verbose=0)[0][0]

    # fallback soft voting
    p1 = rf.predict_proba(x)[0][1]
    p2 = xgb_model.predict_proba(x)[0][1]
    p3 = ann.predict(x, verbose=0)[0][0]

    return (p1 + p2 + p3) / 3