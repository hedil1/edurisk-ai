import shap
import numpy as np

def explain(model, X):

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    return values