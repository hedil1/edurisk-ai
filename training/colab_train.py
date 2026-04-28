import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset_finale.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  EduRisk AI — Training Pipeline (FIXED)")
print("=" * 60)

# ================= LOAD DATA =================
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.fillna(0)

# ================= TARGET =================
if "Dropout" not in df.columns:
    df["Dropout"] = (df["score engagement (kpo)"] < 0.4).astype(int)

# ================= FEATURES =================
X = df.drop(columns=["Dropout", "candidat"], errors="ignore")
X = X.select_dtypes(include=[np.number])
y = df["Dropout"]

columns = X.columns.tolist()

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ================= SMOTE =================
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ================= RANDOM FOREST =================
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)

rf_proba = rf.predict_proba(X_test)[:, 1]

# ================= XGBOOST =================
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train_bal, y_train_bal)

xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# ================= ANN FIXED =================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

ann = Sequential([
    Input(shape=(X_train.shape[1],)),   # ✅ FIX IMPORTANT

    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

ann.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[es],
    verbose=0
)

ann_proba = ann.predict(X_test, verbose=0).flatten()

# ================= ENSEMBLE =================
ensemble_proba = (rf_proba + xgb_proba + ann_proba) / 3

# ================= METRICS =================
print("\nRESULTS:")
print("RF AUC:", roc_auc_score(y_test, rf_proba))
print("XGB AUC:", roc_auc_score(y_test, xgb_proba))
print("ANN AUC:", roc_auc_score(y_test, ann_proba))
print("ENS AUC:", roc_auc_score(y_test, ensemble_proba))

# ================= SAVE MODELS =================
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(columns, os.path.join(MODEL_DIR, "columns.pkl"))

# ✅ IMPORTANT FIX: SAVE IN .keras FORMAT
ann.save(os.path.join(MODEL_DIR, "ann_model.keras"))

print("\n✅ ALL MODELS SAVED (FIXED FORMAT)")