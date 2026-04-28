import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "dataset_finale.xlsx")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("  EduRisk AI — Training Pipeline")
print("=" * 60)

# ================= LOAD DATA =================
print("\n[1/5] Chargement des données...")
df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()
df = df.fillna(0)
print(f"     ✅ {len(df)} élèves chargés, {len(df.columns)} colonnes")

# ================= TARGET =================
print("\n[2/5] Construction de la cible (Dropout)...")
if "Dropout" not in df.columns:
    df["Dropout"] = (df["score engagement (kpo)"] < 0.4).astype(int)
    print(f"     ✅ Cible créée : {df['Dropout'].sum()} abandons ({df['Dropout'].mean()*100:.1f}%)")
else:
    print(f"     ✅ Cible existante : {df['Dropout'].sum()} abandons ({df['Dropout'].mean()*100:.1f}%)")

# ================= FEATURES =================
print("\n[3/5] Préparation des features...")
X = df.drop(columns=["Dropout"], errors="ignore")
y = df["Dropout"]
X = X.select_dtypes(include=[np.number])
columns = X.columns.tolist()
print(f"     ✅ {len(columns)} features numériques retenues")

# ================= SCALE & SPLIT =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     ✅ Split : {len(X_train)} train / {len(X_test)} test")

# ================= TRAIN MODELS =================
print("\n[4/5] Entraînement des modèles...")
results = {}

# --- Random Forest ---
print("     🌲 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, rf_pred),
    "auc": roc_auc_score(y_test, rf_proba)
}
print(f"        ACC={results['Random Forest']['accuracy']:.3f}  AUC={results['Random Forest']['auc']:.3f}")

# --- XGBoost ---
print("     ⚡ XGBoost...")
scale_pos = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos,
    eval_metric="logloss",
    random_state=42,
    verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
results["XGBoost"] = {
    "accuracy": accuracy_score(y_test, xgb_pred),
    "auc": roc_auc_score(y_test, xgb_proba)
}
print(f"        ACC={results['XGBoost']['accuracy']:.3f}  AUC={results['XGBoost']['auc']:.3f}")

# --- ANN ---
print("     🧠 Neural Network (ANN)...")
ann_model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
ann_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
ann_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.15,
    callbacks=[es],
    verbose=0
)
ann_proba = ann_model.predict(X_test, verbose=0).flatten()
ann_pred = (ann_proba > 0.5).astype(int)
results["ANN"] = {
    "accuracy": accuracy_score(y_test, ann_pred),
    "auc": roc_auc_score(y_test, ann_proba)
}
print(f"        ACC={results['ANN']['accuracy']:.3f}  AUC={results['ANN']['auc']:.3f}")

# --- Ensemble ---
ens_proba = (ann_proba + rf_proba + xgb_proba) / 3
ens_pred = (ens_proba > 0.5).astype(int)
results["Ensemble"] = {
    "accuracy": accuracy_score(y_test, ens_pred),
    "auc": roc_auc_score(y_test, ens_proba)
}
print(f"     🔀 Ensemble : ACC={results['Ensemble']['accuracy']:.3f}  AUC={results['Ensemble']['auc']:.3f}")

# ================= SAVE =================
print("\n[5/5] Sauvegarde des modèles...")
joblib.dump(rf_model,   os.path.join(MODEL_DIR, "rf_model.pkl"))
joblib.dump(xgb_model,  os.path.join(MODEL_DIR, "xgb_model.pkl"))
joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(columns,    os.path.join(MODEL_DIR, "columns.pkl"))
ann_model.save(os.path.join(MODEL_DIR, "ann_model.h5"))
print("     ✅ Tous les modèles sauvegardés dans /models/")

# ================= SUMMARY =================
print("\n" + "=" * 60)
print("  RÉSUMÉ DES PERFORMANCES")
print("=" * 60)
print(f"  {'Modèle':<20} {'Accuracy':>10} {'AUC':>10}")
print("-" * 42)
for m, v in results.items():
    print(f"  {m:<20} {v['accuracy']:>10.3f} {v['auc']:>10.3f}")
best = max(results, key=lambda x: results[x]["auc"])
print("=" * 60)
print(f"  🏆 Meilleur modèle (AUC) : {best} — {results[best]['auc']:.3f}")
print("=" * 60)
print(f"\n✅ TRAINING TERMINÉ — {len(columns)} features, {len(df)} élèves")