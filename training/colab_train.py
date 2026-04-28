import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
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
    # Utiliser un seuil sur le score d'engagement
    df["Dropout"] = (df["score engagement (kpo)"] < 0.4).astype(int)
    print(f"     ✅ Cible créée : {df['Dropout'].sum()} abandons ({df['Dropout'].mean()*100:.1f}%)")
else:
    print(f"     ✅ Cible existante : {df['Dropout'].sum()} abandons ({df['Dropout'].mean()*100:.1f}%)")

# ================= FEATURES =================
print("\n[3/5] Préparation des features...")
X = df.drop(columns=["Dropout", "candidat"], errors="ignore")
y = df["Dropout"]

# Garder seulement les colonnes numériques
X = X.select_dtypes(include=[np.number])
columns = X.columns.tolist()
print(f"     ✅ {len(columns)} features numériques retenues")
print(f"     Features: {columns[:5]}... (et {len(columns)-5} autres)")

# ================= SCALE & SPLIT =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split avant SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"     ✅ Split : {len(X_train)} train / {len(X_test)} test")
print(f"     Train - Non-abandons: {(y_train==0).sum()}, Abandons: {(y_train==1).sum()}")
print(f"     Test  - Non-abandons: {(y_test==0).sum()}, Abandons: {(y_test==1).sum()}")

# ================= SMOTE pour rééquilibrer =================
print("\n     📊 Application de SMOTE pour rééquilibrer les classes...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"     ✅ Après SMOTE - Train: {len(X_train_balanced)} samples")
print(f"        Non-abandons: {(y_train_balanced==0).sum()}, Abandons: {(y_train_balanced==1).sum()}")

# ================= TRAIN MODELS =================
print("\n[4/5] Entraînement des modèles...")
results = {}

# --- Random Forest (sur données originales, avec class_weight) ---
print("     🌲 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)  # Utiliser les données originales
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]
results["Random Forest"] = {
    "accuracy": accuracy_score(y_test, rf_pred),
    "auc": roc_auc_score(y_test, rf_proba)
}
print(f"        ACC={results['Random Forest']['accuracy']:.3f}  AUC={results['Random Forest']['auc']:.3f}")

# --- XGBoost (sur données équilibrées par SMOTE) ---
print("     ⚡ XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_weight=3,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
    use_label_encoder=False
)
xgb_model.fit(X_train_balanced, y_train_balanced)  # Utiliser les données équilibrées
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
results["XGBoost"] = {
    "accuracy": accuracy_score(y_test, xgb_pred),
    "auc": roc_auc_score(y_test, xgb_proba)
}
print(f"        ACC={results['XGBoost']['accuracy']:.3f}  AUC={results['XGBoost']['auc']:.3f}")

# --- ANN (sur données originales, avec class_weight) ---
print("     🧠 Neural Network (ANN)...")
# Calcul des poids de classe pour l'ANN
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

ann_model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
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
ann_model.compile(
    optimizer="adam", 
    loss="binary_crossentropy", 
    metrics=["accuracy", "auc"]
)
es = EarlyStopping(
    monitor="val_loss", 
    patience=10, 
    restore_best_weights=True,
    verbose=0
)
history = ann_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    class_weight=class_weight_dict,
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

# --- Ensemble (moyenne des 3 modèles) ---
ens_proba = (rf_proba + xgb_proba + ann_proba) / 3
ens_pred = (ens_proba > 0.5).astype(int)
results["Ensemble"] = {
    "accuracy": accuracy_score(y_test, ens_pred),
    "auc": roc_auc_score(y_test, ens_proba)
}
print(f"     🔀 Ensemble : ACC={results['Ensemble']['accuracy']:.3f}  AUC={results['Ensemble']['auc']:.3f}")

# ================= SAVE MODELS =================
# ================= SAVE MODELS =================
print("\n[5/5] Sauvegarde des modèles...")

try:
    joblib.dump(rf_model,   os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(xgb_model,  os.path.join(MODEL_DIR, "xgb_model.pkl"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(columns,    os.path.join(MODEL_DIR, "columns.pkl"))

    # Sauvegarde robuste pour déploiement
    ann_model.save(os.path.join(MODEL_DIR, "ann_model.h5"))

    print("     ✅ ANN sauvegardé (architecture.json + weights.h5)")
    print("     ✅ Tous les modèles sauvegardés !")

except Exception as e:
    print(f"     ❌ Erreur : {e}")

# ================= DETAILED SUMMARY =================
print("\n" + "=" * 60)
print("  RÉSUMÉ DES PERFORMANCES")
print("=" * 60)
print(f"  {'Modèle':<20} {'Accuracy':>10} {'AUC':>10} {'Status':>10}")
print("-" * 52)
for m, v in results.items():
    status = "✅" if v['auc'] > 0.8 else "⚠️" if v['auc'] > 0.6 else "❌"
    print(f"  {m:<20} {v['accuracy']:>10.3f} {v['auc']:>10.3f} {status:>10}")

best = max(results, key=lambda x: results[x]["auc"])
print("=" * 60)
print(f"  🏆 Meilleur modèle (AUC) : {best} — {results[best]['auc']:.3f}")
print("=" * 60)

# ================= MATRICES DE CONFUSION =================
print("\n📊 Matrices de confusion sur le test set:")
for model_name in results.keys():
    if model_name == "Random Forest":
        cm = confusion_matrix(y_test, rf_pred)
    elif model_name == "XGBoost":
        cm = confusion_matrix(y_test, xgb_pred)
    elif model_name == "ANN":
        cm = confusion_matrix(y_test, ann_pred)
    elif model_name == "Ensemble":
        cm = confusion_matrix(y_test, ens_pred)
    else:
        continue
    
    print(f"\n  {model_name}:")
    print(f"    Vrais Négatifs: {cm[0,0]:3d}  |  Faux Positifs: {cm[0,1]:3d}")
    print(f"    Faux Négatifs:  {cm[1,0]:3d}  |  Vrais Positifs: {cm[1,1]:3d}")

print("\n" + "=" * 60)
print(f"✅ TRAINING TERMINÉ — {len(columns)} features, {len(df)} élèves")
print("=" * 60)
print("\n📁 Modèles sauvegardés dans :", MODEL_DIR)
print("   - rf_model.pkl")
print("   - xgb_model.pkl")
print("   - scaler.pkl")
print("   - columns.pkl")
print("   - ann_model.h5")