# analise_comparativa_local_sem_etapa.py

import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# Models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Metrics and Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score

print("--- STAGE 1: LOADING AND PREPARING DATA ---")
try:
    df = pd.read_parquet('data/saeb_final_processado.parquet')
    print("Processed dataset loaded successfully.")
except FileNotFoundError:
    print("ERROR: File 'saeb_final_processado.parquet' not found.")
    exit()

# Create the target variable using the global mean, as defined
print("Calculating binary performance based on the dataset's global mean...")
media_proficiencia_global = df['PROFICIENCIA_MT_SAEB'].mean()
df['desempenho_binario'] = (df['PROFICIENCIA_MT_SAEB'] >= media_proficiencia_global).astype(int)

# Separate features (X) and target (y)
X = df.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'desempenho_binario'], errors='ignore')
y = df['desempenho_binario']

# **KEY ADJUSTMENT APPLIED HERE**
# We remove the 'ETAPA_ENSINO' (teaching stage) column from the feature set
# so it is not used in the prediction.
print("Removing the 'ETAPA_ENSINO' variable from the model features...")
if 'ETAPA_ENSINO' in X.columns:
    X = X.drop(columns=['ETAPA_ENSINO'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply Scaler (now error-free, as there are no more text columns)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")


print("\n--- STAGE 2: MODEL DEFINITION AND TRAINING (CPU-OPTIMIZED) ---")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(n_jobs=-1, random_state=42),
    "LightGBM": LGBMClassifier(n_jobs=-1, random_state=42, verbosity=-1),
    "CatBoost": CatBoostClassifier(thread_count=-1, random_state=42, verbose=0)
}

detailed_reports = {}
summary_results = []
roc_data = {}

for name, model in models.items():
    print(f"\nTraining model: {name}...")
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    target_names = ['Below Average (0)', 'Above Average (1)']
    report = classification_report(y_test, y_pred, target_names=target_names)
    detailed_reports[name] = report
    
    summary_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    })
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_score(y_test, y_prob)}

# --- STAGE 3: RESULTS AND VISUALIZATIONS ---
# (The rest of the code for generating reports and graphs remains the same)
print("\n" + "="*60)
print(" DETAILED CLASSIFICATION REPORT PER MODEL ")
print("="*60)
for name, report in detailed_reports.items():
    print(f"\nModel: {name}")
    print(report)
    print("-"*60)

results_df = pd.DataFrame(summary_results)
print("\n" + "="*60)
print(" SUMMARY TABLE - OVERALL ACCURACY AND AUC ")
print("="*60)
print(results_df.set_index('Model').round(4))

print("\n--- Generating Comparative ROC Curve ---")

# Define output directory
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 8))
for name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], label=f'{name} (AUC = {data["auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative ROC Curve of Models')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
output_path = os.path.join(output_dir, "comparative_roc_curve.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Comparative ROC curve saved to {output_path}")
# plt.show() # Optional: Commented out to prevent blocking script execution
