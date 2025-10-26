# shap_analysis_final.py 

import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt

# ==============================================================================
# SAMPLING PARAMETER
N_SAMPLES = 40000
# ==============================================================================

print(f"--- STAGE 1: PREPARING DATA FOR SHAP ANALYSIS (Sample of {N_SAMPLES} records) ---")

try:
    df_full = pd.read_parquet('data/saeb_final_processado.parquet')
    n_samples = min(N_SAMPLES, len(df_full))
    df = df_full.sample(n=n_samples, random_state=42)
    del df_full
    print(f"Sample dataset with {len(df)} rows loaded.")
except FileNotFoundError:
    print("ERROR: File 'saeb_final_processado.parquet' not found.")
    exit()

# --- DATA PREPARATION (WITH COMPOSITE MEAN CORRECTION) ---
print("Calculating binary performance based on the composite mean (Math + Portuguese)...")

# 1. Create the composite score for each student
df['composite_score'] = df[['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB']].mean(axis=1)

# 2. Calculate the global mean of this composite score
global_composite_mean = df['composite_score'].mean()

# 3. Create the binary target variable based on the composite score
df['desempenho_binario'] = (df['composite_score'] >= global_composite_mean).astype(int)

# 4. Separate features (X) and target (y), removing all proficiency columns
X = df.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'composite_score', 'desempenho_binario'], errors='ignore')
y = df['desempenho_binario']

# Removing the 'ETAPA_ENSINO' (teaching stage) variable from model features
if 'ETAPA_ENSINO' in X.columns:
    X = X.drop(columns=['ETAPA_ENSINO'])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
print("Data preparation complete.")

# --- STAGE 2: TRAIN MODEL AND CALCULATE SHAP ---
# (The rest of the script remains the same)
print("\n--- STAGE 2: TRAINING MODEL AND CALCULATING SHAP VALUES ---")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train_scaled, y_train)

explainer = shap.TreeExplainer(model)
print("Calculating SHAP values for the test set...")
shap_values = explainer(X_test_scaled_df, check_additivity=False)
print("SHAP calculation complete.")

# --- STAGE 3: GENERATING SHAP PLOTS ---
# (The rest of the script remains the same)
output_dir = "results" # <-- Changed to 'results'
os.makedirs(output_dir, exist_ok=True)

print("\nGenerating the SHAP Summary Plot...")
shap.summary_plot(shap_values.values[:,:,1], X_test_scaled_df, show=False)
plt.title(f"SHAP Summary Plot (Sample Size: {n_samples})")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'summary_plot_geral_final_{n_samples}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot 'summary_plot_geral_final_{n_samples}.png' saved in '{output_dir}'.")

# 2. Dependence Plots for Key Numerical Variables
print("\nGenerating Dependence Plots for selected numerical variables...")
features_of_interest_numeric = [
    'PC_FORMACAO_DOCENTE_FINAL',
    'TAXA_PARTICIPACAO_9EF'
]

for feature in features_of_interest_numeric:
    if feature in X_test_scaled_df.columns:
        print(f"  - Generating for '{feature}'...")
        fig, ax = plt.subplots()
        shap.dependence_plot(
            feature,
            shap_values.values[:,:,1],
            X_test_scaled_df,
            show=False,
            ax=ax
        )
        ax.set_title(f'SHAP Dependence Plot')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'dependence_plot_{feature}_{n_samples}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

print(f"\nDependence plots saved in '{output_dir}'.")
print("\nAnalysis complete!")
