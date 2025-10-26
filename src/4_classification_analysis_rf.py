# analise_classificacao.py 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import time
import os # <--- CORRECTION APPLIED HERE

print("--- STAGE 1: LOADING AND PREPARING DATA ---")
try:
    df = pd.read_parquet('data/saeb_final_processado.parquet')
    print("Dataset 'saeb_final_processado.parquet' loaded successfully.")
except FileNotFoundError:
    print("ERROR: File 'saeb_final_processado.parquet' not found.")
    exit()

# Robust creation of the target variable
print("Calculating binary performance based on the mean of each teaching stage...")
df['media_por_etapa'] = df.groupby('ETAPA_ENSINO')['PROFICIENCIA_MT_SAEB'].transform('mean')
df['desempenho_binario'] = (df['PROFICIENCIA_MT_SAEB'] >= df['media_por_etapa']).astype(int)

# Separation of features (X) and target (y)
X = df.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'desempenho_binario', 'media_por_etapa'], errors='ignore')
y = df['desempenho_binario']

# One-Hot Encoding the 'ETAPA_ENSINO' column
print("Applying One-Hot Encoding to the 'ETAPA_ENSINO' column...")
X = pd.get_dummies(X, columns=['ETAPA_ENSINO'], drop_first=True)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply Scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Data preparation complete.")


print("\n--- STAGE 2: RANDOM FOREST MODEL TRAINING ---")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

print("Training the Random Forest model...")
start_time = time.time()
model.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")


print("\n--- STAGE 3: EVALUATION AND CONFUSION MATRIX GENERATION ---")
y_pred = model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

total = len(y_test)
tn_pct = (tn / total) * 100
fp_pct = (fp / total) * 100
fn_pct = (fn / total) * 100
tp_pct = (tp / total) * 100

labels = np.array([
    [f'True Negative\n{tn}\n({tn_pct:.2f}%)', f'False Positive\n{fp}\n({fp_pct:.2f}%)'],
    [f'False Negative\n{fn}\n({fn_pct:.2f}%)', f'True Positive\n{tp}\n({tp_pct:.2f}%)']
])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
            xticklabels=['Below Average', 'Above Average'],
            yticklabels=['Below Average', 'Above Average'])
plt.title('Confusion Matrix for Random Forest (Full Dataset)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

output_dir = "results" # <-- Changed to 'results'
os.makedirs(output_dir, exist_ok=True) # Now this line will work
plt.savefig(os.path.join(output_dir, 'confusion_matrix_ampliado.png'), dpi=300)
print(f"Plot 'confusion_matrix_ampliado.png' saved in '{output_dir}'.")
# plt.show() # Optional: Commented out
