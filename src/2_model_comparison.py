# analise_comparativa_local_sem_etapa.py

import pandas as pd
import matplotlib.pyplot as plt
import time
import os

# Modelos
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Métricas e Pré-processamento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score

print("--- FASE 1: CARREGANDO E PREPARANDO OS DADOS ---")
try:
    df = pd.read_parquet('data/saeb_final_processado.parquet')
    print("Dataset corrigido carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'saeb_final_processado_corrigido.parquet' não foi encontrado.")
    exit()

# Criação da variável alvo usando a média global, como definido
print("Calculando o desempenho binário com base na média global do dataset...")
media_proficiencia_global = df['PROFICIENCIA_MT_SAEB'].mean()
df['desempenho_binario'] = (df['PROFICIENCIA_MT_SAEB'] >= media_proficiencia_global).astype(int)

# Separação das features e do alvo
X = df.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'desempenho_binario'], errors='ignore')
y = df['desempenho_binario']

# **AJUSTE PRINCIPAL APLICADO AQUI**
# Removemos a coluna 'ETAPA_ENSINO' do conjunto de features para que ela não seja usada na predição.
print("Removendo a variável 'ETAPA_ENSINO' das features do modelo...")
if 'ETAPA_ENSINO' in X.columns:
    X = X.drop(columns=['ETAPA_ENSINO'])

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicação do Scaler (agora sem erros, pois não há mais colunas de texto)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Preparação dos dados concluída.")


print("\n--- FASE 2: DEFINIÇÃO E TREINAMENTO DOS MODELOS (CPU-OPTIMIZED) ---")

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

# --- FASE 3: RESULTADOS E VISUALIZAÇÕES ---
# (O restante do código para gerar os relatórios e gráficos permanece o mesmo)
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
plt.figure(figsize=(10, 8))
for name, data in roc_data.items():
    plt.plot(data["fpr"], data["tpr"], label=f'{name} (AUC = {data["auc"]:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparative ROC Curve of Models')
plt.legend()
plt.grid(True)
plt.show()