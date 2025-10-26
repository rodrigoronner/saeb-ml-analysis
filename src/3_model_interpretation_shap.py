# analise_shap_final.py (Corrigido com Média Composta)

import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt

# ==============================================================================
# PARÂMETRO DE AMOSTRAGEM
N_AMOSTRAS = 40000 
# ==============================================================================

print(f"--- FASE 1: PREPARANDO OS DADOS PARA A ANÁLISE SHAP (Amostra de {N_AMOSTRAS} registros) ---")

try:
    df_full = pd.read_parquet('data/saeb_final_processado.parquet')
    n_samples = min(N_AMOSTRAS, len(df_full))
    df = df_full.sample(n=n_samples, random_state=42)
    del df_full
    print(f"Dataset de amostra com {len(df)} linhas carregado.")
except FileNotFoundError:
    print("ERRO: O arquivo 'saeb_final_processado_corrigido.parquet' não foi encontrado.")
    exit()

# --- PREPARAÇÃO DOS DADOS (COM A CORREÇÃO DA MÉDIA COMPOSTA) ---
print("Calculando o desempenho binário com base na média composta (Matemática + Português)...")

# 1. Cria a pontuação composta para cada aluno
df['composite_score'] = df[['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB']].mean(axis=1)

# 2. Calcula a média global desta pontuação composta
media_global_composita = df['composite_score'].mean()

# 3. Cria a variável alvo binária com base na pontuação composta
df['desempenho_binario'] = (df['composite_score'] >= media_global_composita).astype(int)

# 4. Separação de features (X) e alvo (y), removendo todas as colunas de proficiência
X = df.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'composite_score', 'desempenho_binario'], errors='ignore')
y = df['desempenho_binario']

# Removendo a variável 'ETAPA_ENSINO' das features do modelo
if 'ETAPA_ENSINO' in X.columns:
    X = X.drop(columns=['ETAPA_ENSINO'])

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
print("Preparação dos dados concluída.")

# --- FASE 2: TREINAR MODELO E CALCULAR SHAP ---
# (O restante do script permanece o mesmo)
print("\n--- FASE 2: TREINANDO MODELO E CALCULANDO VALORES SHAP ---")
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train_scaled, y_train)

explainer = shap.TreeExplainer(model)
print("Calculando valores SHAP para o conjunto de teste...")
shap_values = explainer(X_test_scaled_df, check_additivity=False)
print("Cálculo SHAP concluído.")

# --- FASE 3: GERANDO OS GRÁFICOS SHAP ---
# (O restante do script permanece o mesmo)
output_dir = "graficos_shap_finais"
os.makedirs(output_dir, exist_ok=True)

print("\nGerando o Gráfico de Resumo SHAP...")
shap.summary_plot(shap_values.values[:,:,1], X_test_scaled_df, show=False)
plt.title(f"SHAP Summary Plot (Sample Size: {n_samples})")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'summary_plot_geral_final_{n_samples}.png'), dpi=300, bbox_inches='tight')
plt.close()
print(f"Gráfico 'summary_plot_geral_final_{n_samples}.png' salvo em '{output_dir}'.")

# 2. Gráficos de Dependência para as Variáveis Numéricas Principais
print("\nGerando Gráficos de Dependência para as variáveis numéricas selecionadas...")
features_numericas_interesse = [
    'PC_FORMACAO_DOCENTE_FINAL',
    'TAXA_PARTICIPACAO_9EF'
]

for feature in features_numericas_interesse:
    if feature in X_test_scaled_df.columns:
        print(f"  - Gerando para '{feature}'...")
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

print(f"\nGráficos de dependência salvos em '{output_dir}'.")
print("\nAnálise concluída com sucesso!")