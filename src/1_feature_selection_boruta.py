# analise_boruta.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import time

print("--- FASE 1: CARREGANDO E PREPARANDO OS DADOS PARA O BORUTA ---")
try:
    # Carregando o dataset corrigido
    df = pd.read_parquet('data/saeb_final_processado.parquet')
    print("Dataset 'saeb_final_processado_corrigido.parquet' carregado com sucesso.")
except FileNotFoundError:
    print("ERRO: O arquivo 'saeb_final_processado_corrigido.parquet' não foi encontrado.")
    print("Por favor, execute o script 'pre_processamento_final.py' primeiro.")
    exit()

# Para a seleção de features, é viável usar uma amostra representativa
# para acelerar o processo, que é computacionalmente intensivo.
print("Criando uma amostra de 10% do dataset para a análise Boruta...")
df_sample = df.sample(frac=0.1, random_state=42)
del df

print("Preparando os dados da amostra...")
# Criação robusta da variável alvo
df_sample['media_por_etapa'] = df_sample.groupby('ETAPA_ENSINO')['PROFICIENCIA_MT_SAEB'].transform('mean')
df_sample['desempenho_binario'] = (df_sample['PROFICIENCIA_MT_SAEB'] >= df_sample['media_por_etapa']).astype(int)

# Separação das features e do alvo
X = df_sample.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'desempenho_binario', 'media_por_etapa'], errors='ignore')
y = df_sample['desempenho_binario']

# One-Hot Encoding da coluna 'ETAPA_ENSINO'
X = pd.get_dummies(X, columns=['ETAPA_ENSINO'], drop_first=True)

# O Boruta precisa de um array numpy, não de um dataframe escalado
X_values = X.values
y_values = y.values.ravel() # .ravel() ajusta o formato do array
print("Preparação dos dados concluída.")

# --- FASE 2: EXECUÇÃO DO ALGORITMO BORUTA ---
print("\n--- FASE 2: EXECUTANDO O ALGORITMO BORUTA ---")
print("Isso pode levar vários minutos, dependendo do tamanho da amostra...")

# Define o modelo que o Boruta usará internamente (Random Forest)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Inicializa o Boruta
# O parâmetro 'perc' define o quantil para a comparação com as features sombra.
# 'max_iter' é o número máximo de iterações.
boruta_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2, # 0 para silencioso, 1 para progresso, 2 para detalhes
    random_state=42,
    max_iter=50 # Reduzido para uma execução mais rápida; 100 é o padrão
)

# Treina o Boruta
start_time = time.time()
boruta_selector.fit(X_values, y_values)
end_time = time.time()
print(f"Análise Boruta concluída em {end_time - start_time:.2f} segundos.")


# --- FASE 3: EXIBIÇÃO DOS RESULTADOS ---
print("\n--- FASE 3: RESULTADOS DA SELEÇÃO DE FEATURES ---")

# Cria um dataframe com os resultados
results = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': boruta_selector.ranking_,
    'Confirmed': boruta_selector.support_
})

# Filtra apenas as features confirmadas como importantes
confirmed_features = results[results['Confirmed'] == True].sort_values(by='Ranking')

print("\n--- FEATURES CONFIRMADAS COMO IMPORTANTES PELO BORUTA ---")
print(f"O algoritmo Boruta confirmou {len(confirmed_features)} de {len(X.columns)} features como estatisticamente relevantes.")
print("\nLista de features confirmadas (em ordem de importância):")
print(list(confirmed_features['Feature']))

print("\n--- FEATURES REJEITADAS ---")
rejected_features = results[results['Confirmed'] == False]
print(list(rejected_features['Feature']))