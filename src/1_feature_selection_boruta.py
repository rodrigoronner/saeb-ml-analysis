# analise_boruta.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
import time

print("--- STAGE 1: LOADING AND PREPARING DATA FOR BORUTA ---")
try:
    # Loading the processed dataset
    df = pd.read_parquet('data/saeb_final_processado.parquet')
    print("Dataset 'saeb_final_processado.parquet' loaded successfully.")
except FileNotFoundError:
    print("ERROR: File 'saeb_final_processado.parquet' not found.")
    print("Please run the 'pre_processing_final.py' script first.") # Assuming a pre-processing script name
    exit()

# For feature selection, it's feasible to use a representative sample
# to speed up the process, which is computationally intensive.
print("Creating a 10% sample of the dataset for Boruta analysis...")
df_sample = df.sample(frac=0.1, random_state=42)
del df

print("Preparing the sample data...")
# Robust creation of the target variable
df_sample['media_por_etapa'] = df_sample.groupby('ETAPA_ENSINO')['PROFICIENCIA_MT_SAEB'].transform('mean')
df_sample['desempenho_binario'] = (df_sample['PROFICIENCIA_MT_SAEB'] >= df_sample['media_por_etapa']).astype(int)

# Separation of features (X) and target (y)
X = df_sample.drop(columns=['PROFICIENCIA_MT_SAEB', 'PROFICIENCIA_LP_SAEB', 'desempenho_binario', 'media_por_etapa'], errors='ignore')
y = df_sample['desempenho_binario']

# One-Hot Encoding the 'ETAPA_ENSINO' column
X = pd.get_dummies(X, columns=['ETAPA_ENSINO'], drop_first=True)

# Boruta needs a numpy array, not a scaled dataframe
X_values = X.values
y_values = y.values.ravel() # .ravel() adjusts the array shape
print("Data preparation complete.")

# --- STAGE 2: EXECUTING THE BORUTA ALGORITHM ---
print("\n--- STAGE 2: EXECUTING THE BORUTA ALGORITHM ---")
print("This may take several minutes, depending on the sample size...")

# Define the model that Boruta will use internally (Random Forest)
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# Initialize Boruta
# The 'perc' parameter defines the percentile for comparison with shadow features.
# 'max_iter' is the maximum number of iterations.
boruta_selector = BorutaPy(
    estimator=rf,
    n_estimators='auto',
    verbose=2, # 0 for silent, 1 for progress, 2 for details
    random_state=42,
    max_iter=50 # Reduced for a faster run; 100 is the default
)

# Train Boruta
start_time = time.time()
boruta_selector.fit(X_values, y_values)
end_time = time.time()
print(f"Boruta analysis completed in {end_time - start_time:.2f} seconds.")


# --- STAGE 3: DISPLAYING THE RESULTS ---
print("\n--- STAGE 3: FEATURE SELECTION RESULTS ---")

# Create a dataframe with the results
results = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': boruta_selector.ranking_,
    'Confirmed': boruta_selector.support_
})

# Filter only the features confirmed as important
confirmed_features = results[results['Confirmed'] == True].sort_values(by='Ranking')

print("\n--- FEATURES CONFIRMED AS IMPORTANT BY BORUTA ---")
print(f"The Boruta algorithm confirmed {len(confirmed_features)} out of {len(X.columns)} features as statistically relevant.")
print("\nList of confirmed features (in order of importance):")
print(list(confirmed_features['Feature']))

print("\n--- REJECTED FEATURES ---")
rejected_features = results[results['Confirmed'] == False]
print(list(rejected_features['Feature']))
