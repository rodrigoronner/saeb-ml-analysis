# Machine Learning Analysis of SAEB 2023 Dataset

This repository contains the code and data for a machine learning analysis based on the SAEB 2023 dataset. The primary goal of this research is to investigate the relationship between educational factors, particularly teacher training, and student performance in Brazil.

The scripts provided allow for the replication of our key experiments, including feature selection, model comparison, and model interpretation.

## Dataset

* **`data/saeb_final_processado.parquet`**: The primary dataset used for all analyses. This is a pre-processed Parquet file containing anonymized data from the SAEB 2023 survey, including teacher qualifications, school infrastructure, student participation rates, and student proficiency scores in Mathematics (MT) and Language (LP).

## Project Structure

* `/data`: Contains the raw `.parquet` dataset.
* `/src`: Contains all executable Python scripts for the analysis.
* `/results`: The default output directory for all generated plots and reports. This directory is included in `.gitignore`.
* `requirements.txt`: A list of all necessary Python packages.

## Setup and Installation

To replicate this analysis, please follow these steps:

1.  **Clone the repository:**
    ```bash\
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/saeb-ml-analysis.git
    cd saeb-ml-analysis
   

2.  **Install Git LFS:**
    This project uses Git Large File Storage (LFS) for the dataset. You must have Git LFS installed.
    ```bash
    git lfs install
    git lfs pull
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\\Scripts\\activate`
    ```

4.  **Install Dependencies:**
    ```bash\
    pip install -r requirements.txt
    ```

## Running the Experiments

The analysis is divided into several scripts located in the `/src` directory. They are numbered in a suggested order of execution.

**Note on Target Variables:** This project tests multiple definitions for the target variable (`desempenho_binario` - binary performance). Each script clearly defines how its target variable is created.


### 1. Feature Selection (`1_feature_selection_boruta.py`)

This script utilizes the **Boruta** algorithm to perform rigorous feature selection on a 10% sample of the dataset, identifying all statistically relevant features.

* **Target Variable:** `desempenho_binario` is defined as student Math proficiency (`PROFICIENCIA_MT_SAEB`) being **above or below the mean for their specific teaching stage** (`media_por_etapa`).
* **To Run:**
    ```bash
    python src/1_feature_selection_boruta.py
    ```
* **Output:** Prints the list of confirmed important features and rejected features to the console.


---

### 2. Model Comparison (`2_model_comparison.py`)

This script trains and evaluates four different tree-based models to find the best-performing classifier for this task.\

* **Models:** `RandomForestClassifier`, `XGBClassifier`, `LGBMClassifier`, `CatBoostClassifier`.
* **Target Variable:** `desempenho_binario` is defined as student Math proficiency (`PROFICIENCIA_MT_SAEB`) being **above or below the global mean** (`media_proficiencia_global`) across all stages.
* **Note:** This analysis **drops the `ETAPA_ENSINO` (teaching stage) column** to prevent it from being a direct predictor.
* **To Run:**
    ```bash
    python src/2_model_comparison.py
    ```
* **Output:**
    * Prints detailed classification reports for each model to the console.
    * Displays a comparative ROC Curve plot.

---

### 3. Model Interpretation (`3_model_interpretation_shap.py`)\

This script uses **SHAP (SHapley Additive exPlanations)** to interpret the `RandomForestClassifier` and understand *why* it makes its predictions. It runs on a large sample (`40,000`) for stability.

* **Target Variable:** `desempenho_binario` is defined using a **composite score** (mean of Math and Portuguese proficiency) compared to the **global composite mean**.
* **Note:** This analysis also **drops the `ETAPA_ENSINO` column**.
* **To Run:**
    ```bash
    python src/3_model_interpretation_shap.py
    ```
* **Output:**
    * Saves a SHAP summary plot (`summary_plot_geral_final_...png`) to the `/results` folder.
    * Saves SHAP dependence plots for key features to the `/results` folder.

---

### 4. Classification Analysis (`4_classification_analysis_rf.py`)

This script provides a deep-dive analysis into a single `RandomForestClassifier`, focusing on generating a detailed confusion matrix.

* **Target Variable:** Same as the Boruta script: student Math proficiency (`PROFICIENCIA_MT_SAEB`) is compared to the **mean for their specific teaching stage** (`media_por_etapa`).
* **Note:** This analysis **keeps and one-hot encodes `ETAPA_ENSINO`** as a feature.
* **To Run:**
    ```bash
    python src/4_classification_analysis_rf.py
    ```
* **Output:**
    * Saves a detailed confusion matrix plot (`confusion_matrix_ampliado.png`) to the `/results` folder.

## Citation

If you use this code or data in your research, please cite our article: Tertulino, R., & Almeida, R. (2025). A Multi-level Analysis of Factors Associated with Student Performance: A Machine Learning Approach to the SAEB Microdata. arXiv preprint arXiv:2510.22266.
