<div align="center">

# 🤖 Automated Machine Learning Model Benchmark

**Automatically preprocess, train, tune, and benchmark multiple ML models to find the best one for your data — no manual coding required.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-✓-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-✓-02B875?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

</div>

---

## 🔍 What Is This?

A **single-file, zero-configuration ML pipeline** (`main.py`, 466 lines) that takes any tabular CSV and:

1. Cleans and preprocesses the data automatically
2. Auto-detects whether it's a **classification** or **regression** task
3. Applies **SMOTE** if class imbalance is detected
4. Trains and tunes **12+ classification** or **15+ regression** models using `GridSearchCV`
5. Picks the best model using **smart tie-breaking** (priority + cross-validation)
6. Exports a **ready-to-deploy `.pkl`** file

---

## 🗂️ Repository Structure

```
Automated-Machine-Learning-Model-Benchmark/
│
├── main.py                          # ← Entire pipeline in one file (466 lines)
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project metadata
├── uv.lock                          # Lockfile (uv package manager)
├── .python-version                  # Python version pin
├── .gitignore
│
├── datasets/                        # Put your CSV files here
│
├── cleaned_data.csv                 # Auto-generated: preprocessed dataset
└── best_model_Gradient_Boosting.pkl # Auto-generated: saved best model
```

---

## ⚙️ Configuration

All settings live at the **top of `main.py`** — no CLI flags, no config files:

```python
CSV_PATH           = r"datasets/your_file.csv"   # Path to your input CSV
TARGET_COLUMN      = "YourTarget"                 # Column to predict

TEST_SIZE          = 0.20      # 80/20 train-test split
RANDOM_STATE       = 42

MISSING_THRESHOLD  = 0.5       # Drop columns with >50% missing values
OUTLIER_IQR_FACTOR = 1.5       # IQR multiplier for outlier capping
IMBALANCE_THRESHOLD= 0.2       # Apply SMOTE if minority class < 20%
USE_SMOTE          = True      # Toggle SMOTE on/off
SCALE_TARGET       = True      # Scale target for regression tasks
```

---

## 🔄 Pipeline — Step by Step

```
CSV Input
   │
   ▼
[Step 1]  Load dataset → preview first 3 rows
   │
   ▼
[Step 2]  Drop irrelevant columns
          • >50% missing  →  removed
          • constant (nunique ≤ 1)  →  removed
          • ID columns (nunique == n_rows)  →  removed
   │
   ▼
[Step 3]  Type detection
          • numeric with ≤10 unique values  →  treated as categorical
          • everything else  →  numerical
   │
   ▼
[Step 4]  Missing value imputation
          • numerical  →  median
          • categorical  →  mode (most frequent)
   │
   ▼
[Step 5]  Outlier capping  (IQR × 1.5 by default)
   │
   ▼
[Step 6]  Encoding + Scaling
          • ≤10 unique  →  OneHotEncoding
          • >10 unique  →  LabelEncoding
          • numerical features  →  StandardScaler
   │
   ▼
[Step 7]  Export  cleaned_data.csv
   │
   ▼
[Step 8]  Auto-detect task type
          • target unique values ≤ 20  →  Classification
          • target unique values > 20  →  Regression
   │
   ▼
[Step 9]  Class Imbalance Check  (classification only)
          • minority ratio < 0.20  →  SMOTE applied
          • already balanced  →  SMOTE skipped
   │
   ▼
[Step 10]  Train & benchmark all models  (GridSearchCV, 5-fold CV)
   │
   ▼
[Step 10b] Smart model selection  (tie-breaking: priority + CV score)
   │
   ▼
[Step 11]  Save best model  →  best_model_<Name>.pkl
```

---

## 🧠 Models

### Classification  (12 models)

| Model | Hyperparameters Tuned |
|---|---|
| Logistic Regression | `C` |
| Decision Tree | `max_depth`, `min_samples_split` |
| Random Forest | `n_estimators`, `max_depth`, `class_weight` |
| Gradient Boosting | `n_estimators`, `learning_rate`, `max_depth` |
| SVM | `C`, `kernel` |
| K-Nearest Neighbors | `n_neighbors`, `weights` |
| Naive Bayes | `var_smoothing` |
| Extra Trees | `n_estimators`, `max_depth` |
| Bagging (DT base) | `n_estimators`, `max_samples` |
| AdaBoost | `n_estimators`, `learning_rate` |
| **XGBoost** | `n_estimators`, `learning_rate`, `max_depth` |
| **LightGBM** | `n_estimators`, `learning_rate`, `num_leaves` |

### Regression  (15 models)

| Model | Hyperparameters Tuned |
|---|---|
| Linear Regression | — |
| Ridge | `alpha` |
| Lasso | `alpha` |
| ElasticNet | `alpha`, `l1_ratio` |
| Bayesian Ridge | `alpha_init` |
| Decision Tree Regressor | `max_depth`, `min_samples_split` |
| Random Forest Regressor | `n_estimators`, `max_depth` |
| Gradient Boosting Regressor | `n_estimators`, `learning_rate`, `max_depth` |
| SVR | `C`, `kernel` |
| KNN Regressor | `n_neighbors`, `weights` |
| Extra Trees Regressor | `n_estimators`, `max_depth` |
| Bagging Regressor (DT base) | `n_estimators`, `max_samples` |
| AdaBoost Regressor | `n_estimators`, `learning_rate` |
| **XGBoost Regressor** | `n_estimators`, `learning_rate`, `max_depth` |
| **LightGBM Regressor** | `n_estimators`, `learning_rate`, `num_leaves` |

---

## 🏆 Smart Model Selection (Step 10b)

When multiple models tie on test score, the pipeline resolves it in two steps:

1. **Priority order** — simpler, more interpretable models win ties (Logistic Regression beats XGBoost if scores are equal)
2. **Cross-validation** — if CV scores differ by more than **0.005**, the higher CV model wins regardless of priority

Each selection is explained in the console output:

```
──────────────────────────────────────────────────────
  Selection Reason
──────────────────────────────────────────────────────
  ✔ Winner      : Gradient Boosting
  ✔ Why chosen  : Highest Test Accuracy (94.1%) – no tie.
  ✔ Model nature: COMPLEX – sequential boosting
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/techakash32/Automated-Machine-Learning-Model-Benchmark.git
cd Automated-Machine-Learning-Model-Benchmark
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or with `uv` (faster):

```bash
uv sync
```

### 3. Configure & run

Edit the config block at the top of `main.py`:

```python
CSV_PATH      = r"datasets/your_dataset.csv"
TARGET_COLUMN = "your_target_column"
```

Then run:

```bash
python main.py
```

### 4. Collect outputs

| File | Description |
|---|---|
| `cleaned_data.csv` | Preprocessed dataset ready for further use |
| `best_model_<Name>.pkl` | Best trained model (+ target scaler for regression) |

---

## 📦 Dependencies

```
numpy
pandas
scikit-learn
xgboost
lightgbm
imbalanced-learn
joblib
```

> Missing packages are **auto-installed** at runtime — you'll never hit an `ImportError`.

---

## 💡 Sample Console Output

```
████████████████████████████████████████████████████████████
 FULL ML PIPELINE (All Features, Minimal Output)
████████████████████████████████████████████████████████████

═══════════════════════════════════════════════════════════
 STEP 10: Training Models (Accuracy / R² only)
═══════════════════════════════════════════════════════════
  Logistic Regression            Accuracy =  85.3%  (2.1s)
  Decision Tree                  Accuracy =  88.7%  (0.8s)
  Random Forest                  Accuracy =  93.2%  (12.4s)
  Gradient Boosting              Accuracy =  94.1%  (38.2s)
  SVM                            Accuracy =  91.0%  (5.6s)
  KNN                            Accuracy =  89.4%  (1.2s)
  Naive Bayes                    Accuracy =  82.1%  (0.1s)
  Extra Trees                    Accuracy =  92.8%  (10.3s)
  Bagging (DT base)              Accuracy =  91.5%  (6.7s)
  AdaBoost                       Accuracy =  90.2%  (4.9s)
  LightGBM                       Accuracy =  93.9%  (9.1s)
  XGBoost                        Accuracy =  93.5%  (15.8s)

═══════════════════════════════════════════════════════════
 PIPELINE COMPLETE
═══════════════════════════════════════════════════════════
  Cleaned CSV  : cleaned_data.csv
  Task         : CLASSIFICATION
  Best Model   : Gradient Boosting
  Saved file   : best_model_Gradient_Boosting.pkl

 Done. ✅
```

---

## 🔁 Loading the Saved Model

```python
import pickle

# For classification
with open("best_model_Gradient_Boosting.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_new)

# For regression (model + target scaler are saved together)
with open("best_model_Random_Forest_Regressor.pkl", "rb") as f:
    model, target_scaler = pickle.load(f)

predictions_scaled = model.predict(X_new)
predictions = target_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()
```

---

## 🤝 Contributing

Contributions and feature requests are welcome!

1. Fork the repository
2. Create your branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push and open a Pull Request

---

## 👤 Author

**Akash** · [@techakash32](https://github.com/techakash32)

---

<div align="center">

If this project helped you, please consider giving it a ⭐ on [GitHub](https://github.com/techakash32/Automated-Machine-Learning-Model-Benchmark)!

</div>
