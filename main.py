import os, warnings, time, pickle
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
warnings.filterwarnings("ignore")

# Auto‑install missing packages
def _install(*pkgs):
    os.system(f"pip install {' '.join(pkgs)} --break-system-packages -q")

try:
    import xgboost as xgb
except ImportError:
    _install("xgboost")
    import xgboost as xgb
try:
    import lightgbm as lgb
except ImportError:
    _install("lightgbm")
    import lightgbm as lgb
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    _install("imbalanced-learn")
    try:
        from imblearn.over_sampling import SMOTE
        HAS_SMOTE = True
    except:
        HAS_SMOTE = False

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score

# ── Classification models ───────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, BaggingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# ── Regression models ───────────────────────────────────────
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, BaggingRegressor,
                              AdaBoostRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# ═══════════════════ CONFIGURATION ══════════════════════════
CSV_PATH            = r"E:\AI AGENTS\VS CODE AGENTS\10\datasets\Attrition - Attrition.csv"
TARGET_COLUMN       = "Attrition"
CLEANED_CSV_PATH    = "cleaned_data.csv"
TEST_SIZE           = 0.20
RANDOM_STATE        = 42
MISSING_THRESHOLD   = 0.5      # Drop columns with >50% missing
OUTLIER_IQR_FACTOR  = 1.5      # IQR multiplier for capping
IMBALANCE_THRESHOLD = 0.2      # Apply SMOTE if minority <20%
USE_SMOTE           = True
SCALE_TARGET        = True     # Scale target for regression (extra feature)

# ── Tie‑break priority (simpler models first) ─────────────────
TIE_BREAK_PRIORITY = [
    "Logistic Regression", "Linear Regression", "Ridge Regression",
    "Lasso Regression", "ElasticNet", "Bayesian Ridge", "Naive Bayes",
    "KNN", "Decision Tree", "Decision Tree Regressor", "SVM", "SVR",
    "Bagging (DT base)", "AdaBoost", "AdaBoost Regressor", "Extra Trees",
    "Extra Trees Regressor", "Gradient Boosting", "Gradient Boosting Regressor",
    "Random Forest", "Random Forest Regressor", "LightGBM", "LightGBM Regressor",
    "XGBoost", "XGBoost Regressor"
]

# Model complexity map for selection reason
MODEL_COMPLEXITY = {
    "Logistic Regression"         : ("simple",   "linear, highly interpretable"),
    "Linear Regression"           : ("simple",   "linear, highly interpretable"),
    "Ridge Regression"            : ("simple",   "regularised linear"),
    "Lasso Regression"            : ("simple",   "linear with feature selection"),
    "ElasticNet"                  : ("simple",   "combines Ridge + Lasso"),
    "Bayesian Ridge"              : ("simple",   "probabilistic linear model"),
    "Naive Bayes"                 : ("simple",   "probabilistic, very fast"),
    "KNN"                         : ("simple",   "instance‑based, no training"),
    "Decision Tree"               : ("moderate", "rule‑based, interpretable"),
    "Decision Tree Regressor"     : ("moderate", "rule‑based, interpretable"),
    "SVM"                         : ("moderate", "margin‑maximising"),
    "SVR"                         : ("moderate", "margin‑maximising regressor"),
    "Bagging (DT base)"           : ("moderate", "averages many DTs"),
    "AdaBoost"                    : ("moderate", "boosting ensemble"),
    "AdaBoost Regressor"          : ("moderate", "boosting ensemble"),
    "Extra Trees"                 : ("complex",  "randomised forest"),
    "Extra Trees Regressor"       : ("complex",  "randomised forest"),
    "Gradient Boosting"           : ("complex",  "sequential boosting"),
    "Gradient Boosting Regressor" : ("complex",  "sequential boosting"),
    "Random Forest"               : ("complex",  "bagged trees"),
    "Random Forest Regressor"     : ("complex",  "bagged trees"),
    "LightGBM"                    : ("complex",  "leaf‑wise gradient boosting"),
    "LightGBM Regressor"          : ("complex",  "leaf‑wise gradient boosting"),
    "XGBoost"                     : ("complex",  "regularised gradient boosting"),
    "XGBoost Regressor"           : ("complex",  "regularised gradient boosting"),
}
# ══════════════════════════════════════════════════════════════

def print_header(text):
    print("\n" + "═"*60)
    print(f"  {text}")
    print("═"*60)

def print_section(text):
    print(f"\n{'─'*50}\n  {text}\n{'─'*50}")

# ── Smart model selector (from reference file) ─────────────────
def select_best_model_with_reason(results, task, trained_models, X_tr, y_tr, X_te, y_te):
    if task == "classification":
        metric_key = "Test Accuracy"
        metric_label = "Test Accuracy"
        best_score = max(v[metric_key] for v in results.values())
    else:
        metric_key = "R² Score"
        metric_label = "R² Score"
        best_score = max(v[metric_key] for v in results.values())

    tied_models = [name for name, v in results.items() if v[metric_key] == best_score]

    if len(tied_models) == 1:
        best_name = tied_models[0]
        best_model = trained_models[best_name]
        complexity, desc = MODEL_COMPLEXITY.get(best_name, ("unknown", ""))
        print_section("Selection Reason")
        print(f"  ✔ Winner       : {best_name}")
        print(f"  ✔ Why chosen   : Highest {metric_label} ({best_score}) – no tie.")
        print(f"  ✔ Model nature : {complexity.upper()} – {desc}")
        return best_name, best_model

    # Tie handling
    print(f"\n  ⚠ Tie detected: {len(tied_models)} models have {metric_label} = {best_score}")
    # Priority order
    priority_winner = None
    for pref in TIE_BREAK_PRIORITY:
        if pref in tied_models:
            priority_winner = pref
            break
    if not priority_winner:
        priority_winner = tied_models[0]

    # CV among tied
    cv_scores = {}
    for name in tied_models:
        model = trained_models[name]
        scoring = "accuracy" if task == "classification" else "r2"
        scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring=scoring, n_jobs=-1)
        cv_scores[name] = scores.mean()
        print(f"   CV ({name}): {scores.mean():.4f} ± {scores.std():.4f}")

    best_cv_model = max(cv_scores, key=cv_scores.get)
    diff = abs(cv_scores[best_cv_model] - cv_scores.get(priority_winner, 0))
    if diff > 0.005:
        final_winner = best_cv_model
        reason = f"CV score significantly higher ({cv_scores[best_cv_model]:.4f} vs {cv_scores.get(priority_winner,0):.4f})"
    else:
        final_winner = priority_winner
        reason = f"CV scores nearly equal – simpler model preferred"

    best_model = trained_models[final_winner]
    complexity, desc = MODEL_COMPLEXITY.get(final_winner, ("unknown", ""))
    print_section("Selection Reason (Tie-Break)")
    print(f"  ✔ Winner       : {final_winner}")
    print(f"  ✔ Tie decision : {reason}")
    print(f"  ✔ Model nature : {complexity.upper()} – {desc}")
    return final_winner, best_model

# ── Main pipeline ─────────────────────────────────────────────
def run_pipeline():
    print("█"*60)
    print("  FULL ML PIPELINE (All Features, Minimal Output)")
    print("█"*60)

    # ========== STEP 1: Load data ==========
    print_header("STEP 1: Load Dataset")
    df = pd.read_csv(CSV_PATH)
    print(f"  Shape: {df.shape}")
    print(f"  First 3 rows:\n{df.head(3).to_string()}")

    if TARGET_COLUMN not in df.columns:
        print(f"❌ Target '{TARGET_COLUMN}' not found.")
        return

    # ========== STEP 2: Remove irrelevant columns ==========
    print_header("STEP 2: Remove Irrelevant Columns")
    initial_cols = df.shape[1]
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if df[col].isnull().mean() > MISSING_THRESHOLD:
            print(f"  Dropping high-missing: '{col}'")
            df.drop(columns=[col], inplace=True)
        elif df[col].nunique() <= 1:
            print(f"  Dropping constant: '{col}'")
            df.drop(columns=[col], inplace=True)
        elif df[col].nunique() == len(df):
            print(f"  Dropping ID column: '{col}'")
            df.drop(columns=[col], inplace=True)
    print(f"  Removed {initial_cols - df.shape[1]} columns. Remaining: {df.shape[1]}")

    # ========== STEP 3: Column identification ==========
    print_header("STEP 3: Identify Numerical / Categorical")
    num_cols = []
    cat_cols = []
    for col in df.columns:
        if col == TARGET_COLUMN:
            continue
        if is_numeric_dtype(df[col]):
            if df[col].nunique() <= 10:
                cat_cols.append(col)
                print(f"  '{col}' (numeric, {df[col].nunique()} unique) → categorical")
            else:
                num_cols.append(col)
        else:
            cat_cols.append(col)
    print(f"  Numerical : {len(num_cols)} | Categorical : {len(cat_cols)}")

    # ========== STEP 4: Missing value handling ==========
    print_header("STEP 4: Missing Values")
    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        df[num_cols] = imp_num.fit_transform(df[num_cols])
        print("  Numerical → median")
    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
        print("  Categorical → mode")

    # ========== STEP 5: Outlier capping ==========
    print_header("STEP 5: Outlier Handling (IQR)")
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - OUTLIER_IQR_FACTOR*IQR, Q3 + OUTLIER_IQR_FACTOR*IQR
        df[col] = df[col].clip(lower, upper)
    print(f"  Capped outliers in {len(num_cols)} numerical columns (IQR factor={OUTLIER_IQR_FACTOR})")

    # ========== STEP 6: Encoding & scaling ==========
    print_header("STEP 6: Encoding & Scaling")
    # Target encoding if string
    if not is_numeric_dtype(df[TARGET_COLUMN]):
        le_target = LabelEncoder()
        df[TARGET_COLUMN] = le_target.fit_transform(df[TARGET_COLUMN].astype(str))
        print("  Target encoded (string → int)")
    else:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Encode categoricals
    for col in cat_cols[:]:
        n = df[col].nunique()
        if n <= 10:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            print(f"  OneHotEncoded: '{col}' → {n-1} dummies")
        else:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
            print(f"  LabelEncoded: '{col}' ({n} unique)")

    # Scale numerical features
    final_num = df.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COLUMN in final_num:
        final_num.remove(TARGET_COLUMN)
    if final_num:
        scaler = StandardScaler()
        df[final_num] = scaler.fit_transform(df[final_num])
        print(f"  StandardScaler applied on {len(final_num)} features")

    # ========== STEP 7: Export cleaned CSV ==========
    print_header("STEP 7: Export Cleaned CSV")
    df.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"  Saved to: {CLEANED_CSV_PATH}")

    # ========== STEP 8: Determine task type ==========
    print_header("STEP 8: Task Detection")
    n_unique = df[TARGET_COLUMN].nunique()
    if n_unique <= 20:
        TASK = "classification"
        print(f"  Target unique values = {n_unique} → CLASSIFICATION")
    else:
        TASK = "regression"
        print(f"  Target unique values = {n_unique} → REGRESSION")

    # ========== Prepare X, y and split ==========
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values
    if TASK == "classification":
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                                  random_state=RANDOM_STATE)
    print(f"\n  Train size: {X_tr.shape[0]} | Test size: {X_te.shape[0]} | Features: {X_tr.shape[1]}")

    # ========== STEP 9: SMOTE for classification ==========
    if TASK == "classification" and USE_SMOTE and HAS_SMOTE:
        from collections import Counter
        cnt = Counter(y_tr)
        minority_ratio = min(cnt.values()) / max(cnt.values())
        print_header("STEP 9: Class Imbalance (SMOTE)")
        print(f"  Class distribution: {dict(cnt)} | Minority ratio = {minority_ratio:.3f}")
        if minority_ratio < IMBALANCE_THRESHOLD:
            k = min(5, min(cnt.values())-1)
            if k >= 1:
                sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                print(f"  SMOTE applied. New distribution: {Counter(y_tr)}")
            else:
                print("  SMOTE skipped – minority class too small for k_neighbors")
        else:
            print("  Data already balanced – SMOTE skipped.")
    elif TASK == "classification":
        print_header("STEP 9: SMOTE")
        print("  SMOTE disabled (USE_SMOTE=False) or not available.")

    # ========== STEP 10: Models and hyperparameter grids ==========
    # (only the grids that are used; simpler than reference file but effective)
    if TASK == "classification":
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                {"C": [0.01, 0.1, 1, 10]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                {"max_depth": [3,5,8,None], "min_samples_split": [2,5,10]}),
            "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE),
                {"n_estimators": [100,200], "max_depth": [5,10,None], "class_weight":[None,"balanced"]}),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "max_depth":[3,5]}),
            "SVM": (SVC(random_state=RANDOM_STATE),
                {"C":[0.1,1,10], "kernel":["rbf","linear"]}),
            "KNN": (KNeighborsClassifier(),
                {"n_neighbors":[3,5,7,11], "weights":["uniform","distance"]}),
            "Naive Bayes": (GaussianNB(),
                {"var_smoothing":[1e-9,1e-8,1e-7]}),
            "Extra Trees": (ExtraTreesClassifier(random_state=RANDOM_STATE),
                {"n_estimators":[100,200], "max_depth":[5,10,None]}),
            "Bagging (DT base)": (BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=RANDOM_STATE),
                {"n_estimators":[50,100], "max_samples":[0.7,1.0]}),
            "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE),
                {"n_estimators":[50,100], "learning_rate":[0.5,1.0]}),
        }
        if 'lightgbm' in globals():
            models["LightGBM"] = (lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
                {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "num_leaves":[31,63]})
        if 'xgboost' in globals():
            models["XGBoost"] = (xgb.XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                                                    use_label_encoder=False),
                {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "max_depth":[3,5,7]})
        cv_method = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring_metric = "accuracy"
    else:  # regression
        models = {
            "Linear Regression": (LinearRegression(), {}),
            "Ridge Regression": (Ridge(random_state=RANDOM_STATE),
                {"alpha":[0.1,1,10,100]}),
            "Lasso Regression": (Lasso(max_iter=5000),
                {"alpha":[0.001,0.01,0.1,1]}),
            "ElasticNet": (ElasticNet(max_iter=5000),
                {"alpha":[0.01,0.1,1], "l1_ratio":[0.3,0.5,0.7]}),
            "Bayesian Ridge": (BayesianRidge(),
                {"alpha_init":[1e-6,1e-5,1e-4]}),
            "Decision Tree Regressor": (DecisionTreeRegressor(random_state=RANDOM_STATE),
                {"max_depth":[3,5,8,None], "min_samples_split":[2,5,10]}),
            "Random Forest Regressor": (RandomForestRegressor(random_state=RANDOM_STATE),
                {"n_estimators":[100,200], "max_depth":[5,10,None]}),
            "Gradient Boosting Regressor": (GradientBoostingRegressor(random_state=RANDOM_STATE),
                {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "max_depth":[3,5]}),
            "SVR": (SVR(),
                {"C":[0.1,1,10], "kernel":["rbf","linear"]}),
            "KNN": (KNeighborsRegressor(),
                {"n_neighbors":[3,5,7,11], "weights":["uniform","distance"]}),
            "Extra Trees Regressor": (ExtraTreesRegressor(random_state=RANDOM_STATE),
                {"n_estimators":[100,200], "max_depth":[5,10,None]}),
            "Bagging (DT base)": (BaggingRegressor(estimator=DecisionTreeRegressor(), random_state=RANDOM_STATE),
                {"n_estimators":[50,100], "max_samples":[0.7,1.0]}),
            "AdaBoost Regressor": (AdaBoostRegressor(random_state=RANDOM_STATE),
                {"n_estimators":[50,100], "learning_rate":[0.5,1.0]}),
        }
        if 'lightgbm' in globals():
            models["LightGBM Regressor"] = (lgb.LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
                {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "num_leaves":[31,63]})
        if 'xgboost' in globals():
            models["XGBoost Regressor"] = (xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
                {"n_estimators":[100,200], "learning_rate":[0.01,0.1], "max_depth":[3,5,7]})
        cv_method = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring_metric = "r2"

    # Optional target scaling for regression (extra feature)
    target_scaler = None
    if TASK == "regression" and SCALE_TARGET:
        target_scaler = StandardScaler()
        y_tr = target_scaler.fit_transform(y_tr.reshape(-1,1)).ravel()
        y_te_scaled = target_scaler.transform(y_te.reshape(-1,1)).ravel()
        print("\n  Target scaling applied (StandardScaler on y_train)")
    else:
        y_te_scaled = y_te

    # ========== Train and evaluate (minimal output) ==========
    print_header("STEP 10: Training Models (Accuracy / R² only)")
    results = {}
    trained_models = {}
    for name, (model, grid) in models.items():
        t0 = time.time()
        try:
            if grid:
                gs = GridSearchCV(model, grid, cv=cv_method, scoring=scoring_metric, n_jobs=-1, refit=True)
                gs.fit(X_tr, y_tr)
                best = gs.best_estimator_
            else:
                best = model
                best.fit(X_tr, y_tr)
            elapsed = time.time() - t0

            y_pred_scaled = best.predict(X_te)
            if TASK == "regression" and SCALE_TARGET and target_scaler is not None:
                y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
                r2 = r2_score(y_te, y_pred) * 100
                print(f"  {name:30s}  R² = {r2:5.1f}%  ({elapsed:.1f}s)")
                results[name] = {"R² Score": r2}
            elif TASK == "regression":
                r2 = r2_score(y_te, y_pred_scaled) * 100
                print(f"  {name:30s}  R² = {r2:5.1f}%  ({elapsed:.1f}s)")
                results[name] = {"R² Score": r2}
            else:
                acc = accuracy_score(y_te, y_pred_scaled) * 100
                print(f"  {name:30s}  Accuracy = {acc:5.1f}%  ({elapsed:.1f}s)")
                results[name] = {"Test Accuracy": acc}
            trained_models[name] = best
        except Exception as e:
            print(f"  ✗ {name:30s} failed: {e}")

    # ========== STEP 10b: Smart model selection ==========
    print_header("STEP 10b: Smart Model Selection")
    best_name, best_model = select_best_model_with_reason(
        results, TASK, trained_models, X_tr, y_tr, X_te, y_te_scaled if TASK=="regression" and SCALE_TARGET else y_te
    )

    # ========== STEP 11: Save best model ==========
    print_header("STEP 11: Save Best Model")
    safe_name = best_name.replace(" ", "_").replace("(", "").replace(")", "")
    pkl_path = f"best_model_{safe_name}.pkl"
    if TASK == "regression" and SCALE_TARGET and target_scaler is not None:
        with open(pkl_path, "wb") as f:
            pickle.dump((best_model, target_scaler), f)
        print(f"  Model + TargetScaler saved → {pkl_path}")
    else:
        with open(pkl_path, "wb") as f:
            pickle.dump(best_model, f)
        print(f"  Model saved → {pkl_path}")

    print_header("PIPELINE COMPLETE")
    print(f"  Cleaned CSV   : {CLEANED_CSV_PATH}")
    print(f"  Task          : {TASK.upper()}")
    print(f"  Best Model    : {best_name}")
    print(f"  Saved file    : {pkl_path}")
    print("\n  Done. ✅")

if __name__ == "__main__":
    run_pipeline()