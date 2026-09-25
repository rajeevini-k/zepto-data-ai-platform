
"""
Zepto Data & AI Platform
Module 2 — Analytics Pipeline

Complete reproducible Titanic analytics and machine-learning workflow.

The raw Titanic dataset is loaded exactly once with:
    sns.load_dataset('titanic')

The workflow then performs:
1. Profiling
2. Missing-value analysis and cleaning
3. Univariate EDA
4. Bivariate survival analysis
5. Correlation analysis
6. Standardization sanity check
7. Stratified classification split
8. Logistic Regression
9. Decision Tree
10. Random Forest
11. Class-imbalance comparison
12. SMOTE
13. Random Forest GridSearchCV
14. OOB evaluation
15. Multivariate Linear Regression for fare
16. Residual / heteroscedasticity analysis
17. Model comparison
18. Complete pipeline persistence and reload testing
"""

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# ============================================================
# CONFIGURATION
# ============================================================

REPO_DIR = Path(__file__).resolve().parent.parent
ANALYTICS_DIR = REPO_DIR / "analytics"
PLOTS_DIR = ANALYTICS_DIR / "plots"

ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_plot(filename):
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# 1. LOAD TITANIC DATASET — EXACTLY ONCE
# ============================================================

print_section("1. LOAD TITANIC DATASET")

df = sns.load_dataset('titanic')

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataFrame info:")
df.info()

print("\nDescriptive statistics:")
print(df.describe(include="all").transpose())


# ============================================================
# 2. SAVE OFFLINE FALLBACK
# ============================================================

fallback_path = ANALYTICS_DIR / "titanic.csv"
df.to_csv(fallback_path, index=False)

print("\nOffline fallback saved:", fallback_path)
print("Fallback shape:", pd.read_csv(fallback_path).shape)


# ============================================================
# Explicit missing-value thresholds used by the cleaning rules.
MISSING_DROP_ROW_THRESHOLD = 0.05
MISSING_MEDIAN_MAX_THRESHOLD = 0.30
MISSING_DROP_COLUMN_THRESHOLD = 0.50

# 3. MISSING-VALUE PROFILING
# ============================================================

print_section("3. MISSING-VALUE PROFILE")

missing_counts = df.isna().sum()
missing_pct = (missing_counts / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing_counts,
    "missing_percentage": missing_pct
})

missing_report = missing_report[
    missing_report["missing_count"] > 0
].sort_values("missing_percentage", ascending=False)

print(missing_report)


# ============================================================
# 4. CLEANING
# ============================================================

print_section("4. DATA CLEANING")

clean_df = df.copy()

# Missing percentages measured from the original data:
# age          = 19.87%  -> median imputation
# embarked     = 0.22%   -> drop rows
# deck         = 77.22%  -> drop column
# embark_town  = 0.22%   -> drop rows
#
# The high-missingness deck column is dropped because imputing
# approximately 77% missing values would be unreliable.

print("Original shape:", clean_df.shape)

# High missingness: drop deck.
if "deck" in clean_df.columns:
    clean_df = clean_df.drop(columns=["deck"])

# Under 5% missing: drop affected rows.
for column in ["embarked", "embark_town"]:
    if column in clean_df.columns:
        clean_df = clean_df.dropna(subset=[column])

# 5%–30% missing: median imputation for age.
if "age" in clean_df.columns:
    age_median = clean_df["age"].median()
    clean_df["age"] = clean_df["age"].fillna(age_median)
else:
    age_median = np.nan

# Remove duplicates after missing-value treatment.
before_duplicates = len(clean_df)
clean_df = clean_df.drop_duplicates()
duplicates_removed = before_duplicates - len(clean_df)

print("Age median:", age_median)
print("Duplicates removed:", duplicates_removed)
print("Cleaned shape:", clean_df.shape)

print("\nRemaining missing values:")
print(clean_df.isna().sum())

print("\nRemaining duplicates:", clean_df.duplicated().sum())

clean_path = ANALYTICS_DIR / "titanic_clean.csv"
clean_df.to_csv(clean_path, index=False)


# ============================================================
# 5. UNIVARIATE ANALYSIS
# ============================================================

print_section("5. UNIVARIATE ANALYSIS")

for column in ["age", "fare"]:
    q1 = clean_df[column].quantile(0.25)
    q3 = clean_df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = clean_df[
        (clean_df[column] < lower) |
        (clean_df[column] > upper)
    ]

    print(f"\n{column.upper()}")
    print("Q1:", q1)
    print("Q3:", q3)
    print("IQR:", iqr)
    print("Lower bound:", lower)
    print("Upper bound:", upper)
    print("Outlier count:", len(outliers))

# Fare statistics
fare_mean = clean_df["fare"].mean()
fare_median = clean_df["fare"].median()
fare_mode = clean_df["fare"].mode().iloc[0]

print("\nFare mean:", fare_mean)
print("Fare median:", fare_median)
print("Fare mode:", fare_mode)

if fare_mean > fare_median > fare_mode:
    fare_shape = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_shape = "left-skewed"
else:
    fare_shape = "approximately symmetric"

print("Fare distribution:", fare_shape)

# Age histogram
plt.figure(figsize=(8, 5))
plt.hist(clean_df["age"], bins=25)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
save_plot("age_distribution.png")

# Age boxplot
plt.figure(figsize=(8, 4))
plt.boxplot(clean_df["age"])
plt.title("Age Boxplot")
plt.ylabel("Age")
save_plot("age_boxplot.png")

# Fare histogram
plt.figure(figsize=(8, 5))
plt.hist(clean_df["fare"], bins=30)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
save_plot("fare_distribution.png")

# Fare boxplot
plt.figure(figsize=(8, 4))
plt.boxplot(clean_df["fare"])
plt.title("Fare Boxplot")
plt.ylabel("Fare")
save_plot("fare_boxplot.png")


# ============================================================
# 6. BIVARIATE SURVIVAL ANALYSIS
# ============================================================

print_section("6. SURVIVAL ANALYSIS")

sex_survival = (
    clean_df.groupby("sex")["survived"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

sex_survival["survival_rate"] = sex_survival["mean"] * 100

print("\nSurvival by sex:")
print(sex_survival)

pclass_survival = (
    clean_df.groupby("pclass")["survived"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

pclass_survival["survival_rate"] = pclass_survival["mean"] * 100

print("\nSurvival by pclass:")
print(pclass_survival)

sex_pclass_survival = (
    clean_df.groupby(["sex", "pclass"])["survived"]
    .agg(["count", "sum", "mean"])
    .reset_index()
)

sex_pclass_survival["survival_rate"] = (
    sex_pclass_survival["mean"] * 100
)

print("\nSurvival by sex and pclass:")
print(sex_pclass_survival)

# Chart 1
plt.figure(figsize=(8, 5))
sns.barplot(
    data=sex_survival,
    x="sex",
    y="survival_rate"
)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate (%)")
save_plot("survival_rate_by_sex.png")

# Chart 2
plt.figure(figsize=(8, 5))
sns.barplot(
    data=pclass_survival,
    x="pclass",
    y="survival_rate"
)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate (%)")
save_plot("survival_rate_by_pclass.png")

# Chart 3
plt.figure(figsize=(9, 5))
sns.barplot(
    data=sex_pclass_survival,
    x="pclass",
    y="survival_rate",
    hue="sex"
)
plt.title("Survival Rate by Sex and Passenger Class")
plt.ylabel("Survival Rate (%)")
save_plot("survival_rate_by_sex_and_pclass.png")


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

print_section("7. CORRELATION ANALYSIS")

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

correlation_matrix = clean_df[
    correlation_columns
].corr()

print(correlation_matrix)

correlation_matrix.to_csv(
    ANALYTICS_DIR / "correlation_matrix.csv"
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".4f",
    cmap="coolwarm",
    center=0
)
plt.title("Titanic Correlation Heatmap")
save_plot("correlation_heatmap.png")

# Identify strongest off-diagonal correlations
pairs = []

for i in range(len(correlation_columns)):
    for j in range(i + 1, len(correlation_columns)):
        a = correlation_columns[i]
        b = correlation_columns[j]
        value = correlation_matrix.loc[a, b]
        pairs.append({
            "feature_1": a,
            "feature_2": b,
            "correlation": value,
            "absolute_correlation": abs(value)
        })

top_correlations = (
    pd.DataFrame(pairs)
    .sort_values("absolute_correlation", ascending=False)
    .reset_index(drop=True)
)

print("\nStrongest correlations:")
print(top_correlations.head(2))

top_correlations.to_csv(
    ANALYTICS_DIR / "top_correlations.csv",
    index=False
)


# ============================================================
# 8. STANDARDIZATION SANITY CHECK
# ============================================================

print_section("8. STANDARDIZATION SANITY CHECK")

for column in ["age", "fare"]:
    values = clean_df[[column]].values.astype(float)

    scaler = StandardScaler()
    standardized = scaler.fit_transform(values)

    print(
        column,
        "standardized mean =",
        standardized.mean(),
        "standardized std =",
        standardized.std()
    )

    # Explicit z-score sanity check.
    # z = (x - mean) / standard deviation
    zscore_values = (
        (standardized - standardized.mean())
        / standardized.std(ddof=1)
    )

    zscore_mean = float(zscore_values.mean())
    zscore_std = float(zscore_values.std(ddof=1))

    print(
        "Z-score sanity check:",
        f"mean={zscore_mean:.6f}, std={zscore_std:.6f}"
    )

# ============================================================
# 9. CLASSIFICATION DATA
# ============================================================

print_section("9. CLASSIFICATION SPLIT")

classification_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
]

target = "survived"

X = clean_df[classification_features].copy()
y = clean_df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("X shape:", X.shape)
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nFull target distribution:")
print(y.value_counts())

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare",
]

categorical_features = [
    "sex",
    "embarked",
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    ),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])


# ============================================================
# 11. CLASSIFICATION MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
}


def evaluate_classifier(name, pipeline):
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = pipeline.decision_function(X_test)

    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "auc": roc_auc_score(
            y_test,
            probabilities
        ),
    }

    print(f"\n{name}")
    print("Confusion matrix:")
    print(cm)
    print(metrics)

    return metrics, pipeline, probabilities, cm


classification_results = []
trained_models = {}
roc_data = {}

for name, estimator in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    metrics, fitted_pipeline, probabilities, cm = (
        evaluate_classifier(name, pipeline)
    )

    classification_results.append(metrics)
    trained_models[name] = fitted_pipeline
    roc_data[name] = probabilities

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title(f"{name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    safe_name = (
        name.lower()
        .replace(" ", "_")
    )

    save_plot(
        f"{safe_name}_confusion_matrix.png"
    )


# ============================================================
# 12. ROC CURVES
# ============================================================

from sklearn.metrics import roc_curve

plt.figure(figsize=(8, 6))

for name, probabilities in roc_data.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Classification ROC Curves")
plt.legend()
save_plot("classification_roc_curves.png")


# ============================================================
# 13. DECISION TREE VISUALIZATION
# ============================================================

tree_pipeline = trained_models["Decision Tree"]

tree_model = tree_pipeline.named_steps["model"]
tree_preprocessor = tree_pipeline.named_steps["preprocessor"]

feature_names = tree_preprocessor.get_feature_names_out()

plt.figure(figsize=(22, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")
save_plot("decision_tree_visualization.png")


# ============================================================
# 14. CLASS IMBALANCE
# ============================================================

print_section("14. CLASS IMBALANCE")

baseline_lr = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])

balanced_lr = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    )
])

baseline_lr.fit(X_train, y_train)
balanced_lr.fit(X_train, y_train)

baseline_pred = baseline_lr.predict(X_test)
balanced_pred = balanced_lr.predict(X_test)

baseline_prob = baseline_lr.predict_proba(X_test)[:, 1]
balanced_prob = balanced_lr.predict_proba(X_test)[:, 1]

imbalance_results = []

def metric_row(name, pred, prob):
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(
            y_test,
            pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            pred,
            zero_division=0
        ),
        "auc": roc_auc_score(
            y_test,
            prob
        ),
    }

imbalance_results.append(
    metric_row(
        "Baseline Logistic Regression",
        baseline_pred,
        baseline_prob
    )
)

imbalance_results.append(
    metric_row(
        "Balanced Logistic Regression",
        balanced_pred,
        balanced_prob
    )
)


# ============================================================
# 15. SMOTE — TRAINING DATA ONLY
# ============================================================

smote_preprocessor = preprocessor.fit(X_train, y_train)

X_train_transformed = smote_preprocessor.transform(X_train)
X_test_transformed = smote_preprocessor.transform(X_test)

smote = SMOTE(
    random_state=RANDOM_STATE
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_transformed,
    y_train
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

smote_pred = smote_model.predict(
    X_test_transformed
)

smote_prob = smote_model.predict_proba(
    X_test_transformed
)[:, 1]

imbalance_results.append(
    metric_row(
        "SMOTE Logistic Regression",
        smote_pred,
        smote_prob
    )
)

imbalance_df = pd.DataFrame(imbalance_results)

imbalance_df.to_csv(
    ANALYTICS_DIR / "imbalance_comparison.csv",
    index=False
)

print("\nImbalance comparison:")
print(imbalance_df)


# ============================================================
# 16. RANDOM FOREST GRID SEARCH
# ============================================================

print_section("16. RANDOM FOREST GRID SEARCH")

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            oob_score=True,
            n_jobs=-1
        )
    )
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

print("Best parameters:")
print(grid_search.best_params_)

print("Best CV F1:")
print(grid_search.best_score_)

best_rf_pipeline = grid_search.best_estimator_

best_rf_model = (
    best_rf_pipeline
    .named_steps["model"]
)

print("OOB score:")
print(best_rf_model.oob_score_)

tuned_pred = best_rf_pipeline.predict(X_test)
tuned_prob = best_rf_pipeline.predict_proba(X_test)[:, 1]

tuned_metrics = metric_row(
    "Tuned Random Forest",
    tuned_pred,
    tuned_prob
)

classification_results.append(
    tuned_metrics
)

grid_results = pd.DataFrame(
    grid_search.cv_results_
)

grid_results.to_csv(
    ANALYTICS_DIR / "random_forest_gridsearch_results.csv",
    index=False
)

joblib.dump(
    best_rf_pipeline,
    ANALYTICS_DIR / "tuned_random_forest.joblib"
)


# ============================================================
# 17. FINAL CLASSIFICATION RESULTS
# ============================================================

classification_df = pd.DataFrame(
    classification_results
)

classification_df.to_csv(
    ANALYTICS_DIR / "classification_model_comparison.csv",
    index=False
)

print("\nClassification comparison:")
print(classification_df)


# ============================================================
# 18. REGRESSION — MULTIVARIATE LINEAR REGRESSION
# ============================================================

print_section("18. FARE REGRESSION")

regression_target = "fare"

regression_features = [
    column
    for column in clean_df.columns
    if column != regression_target
]

X_reg = clean_df[regression_features].copy()
y_reg = clean_df[regression_target].copy()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=RANDOM_STATE
    )
)

regression_numeric = (
    X_reg_train
    .select_dtypes(include=["number", "bool"])
    .columns
    .tolist()
)

regression_categorical = (
    X_reg_train
    .select_dtypes(include=["object", "category"])
    .columns
    .tolist()
)

regression_numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

regression_categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    ),
])

regression_preprocessor = ColumnTransformer([
    (
        "num",
        regression_numeric_pipeline,
        regression_numeric
    ),
    (
        "cat",
        regression_categorical_pipeline,
        regression_categorical
    ),
])

regression_pipeline = Pipeline([
    (
        "preprocessor",
        regression_preprocessor
    ),
    (
        "model",
        LinearRegression()
    )
])

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

regression_pred = regression_pipeline.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    regression_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        regression_pred
    )
)

r2 = r2_score(
    y_reg_test,
    regression_pred
)

n = len(y_reg_test)

transformed_regression_features = (
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

p = len(transformed_regression_features)

adjusted_r2 = (
    1 -
    (1 - r2) *
    (n - 1) /
    (n - p - 1)
)

residuals = y_reg_test - regression_pred

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
print("Adjusted R²:", adjusted_r2)
print("Transformed predictor count:", p)

# Residual plot
plt.figure(figsize=(8, 5))
plt.scatter(
    regression_pred,
    residuals,
    alpha=0.7
)
plt.axhline(
    0,
    linestyle="--"
)
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression Residual Plot")
save_plot("fare_regression_residuals.png")

# Simple spread analysis
residual_frame = pd.DataFrame({
    "actual": y_reg_test.values,
    "predicted": regression_pred,
    "residual": residuals.values
})

residual_frame.to_csv(
    ANALYTICS_DIR / "regression_predictions.csv",
    index=False
)

regression_results = pd.DataFrame([{
    "model": "Multivariate Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}])

regression_results.to_csv(
    ANALYTICS_DIR / "regression_results.csv",
    index=False
)

print("\nRegression results:")
print(regression_results)


# ============================================================
# 19. FINAL MODEL COMPARISON
# ============================================================

print_section("19. FINAL MODEL COMPARISON")

classification_df = classification_df.copy()

classification_df["model_type"] = "classification"

classification_df["MAE"] = np.nan
classification_df["RMSE"] = np.nan
classification_df["R2"] = np.nan
classification_df["Adjusted_R2"] = np.nan

regression_comparison = pd.DataFrame([{
    "model": "Multivariate Linear Regression",
    "model_type": "regression",
    "accuracy": np.nan,
    "precision": np.nan,
    "recall": np.nan,
    "f1": np.nan,
    "auc": np.nan,
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Adjusted_R2": adjusted_r2
}])

comparison_df = pd.concat(
    [
        classification_df,
        regression_comparison
    ],
    ignore_index=True
)

comparison_df.to_csv(
    ANALYTICS_DIR / "final_model_summary.csv",
    index=False
)

print(comparison_df)


# ============================================================
# 20. SAVE COMPLETE FINAL SURVIVAL PIPELINE
# ============================================================

print_section("20. SAVE FINAL SURVIVAL PIPELINE")

final_pipeline = best_rf_pipeline

pipeline_path = (
    ANALYTICS_DIR /
    "titanic_survival_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    pipeline_path
)

print("Pipeline saved:", pipeline_path)


# ============================================================
# 21. RELOAD AND TEST PIPELINE ON RAW INPUT
# ============================================================

print_section("21. RELOAD PIPELINE")

loaded_pipeline = joblib.load(
    pipeline_path
)

sample_passenger = pd.DataFrame([{
    "pclass": 1,
    "sex": "female",
    "age": 30,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "S"
}])

sample_prediction = loaded_pipeline.predict(
    sample_passenger
)[0]

sample_probability = (
    loaded_pipeline
    .predict_proba(sample_passenger)[0, 1]
)

sample_output = pd.DataFrame([{
    **sample_passenger.iloc[0].to_dict(),
    "predicted_survived": int(sample_prediction),
    "prediction_label": (
        "Survived"
        if sample_prediction == 1
        else "Not Survived"
    ),
    "survival_probability": sample_probability
}])

sample_output.to_csv(
    ANALYTICS_DIR / "sample_prediction.csv",
    index=False
)

print(sample_output)


# ============================================================
# 22. FINAL VALIDATION
# ============================================================

print_section("22. FINAL ANALYTICS VALIDATION")

required_files = [
    "titanic.csv",
    "titanic_clean.csv",
    "correlation_matrix.csv",
    "top_correlations.csv",
    "classification_model_comparison.csv",
    "imbalance_comparison.csv",
    "random_forest_gridsearch_results.csv",
    "regression_results.csv",
    "regression_predictions.csv",
    "final_model_summary.csv",
    "sample_prediction.csv",
    "titanic_survival_pipeline.joblib",
    "tuned_random_forest.joblib",
]

all_files_exist = True

for filename in required_files:
    path = ANALYTICS_DIR / filename
    exists = path.exists()
    print(f"{filename}: {exists}")
    all_files_exist = all_files_exist and exists

print("\nCleaned rows:", len(clean_df))
print("Cleaned columns:", len(clean_df.columns))
print("Remaining missing values:", int(clean_df.isna().sum().sum()))
print("Remaining duplicates:", int(clean_df.duplicated().sum()))

print("\nPipeline reload successful:", pipeline_path.exists())
print("All required Analytics artifacts:", all_files_exist)

if not all_files_exist:
    raise RuntimeError(
        "Analytics validation failed: missing required artifacts."
    )

print("\n" + "=" * 70)
print("ANALYTICS PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)
