# Module 2 — Analytics

## Overview

This module performs an end-to-end analytics and machine-learning workflow using the Titanic dataset.

The workflow covers:

1. Dataset loading and fallback creation
2. Data profiling and quality assessment
3. Defensible missing-value treatment
4. Duplicate removal
5. Exploratory data analysis
6. Univariate statistics and outlier analysis
7. Survival-rate analysis
8. Correlation analysis
9. Classification modeling
10. Class-imbalance experiments
11. Random Forest hyperparameter tuning
12. Out-of-bag evaluation
13. Fare regression
14. Residual analysis and heteroscedasticity assessment
15. Final model comparison
16. Complete pipeline persistence and reload testing

---

# 1. Dataset

The Titanic dataset was loaded using Seaborn's built-in loader:

```python
sns.load_dataset("titanic")
```

The original dataset contained:

- 891 rows
- 15 columns

The raw dataset was saved as:

```text
analytics/titanic.csv
```

This provides a local fallback copy for reproducibility if the external dataset loader is unavailable during a later run.

The cleaned dataset is stored as:

```text
analytics/titanic_clean.csv
```

---

# 2. Data Cleaning

The original Titanic dataset contained missing values in:

| Column | Missing | Missing % |
|---|---:|---:|
| age | 177 | 19.87% |
| embarked | 2 | 0.22% |
| deck | 688 | 77.22% |
| embark_town | 2 | 0.22% |

## deck

`deck` had approximately 77.22% missing values.

Decision: drop the column.

## embarked and embark_town

Both columns had approximately 0.22% missing values.

Decision: remove the affected rows. Two rows were removed.

## age

`age` had approximately 19.87% missing values.

Decision: median imputation.

Median age used: **28.0**.

## Duplicate rows

Duplicate rows were removed after the missing-value treatment.

Final cleaned dataset:

- 773 rows
- 14 columns
- 0 missing values
- 0 duplicate rows

The cleaned dataset is saved as:

```text
analytics/titanic_clean.csv
```

---

# 3. Exploratory Data Analysis (EDA)

## 3.1 Fare statistics

| Statistic | Value |
|---|---:|
| Mean | 34.7617 |
| Median | 15.9000 |
| Mode | 13.0000 |

The mean is substantially greater than the median, indicating a right-skewed fare distribution.

Fare IQR analysis:

- Q1 = 8.05
- Q3 = 33.50
- IQR = 25.45
- Lower bound = -30.125
- Upper bound = 71.675
- 100 fare observations were outside the IQR-based bounds.

These observations were retained because high fares can represent genuine passenger observations rather than data errors.

Artifacts:

```text
analytics/plots/fare_distribution.png
analytics/plots/fare_boxplot.png
```

## 3.2 Age statistics and outliers

Age IQR analysis:

- Q1 = 21
- Q3 = 36
- IQR = 15
- Lower bound = -1.5
- Upper bound = 58.5
- 26 age observations were outside the IQR-based bounds.

These observations were retained.

Artifacts:

```text
analytics/plots/age_distribution.png
analytics/plots/age_boxplot.png
```

---

# 4. Survival Analysis

## 4.1 Survival by sex

| Sex | Passengers | Survivors | Survival Rate |
|---|---:|---:|---:|
| Female | 290 | 214 | 73.79% |
| Male | 483 | 104 | 21.53% |

The cleaned data shows a substantially higher observed survival rate among female passengers than male passengers.

Artifact:

```text
analytics/plots/survival_rate_by_sex.png
```

## 4.2 Survival by passenger class

| Passenger Class | Passengers | Survivors | Survival Rate |
|---|---:|---:|---:|
| 1 | 208 | 131 | 62.98% |
| 2 | 164 | 83 | 50.61% |
| 3 | 401 | 104 | 25.94% |

The observed survival rate decreases from first class to third class.

Artifact:

```text
analytics/plots/survival_rate_by_pclass.png
```

## 4.3 Survival by sex and passenger class

| Sex | Class | Passengers | Survivors | Survival Rate |
|---|---:|---:|---:|---:|
| Female | 1 | 91 | 88 | 96.70% |
| Female | 2 | 72 | 66 | 91.67% |
| Female | 3 | 127 | 60 | 47.24% |
| Male | 1 | 117 | 43 | 36.75% |
| Male | 2 | 92 | 17 | 18.48% |
| Male | 3 | 274 | 44 | 16.06% |

The combined analysis shows that survival rates vary substantially across both sex and passenger class.

Artifact:

```text
analytics/plots/survival_rate_by_sex_and_pclass.png
```

---

# 5. Correlation Analysis

The required six-column correlation matrix uses exactly:

```text
survived
pclass
age
sibsp
parch
fare
```

| | survived | pclass | age | sibsp | parch | fare |
|---|---:|---:|---:|---:|---:|---:|
| survived | 1.0000 | -0.3284 | -0.0832 | -0.0363 | 0.0716 | 0.2452 |
| pclass | -0.3284 | 1.0000 | -0.3394 | 0.0853 | 0.0372 | -0.5535 |
| age | -0.0832 | -0.3394 | 1.0000 | -0.2784 | -0.1815 | 0.0896 |
| sibsp | -0.0363 | 0.0853 | -0.2784 | 1.0000 | 0.3791 | 0.1352 |
| parch | 0.0716 | 0.0372 | -0.1815 | 0.3791 | 1.0000 | 0.1922 |
| fare | 0.2452 | -0.5535 | 0.0896 | 0.1352 | 0.1922 | 1.0000 |

The two strongest absolute off-diagonal correlations are:

1. `pclass` and `fare`: **-0.5535**
2. `sibsp` and `parch`: **0.3791**

The negative pclass/fare relationship reflects the numerical encoding of passenger class, where lower class numbers correspond to higher passenger class. The positive sibsp/parch relationship indicates an association between the number of siblings/spouses and parents/children aboard.

Artifacts:

```text
analytics/correlation_matrix.csv
analytics/top_correlations.csv
analytics/plots/correlation_heatmap.png
```

---

# 6. Classification

## 6.1 Features and target

Target:

```text
survived
```

Initial classification features:

```text
pclass
sex
age
sibsp
parch
fare
embarked
```

The `alive` column was excluded because it directly represents the target and would introduce target leakage.

Redundant derived variables such as `class`, `who`, `adult_male`, and `alone` were also excluded from the initial classification feature set.

---

# 7. Train/Test Split

The dataset was divided using an 80/20 stratified split.

| Class | Count | Percentage |
|---|---:|---:|
| 0 — Not Survived | 455 | 58.86% |
| 1 — Survived | 318 | 41.14% |

Training observations: **618**

Testing observations: **155**

The class proportions were preserved using stratification.

---

# 8. Preprocessing

Numerical features:

```text
pclass
age
sibsp
parch
fare
```

Categorical features:

```text
sex
embarked
```

Numerical preprocessing:

- Median imputation
- StandardScaler

Categorical preprocessing:

- Most-frequent imputation
- OneHotEncoder(handle_unknown="ignore")

All preprocessing used in model pipelines is fitted on training data and then applied to test data.

---

# 9. Baseline Classification Models

Three baseline classifiers were trained:

- Logistic Regression
- Decision Tree
- Random Forest

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7290 | 0.6897 | 0.6250 | 0.6557 | 0.7869 |
| Decision Tree | 0.7419 | 0.8750 | 0.4375 | 0.5833 | 0.7690 |
| Random Forest | 0.7548 | 0.7241 | 0.6562 | 0.6885 | 0.7837 |

Artifacts:

```text
analytics/plots/logistic_regression_confusion_matrix.png
analytics/plots/decision_tree_confusion_matrix.png
analytics/plots/random_forest_confusion_matrix.png
analytics/plots/classification_roc_curves.png
analytics/plots/decision_tree_visualization.png
```

---

# 10. Class-Imbalance Experiment

Three Logistic Regression approaches were compared:

1. Baseline Logistic Regression
2. Logistic Regression with `class_weight="balanced"`
3. Logistic Regression with SMOTE

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Baseline Logistic Regression | 0.7290 | 0.6897 | 0.6250 | 0.6557 | 0.7869 |
| Balanced Logistic Regression | 0.7290 | 0.6719 | 0.6719 | 0.6719 | 0.7869 |
| SMOTE Logistic Regression | 0.7355 | 0.6825 | 0.6719 | 0.6772 | 0.7842 |

SMOTE was applied only to the training data.

Before SMOTE:

```text
0 = 364
1 = 254
```

After SMOTE:

```text
0 = 364
1 = 364
```

Both class weighting and SMOTE increased positive-class recall from 0.6250 to 0.6719. SMOTE produced an F1 score of 0.6772.

Artifact:

```text
analytics/imbalance_comparison.csv
```

---

# 11. Random Forest Hyperparameter Tuning

Random Forest hyperparameters were tuned using 5-fold GridSearchCV.

Search space:

```text
n_estimators: [100, 200]
max_depth: [None, 5, 10]
max_features: ['sqrt', 'log2']
```

Scoring: **F1**

Best parameters:

```text
n_estimators = 100
max_depth = 10
max_features = sqrt
```

Best cross-validation F1: **0.7345**

OOB score: **0.7848**

Tuned test-set performance:

| Metric | Result |
|---|---:|
| Accuracy | 0.8000 |
| Precision | 0.8235 |
| Recall | 0.6562 |
| F1 | 0.7304 |
| AUC | 0.7890 |

The tuned model improved test accuracy and F1 compared with the original Random Forest.

Artifacts:

```text
analytics/random_forest_gridsearch_results.csv
analytics/tuned_random_forest.joblib
```

---

# 12. Regression

A separate regression task predicts `fare` from the other available Titanic features.

The regression pipeline uses:

- Median imputation for numerical variables
- Standardization of numerical variables
- Most-frequent imputation for categorical variables
- One-hot encoding
- Random Forest Regression

| Metric | Result |
|---|---:|
| MAE | 16.8167 |
| RMSE | 44.2771 |
| R² | 0.3683 |
| Adjusted R² | 0.2574 |

The model explains approximately 36.83% of the variance in the test-set fare values according to R².

Artifact:

```text
analytics/regression_results.csv
```

---

# 13. Residual Analysis

The residual plot showed a much larger residual spread at higher predicted fare values.

| Prediction Range | Residual Std |
|---|---:|
| 7.09–12.762 | 2.5919 |
| 12.762–29.318 | 8.2125 |
| 29.318–366.998 | 74.9213 |

The calculated residual-spread ratio was **28.9063**.

This substantial difference in residual spread suggests possible heteroscedasticity.

Artifacts:

```text
analytics/plots/fare_regression_residuals.png
analytics/regression_predictions.csv
```

---


# 13A. Graph Interpretations

The following interpretations summarize the main visual patterns shown in the
Module 2 charts. The interpretations describe observed patterns in the
cleaned Titanic dataset and do not imply causal relationships.

## 13A.1 Age Distribution

The age histogram shows that passenger ages are concentrated in the younger
and middle-age ranges, with fewer observations at older ages. The distribution
is not perfectly symmetric, so the median and spread are useful alongside the
mean when describing passenger age.

**Chart:** `analytics/plots/age_distribution.png`

## 13A.2 Age Boxplot

The age boxplot shows the central distribution of passenger ages together
with observations identified as IQR-based outliers. These outliers were
retained because an unusual age value is not automatically a data-entry error.

**Chart:** `analytics/plots/age_boxplot.png`

## 13A.3 Fare Distribution

The fare histogram is strongly right-skewed, with many passengers paying
relatively lower fares and a smaller number paying substantially higher fares.
This pattern is consistent with the large difference between the mean fare
(34.7617) and median fare (15.9000).

**Chart:** `analytics/plots/fare_distribution.png`

## 13A.4 Fare Boxplot

The fare boxplot highlights the large upper-tail spread in passenger fares
and the IQR-based outliers. The 100 observations outside the IQR bounds were
retained because high fares can represent genuine passenger observations
rather than data errors.

**Chart:** `analytics/plots/fare_boxplot.png`

## 13A.5 Survival Rate by Sex

The survival-rate chart shows a substantially higher observed survival rate
among female passengers (73.79%) than male passengers (21.53%) in the cleaned
dataset. This is an observed association in this dataset and should not be
interpreted as a causal effect of sex.

**Chart:** `analytics/plots/survival_rate_by_sex.png`

## 13A.6 Survival Rate by Passenger Class

The chart shows that the observed survival rate decreases from first class
to third class. First-class passengers had a survival rate of 62.98%, while
third-class passengers had a survival rate of 25.94%.

**Chart:** `analytics/plots/survival_rate_by_pclass.png`

## 13A.7 Survival Rate by Sex and Passenger Class

The combined chart shows substantial differences across both sex and
passenger class. The highest observed survival rate was for female
first-class passengers (96.70%), while male third-class passengers had the
lowest observed survival rate among the displayed groups (16.06%).

**Chart:** `analytics/plots/survival_rate_by_sex_and_pclass.png`

## 13A.8 Correlation Heatmap

The correlation heatmap shows that `pclass` and `fare` have the strongest
absolute correlation among the required six variables, with a correlation
of -0.5535. `pclass` also has a negative correlation with `survived`
(-0.3470), while `fare` has a positive correlation with `survived` (0.2641);
these are associations rather than evidence of causation.

**Chart:** `analytics/plots/correlation_heatmap.png`

## 13A.9 Classification ROC Curves

The ROC curves compare the ability of the classification models to separate
survived and non-survived passengers across classification thresholds.
Curves that remain closer to the upper-left region indicate stronger
discrimination on the evaluation data, while the diagonal represents
approximately random discrimination.

**Chart:** `analytics/plots/classification_roc_curves.png`

## 13A.10 Regression Residual Plot

The residual plot shows a substantially larger spread of residuals at higher
predicted fare values. The calculated residual-spread ratio was 28.9063,
which suggests possible heteroscedasticity in the multivariate linear
regression residuals.

**Chart:** `analytics/plots/fare_regression_residuals.png`


# 14. Final Model Comparison

The complete classification comparison is stored in:

```text
analytics/classification_model_comparison.csv
```

The tuned Random Forest achieved:

- Accuracy = 0.8000
- Precision = 0.8235
- Recall = 0.6562
- F1 = 0.7304
- AUC = 0.7890

The tuned Random Forest is used as the final classification pipeline candidate based on its observed combination of test accuracy and F1.

The Balanced Logistic Regression produced the highest recall at 0.6719. This difference is retained because model selection can depend on the evaluation metric emphasized.

---

# 15. Saved Production-Style Pipeline

The complete classification pipeline was saved using Joblib:

```text
analytics/titanic_survival_pipeline.joblib
```

The saved pipeline contains:

```text
Raw input
    ↓
ColumnTransformer
    ↓
Numerical imputation + scaling
    ↓
Categorical imputation + one-hot encoding
    ↓
Tuned Random Forest
    ↓
Prediction
```

The saved pipeline was reloaded successfully.

Reloaded test accuracy: **0.8000**

Original and reloaded predictions were identical.

A raw passenger record was also passed directly into the reloaded pipeline.

Example:

```text
pclass   = 1
sex      = female
age      = 30
sibsp    = 0
parch    = 0
fare     = 80.0
embarked = S
```

Example model output:

```text
Predicted class: 1
Predicted label: Survived
Survival probability: 1.0000
```

This is a model prediction for the supplied feature values and should not be interpreted as a general real-world probability estimate.

---

# 16. Analytics Artifacts

## Data

```text
titanic.csv
titanic_clean.csv
correlation_matrix.csv
top_correlations.csv
```

## Classification

```text
classification_model_comparison.csv
imbalance_comparison.csv
random_forest_gridsearch_results.csv
final_model_summary.csv
titanic_survival_pipeline.joblib
tuned_random_forest.joblib
sample_prediction.csv
```

## Regression

```text
regression_results.csv
regression_predictions.csv
```

## Plots

```text
age_boxplot.png
age_distribution.png
classification_roc_curves.png
correlation_heatmap.png
decision_tree_confusion_matrix.png
decision_tree_visualization.png
fare_boxplot.png
fare_distribution.png
fare_regression_residuals.png
logistic_regression_confusion_matrix.png
random_forest_confusion_matrix.png
survival_rate_by_pclass.png
survival_rate_by_sex.png
survival_rate_by_sex_and_pclass.png
```

---

# 17. Reproducibility

From the repository root:

```bash
pip install -r requirements.txt
```

The Titanic dataset can be loaded using:

```python
import seaborn as sns
titanic = sns.load_dataset("titanic")
```

The committed fallback is available at:

```text
analytics/titanic.csv
```

The saved classification pipeline can be loaded with:

```python
import joblib
pipeline = joblib.load("analytics/titanic_survival_pipeline.joblib")
```

A raw passenger DataFrame containing the expected feature columns can then be supplied directly to:

```python
pipeline.predict(raw_data)
pipeline.predict_proba(raw_data)
```

---

# 18. Key Conclusions

The cleaned Titanic dataset contains 773 observations after the documented cleaning process.

Observed survival rates differed substantially by sex and passenger class.

The required six-variable correlation analysis identified:

- `pclass` and `fare` = **-0.5535**
- `sibsp` and `parch` = **0.3791**

For classification, the tuned Random Forest achieved:

- Accuracy = **0.8000**
- F1 = **0.7304**
- AUC = **0.7890**

The imbalance experiment demonstrated that class weighting and SMOTE increased positive-class recall.

For regression, the Random Forest achieved:

- MAE = **16.8167**
- RMSE = **44.2771**
- R² = **0.3683**
- Adjusted R² = **0.2574**

Residual diagnostics indicated substantial variation in residual spread across predicted-fare ranges, suggesting possible heteroscedasticity.

The complete tuned classification pipeline was persisted with Joblib, successfully reloaded, and verified to produce identical predictions.
