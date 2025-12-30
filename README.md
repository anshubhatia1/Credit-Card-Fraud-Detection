# Credit Card Fraud Detection - Complete Analysis Workflow

## Overview
This notebook implements a comprehensive fraud detection pipeline on the Credit Card Dataset. The analysis addresses the critical challenge of **highly imbalanced datasets** (0.17% fraud vs 99.83% non-fraud) using various techniques to build an effective fraud detection model.

---

## Workflow Summary

### **PHASE 1: Data Loading & Exploration**

#### Step 1: Load and Inspect Data
- Load creditcard.csv dataset
- Display basic information: shape (284,807 rows × 31 columns)
- Examine target variable distribution (Class): severely imbalanced

#### Step 2: Understand Data Characteristics
- **Descriptive Statistics**: Compute mean, median, quartiles for all features
- **Skewness Detection**: Identify right-skewed, left-skewed, or symmetric distributions
- **Outlier Detection**: Use IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR) to detect outliers
- **Heavy-Tail Analysis**: 
  - Calculate kurtosis (measure of tail heaviness)
  - Compute tail ratio: (99th percentile - 1st percentile) / IQR
  - Flag columns with kurtosis > 3 or tail_ratio > 10

**Finding**: Most features (27/30) exhibit heavy-tail distributions with extreme values

---

### **PHASE 2: Data Preprocessing**

#### Step 1: Handle Outliers & Skewness (3-Step Approach)

**1. Winsorization**
```
Cap extreme values at 1st and 99th percentiles
df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
```
- Reduces impact of extreme outliers
- Preserves data distribution shape

**2. Power Transformation (Yeo-Johnson)**
```
PowerTransformer(method='yeo-johnson')
```
- Reduces skewness in the data
- Makes distributions more symmetric
- Improves model performance

**3. Robust Scaling**
```
RobustScaler()
```
- Scales features using median and IQR (resistant to outliers)
- Formula: (X - median) / IQR
- Better than StandardScaler for data with outliers

**Note**: Time and Class columns excluded from transformation

#### Step 2: Verify Preprocessing
- Visualize distributions before/after transformation
- Confirm data quality improvements

---

### **PHASE 3: Feature Analysis**

#### Step 1: Correlation Analysis
- Compute **Spearman correlation matrix** (more robust for non-linear relationships)
- **Finding**: All correlation values < 0.50 → low multicollinearity ✅

#### Step 2: Variance Inflation Factor (VIF)
- Measure multicollinearity among features
- **Finding**: All VIF values in range 1-2 → excellent, no multicollinearity ✅

#### Step 3: Feature-Target Relationship
- Calculate **Mutual Information (MI)** for each feature vs target
- Rank features by their predictive power
- Visualize top features using violin plots
- **Finding**: V10, V17, V4, V12, V14 are most informative

---

### **PHASE 4: Data Splitting**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    stratify=y,  # Maintains class distribution
    random_state=42
)
```

- **80% Training**: Used for model fitting
- **20% Testing**: Used for final evaluation
- **Stratified Split**: Maintains fraud/non-fraud ratio in both sets

---

### **PHASE 5: Handling Class Imbalance - Multiple Approaches**

#### **Approach A: Random Oversampling (ROS)**

**Concept**: Duplicate minority class samples randomly

```python
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
```

**Results**:
- ✅ Logistic Regression: No overfitting, ROC-AUC gap = 0.014 (healthy)
- ❌ Random Forest: Severe overfitting, train AUC = 1.00 → test AUC = 0.96 (memorization)

**Why Overfitting?**: Trees split until leaves are pure; duplicated samples enable memorization

---

#### **Approach B: SMOTE (Synthetic Minority Over-sampling Technique)**

**Concept**: Generate synthetic minority samples by interpolating between neighbors

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy=0.25, k_neighbors=5, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
```

**Results**: Similar to ROS, both models show improvements but still concern over Random Forest generalization

---

#### **Approach C: SMOTETomek (Hybrid Method)**

**Concept**: Combines SMOTE (oversampling) + Tomek links (undersampling boundary samples)

```python
from imblearn.combine import SMOTETomek
smote_tomek = SMOTETomek(sampling_strategy=0.25, random_state=42)
X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)
```

---

#### **Approach D: No Oversampling (Baseline)**

**Concept**: Train on original imbalanced data with proper regularization

```python
log_reg_base = LogisticRegression(
    C=0.3,                          # Stronger regularization
    class_weight="balanced",        # Higher penalty for fraud misclassification
    max_iter=2000,
    penalty="l2",
    random_state=42
)
```

**Finding**: **Best approach!** Avoids overfitting while maintaining good performance

---

### **PHASE 6: Model Training & Evaluation**

#### **Models Evaluated**:

1. **Logistic Regression**
   - Linear decision boundary
   - Naturally regularized
   - Good generalization
   
2. **Random Forest Classifier**
   - Non-linear, captures complex patterns
   - Prone to overfitting with imbalanced data

#### **Evaluation Metrics**:

```
1. Confusion Matrix:
   - True Positives (TP): Correctly predicted fraud
   - False Positives (FP): Incorrectly flagged as fraud
   - True Negatives (TN): Correctly predicted non-fraud
   - False Negatives (FN): Missed fraud cases

2. ROC-AUC Score:
   - Measures ranking ability
   - Target: > 0.95 for fraud detection

3. PR-AUC (Precision-Recall AUC):
   - Better for imbalanced datasets
   - Focuses on minority class

4. Classification Report:
   - Precision: Of predicted frauds, how many are actually frauds?
   - Recall: Of actual frauds, how many did we catch?
   - F1-Score: Harmonic mean (balance precision & recall)
```

---

### **PHASE 7: Overfitting Analysis**

#### **Step 1: Train vs Test Comparison**
- Train models on original data
- Compare train vs test metrics
- Random Forest shows 0.23 gap in ROC-AUC → **overfitting detected**

#### **Step 2: Cross-Validation**
```python
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv=cv)
```

- Logistic Regression: Mean CV AUC ≈ 0.97 (stable)
- Random Forest: Shows high variance across folds

#### **Step 3: Learning Curves**
- Plot training vs validation AUC vs training set size
- Diagnoses bias (underfitting) vs variance (overfitting) issues

**Conclusion**: Logistic Regression with balanced class weights is the winner ✅

---

### **PHASE 8: Cost-Sensitive Learning**

#### **Motivation**
- Missing a fraud (False Negative) costs more than a false alarm (False Positive)
- Traditional models treat both errors equally

#### **Implementation**
```python
cost_ratio = 10  # Fraud misclassification penalty

log_reg_cost = LogisticRegression(
    class_weight={0: 1, 1: cost_ratio},  # 10x penalty for fraud errors
    C=0.3,
    random_state=42
)
```

**Effect**:
- ⬆️ Precision improves (fewer false alarms)
- ⬇️ Recall may decrease (but catches more actual fraud)
- ✅ Better F1-Score for fraud class

---

### **PHASE 9: Threshold Tuning**

#### **Problem with Default Threshold (0.5)**
- Logistic Regression outputs probabilities [0, 1]
- Default: Predict 1 if probability > 0.5
- For imbalanced data, this is suboptimal (too many false positives)

#### **Solution: Grid Search for Optimal Threshold**

```python
thresholds = np.arange(0.0, 1.01, 0.05)
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    Calculate: precision, recall, F1-score
```

**Results on Validation Set**:
- Test different thresholds (0.0, 0.05, 0.10, ..., 1.0)
- Find threshold that maximizes F1-score
- **Optimal Threshold ≈ 0.85**

#### **What Does Threshold = 0.85 Mean?**
- "Predict fraud only if model is ≥85% confident"
- Reduces false positives
- Improves precision while maintaining reasonable recall

---

### **PHASE 10: Model Explainability with SHAP**

#### **Why Explainability Matters?**
- Understand which features drive fraud predictions
- Build trust in the model
- Identify potential biases

#### **SHAP Implementation**
```python
import shap

explainer = shap.LinearExplainer(
    log_reg_cost,
    X_train,
    feature_perturbation="interventional"
)

shap_values = explainer.shap_values(X_test)
```

#### **Visualizations**:

1. **Summary Plot (Bar)**
   - Shows feature importance
   - Ranked by average |SHAP value|

2. **Summary Plot (Beeswarm)**
   - Shows relationship between feature values and SHAP values
   - Color: Red = high feature value, Blue = low feature value
   - Position: How much it pushes prediction left/right

3. **Waterfall Plot**
   - Explains individual prediction
   - Shows contribution of each feature
   - Base value → Prediction step-by-step

---

## Key Findings & Recommendations

### ✅ What Worked
1. **Logistic Regression** outperformed Random Forest due to generalization
2. **Balanced class weights** effectively handled imbalance without data manipulation
3. **Threshold tuning to 0.85** optimized precision-recall tradeoff
4. **Cost-sensitive learning** aligned model with business objectives
5. **SHAP explainability** made predictions interpretable

### ❌ What Didn't Work
1. **Random Oversampling** → Random Forest memorized duplicates
2. **SMOTE/SMOTETomek** → Similar issues with tree-based models
3. **Default 0.5 threshold** → Too many false positives in imbalanced setting

### 📊 Final Model Performance
- **Model**: Logistic Regression with cost-sensitive weights
- **Threshold**: 0.85
- **Test ROC-AUC**: ~0.97
- **Test Recall (Fraud)**: ~84%
- **Test Precision (Fraud)**: ~92%
- **F1-Score**: ~0.88

---

## Technical Stack

| Component | Library | Version |
|-----------|---------|---------|
| Data Processing | Pandas, NumPy | Latest |
| Visualization | Matplotlib, Seaborn | Latest |
| Preprocessing | Scikit-learn | Latest |
| Class Imbalance | Imbalanced-learn (imblearn) | Latest |
| Model Explainability | SHAP | Latest |

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone or download the project**
   ```bash
   cd Credit-Card-Fraud-Detection
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This installs all required packages:
   - Data processing: pandas, numpy
   - Visualization: matplotlib, seaborn
   - Statistics: scipy, statsmodels
   - Machine Learning: scikit-learn
   - Class imbalance handling: imbalanced-learn
   - Model explainability: shap
   - Jupyter: jupyter, ipython (for notebook execution)

---

## How to Run

1. Place `creditcard.csv` in the `data/` folder
2. Open `test.ipynb` in Jupyter Notebook
3. Execute cells sequentially from top to bottom
4. Observe outputs at each phase

---

## Dataset Information

**Source**: Kaggle Credit Card Fraud Detection Dataset

**Statistics**:
- **Total Records**: 284,807
- **Features**: 28 PCA-transformed + Time + Amount
- **Target**: Class (0=Non-fraud, 1=Fraud)
- **Fraud Rate**: 0.17% (492 fraud cases)
- **Class Distribution**: Highly imbalanced

---

## Conclusion

This analysis demonstrates a **production-ready approach** to fraud detection addressing:
- ✅ Extreme class imbalance
- ✅ Model overfitting risks
- ✅ Business cost considerations (cost-sensitive learning)
- ✅ Decision threshold optimization
- ✅ Model interpretability

The final model balances **high recall** (catching fraud) with **low false positive rate** (minimizing customer friction), making it suitable for real-world deployment.
