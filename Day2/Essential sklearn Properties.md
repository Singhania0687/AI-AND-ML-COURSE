# Essential `sklearn` Properties & Modules for Developers

---

## 1. Core Modules of `sklearn`

| Module | Purpose |
|--------|---------|
| `sklearn.datasets` | Load example datasets (e.g., Iris, Boston) for testing and practice |
| `sklearn.model_selection` | Train-test splitting, cross-validation, hyperparameter tuning (`GridSearchCV`) |
| `sklearn.preprocessing` | Data preprocessing: scaling, normalization, encoding categorical data |
| `sklearn.linear_model` | Linear models: Linear Regression, Logistic Regression, Ridge, Lasso |
| `sklearn.tree` | Decision Trees and tree-based models |
| `sklearn.ensemble` | Ensemble methods: Random Forest, Gradient Boosting |
| `sklearn.svm` | Support Vector Machines |
| `sklearn.neighbors` | K-Nearest Neighbors algorithm |
| `sklearn.cluster` | Clustering algorithms (K-Means, DBSCAN, etc.) |
| `sklearn.metrics` | Model evaluation metrics (accuracy, precision, recall, RMSE) |
| `sklearn.pipeline` | Pipeline to chain preprocessing + model steps |
| `sklearn.decomposition` | Dimensionality reduction (PCA, etc.) |

---

## 2. Important Properties / Classes

### a. Estimators
All algorithms in `sklearn` are implemented as **estimators** — classes that implement `.fit()` and `.predict()` methods.  
**Examples:** `LogisticRegression()`, `RandomForestClassifier()`, `KMeans()`

### b. `fit(X, y)`
Trains the model on data `X` (features) and labels `y` (for supervised learning).

### c. `predict(X)`
Predicts the output for new data `X`.

### d. `transform(X)`
Used for preprocessing or dimensionality reduction. Transforms data `X` (e.g., scaling or PCA).

### e. `fit_transform(X[, y])`
Combines `fit` and `transform` in one step (commonly used in preprocessing).

### f. `score(X, y)`
Returns the model’s performance metric.  
By default: accuracy for classifiers.

---

## 3. Key Features You Should Know

| Feature | Explanation & Use Case |
|--------|------------------------|
| **Train-Test Split** (`train_test_split`) | Splits dataset into train and test parts for unbiased evaluation |
| **Cross-Validation** (`cross_val_score`) | Evaluates model performance more reliably by training/testing on multiple splits |
| **Grid Search** (`GridSearchCV`) | Automates hyperparameter tuning by exhaustively searching parameter combinations |
| **Pipelines** (`Pipeline`) | Chains steps (e.g., scaling + classification) into one estimator to simplify code and avoid data leakage |
| **Preprocessing Tools** | Scaling (`StandardScaler`, `MinMaxScaler`), Encoding (`OneHotEncoder`), Imputation (`SimpleImputer`) to prepare raw data |
| **Metrics** | For classification: `accuracy_score`, `precision_score`, `recall_score`, `f1_score`<br>For regression: `mean_squared_error`, `r2_score` |
| **Feature Selection** (`SelectKBest`, `RFE`) | Selects important features to reduce overfitting and improve model efficiency |
| **Model Persistence** (`joblib`, `pickle`) | Save and load trained models to reuse without retraining |

---

## 4. Typical Workflow in `sklearn`

1. **Load data** (from `datasets` or your own CSV)
2. **Preprocess data** (using `preprocessing` tools)
3. **Split data** (`train_test_split`)
4. **Choose model** (e.g., `RandomForestClassifier()`)
5. **Train model** (`fit`)
6. **Predict results** (`predict`)
7. **Evaluate model** (using `metrics`)
8. **Tune hyperparameters** (`GridSearchCV`) if needed
9. **Save the trained model** for production use

---
