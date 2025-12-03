# CSE422: Mushroom Classification Project

## 🍄 Project Overview

This project focuses on the binary classification of mushrooms as either **edible ('e')** or **poisonous ('p')** based on various morphological characteristics. It implements a complete machine learning workflow, encompassing data preprocessing, exploratory data analysis (EDA), supervised classification using multiple algorithms, and unsupervised clustering analysis (KMeans).

The primary goal is to systematically compare the performance of various machine learning models on this structured dataset.

## 💾 Dataset

The project utilizes the `mushroom_dataset.csv`.

*   **Size:** Approximately 105,000 data points.
*   **Target Variable:** `class` (binary classification).
*   **Features:** A mix of quantitative (`cap-diameter`, `stem-height`, `stem-width`) and categorical characteristics.

## 🛠️ Methodology and Workflow

The project followed a standard data science pipeline executed within the Jupyter Notebook (`CSE422_Project.ipynb`).

### 1. Data Loading and Initial Inspection

*   Initial data characteristics, including shape, data types, and class distribution, were verified. The problem was confirmed as a binary classification task.

### 2. Data Preprocessing and Cleaning

**Feature Dropping:**
Five low-variance or highly sparse columns were dropped to reduce dimensionality and noise:
*   `stem-root`
*   `veil-type`
*   `veil-color`
*   `spore-print-color`
*   `has-ring`

**Missing Value Imputation:**
Missing categorical values in the remaining features (`gill-spacing`, `stem-surface`, `cap-surface`, `gill-attachment`, `ring-type`) were imputed using specific constant values (e.g., 'c', 's', 't', 'a', 'f').

**Feature Encoding:**
The `LabelEncoder` was applied to all remaining categorical features, transforming them into numerical data suitable for modeling.

### 3. Exploratory Data Analysis (EDA)

*   Statistical summaries (`describe`, variance, skewness) were generated for numerical features.
*   Distributions were visualized using histograms, density plots, and boxplots to inspect skewness and outliers.
*   Correlation matrices (Pearson, Spearman, Kendall) were calculated and visualized to understand feature-to-feature and feature-to-target relationships.

### 4. Supervised Learning (Classification)

The dataset was split using a stratified 70%/30% split initially, and later refined to an 80%/20% stratified split for the final comprehensive comparison.

The following classifiers were implemented and evaluated:

| Model Category | Specific Model Implemented |
| :--- | :--- |
| **Neighbors** | K-Nearest Neighbors (`n_neighbors=6`) |
| **Decision Tree** | Decision Tree Classifier |
| **Linear Model** | Logistic Regression |
| **Probabilistic** | Gaussian Naive Bayes |
| **Neural Network** | MLP Classifier (Multi-layer Perceptron) |
| **Ensemble** | Random Forest Classifier |

**Evaluation:**
Model performance was rigorously assessed using a standard set of metrics derived from `sklearn.metrics`:
*   Accuracy
*   Precision
*   Recall
*   F1-Score
*   ROC AUC
*   Log Loss

The notebook includes visual comparisons of these metrics (bar charts and ROC curves) and detailed Confusion Matrices for each model.

### 5. Unsupervised Learning (Clustering)

K-Means clustering was performed on the scaled feature data, assuming $k=2$ clusters (to match the true classes).

*   Data was scaled using `StandardScaler`.
*   Cluster quality was measured using the **Silhouette Score**.
*   **Principal Component Analysis (PCA)** was used to project the data into 2 dimensions for visualization, comparing the KMeans-derived clusters against the known true labels.
*   A **Contingency Table** was generated to quantify the mapping between the derived cluster IDs and the true class labels.

### 6. Supplementary Regression Analysis

As an additional exercise, Linear Regression was applied to the data. This analysis included basic outlier detection on training residuals (using $3\sigma$ threshold) and reported standard regression metrics (MSE, RMSE, MAE, and R²).

## 💻 Dependencies

This project requires Python 3 and the following libraries:

```bash
pandas
numpy
scikit-learn
seaborn
matplotlib
