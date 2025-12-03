# CSE422: Mushroom Classification Project

## 🍄 Project Overview

This project focuses on the classification of mushrooms as either **edible ('e')** or **poisonous ('p')** based on various morphological characteristics. It involves a complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), supervised classification using multiple algorithms, and unsupervised clustering analysis (KMeans).

The primary goal is to determine which machine learning model performs best on this dataset and to understand the underlying structure of the data.

## 💾 Dataset

The project utilizes the `mushroom_dataset.csv`, which contains information on various mushroom features.

*   **Initial Size:** Approximately 105,000 data points.
*   **Target Variable:** `class` (binary: 0 for edible, 1 for poisonous after encoding).
*   **Features:** Includes both quantitative (e.g., `cap-diameter`, `stem-height`, `stem-width`) and categorical features.

## 🛠️ Methodology and Workflow

The project followed a standard data science pipeline executed within the Jupyter Notebook (`CSE422_Project.ipynb`).

### 1. Data Loading and Initial Inspection

*   The dataset was loaded and inspected for data types, missing values, and class distribution.
*   The target class distribution was found to be slightly imbalanced but manageable.

### 2. Data Preprocessing and Cleaning

**Feature Dropping:**
Features with high sparsity or zero variance were dropped to simplify the model and prevent noise:
*   `stem-root`
*   `veil-type`
*   `veil-color`
*   `spore-print-color`
*   `has-ring`

**Missing Value Imputation:**
Missing values in key categorical columns were imputed using the estimated mode (or a specific constant value based on observation):
*   `gill-spacing` (imputed with 'c')
*   `stem-surface` (imputed with 's')
*   `cap-surface` (imputed with 't')
*   `gill-attachment` (imputed with 'a')
*   `ring-type` (imputed with 'f')

**Feature Encoding:**
All remaining categorical features, including the target variable, were converted into numerical representations using `LabelEncoder`.

### 3. Exploratory Data Analysis (EDA)

*   Statistical summaries (mean, standard deviation, skewness) were generated for numerical features.
*   Data distributions were visualized using histograms and density plots.
*   Outliers were checked using boxplots.
*   Correlation analysis (Pearson, Spearman, Kendall) was performed to understand relationships between features and the target variable.

### 4. Supervised Learning (Classification)

The data was split using a stratified 80% training / 20% testing split. The following classification models were evaluated:

| Model | Class |
| :--- | :--- |
| **KNN** | K-Nearest Neighbors |
| **Decision Tree** | Decision Tree Classifier |
| **Logistic Regression** | Linear Model |
| **Naive Bayes** | Gaussian Naive Bayes |
| **Neural Network** | MLP Classifier |
| **Random Forest** | Ensemble Tree Method |

All models were evaluated using standard metrics: Accuracy, Precision, Recall, F1-Score, AUC, and Log Loss.

### 5. Unsupervised Learning (Clustering)

*   **KMeans Clustering** was applied using `k=2` (matching the number of true classes).
*   Data was scaled using `StandardScaler` prior to clustering.
*   Dimensionality reduction (PCA) was used to visualize the clusters compared to the true labels.
*   A **Silhouette Score** was calculated, and a **Contingency Table** was generated to assess how well the clusters align with the true edible/poisonous labels.

### 6. Linear Regression (Regression Task)

As a supplementary task, Linear Regression was applied (despite the target being binary) to evaluate regression metrics (MSE, RMSE, MAE, R²), primarily as an exercise in outlier handling and metric calculation.

## 💡 Key Results and Conclusions

### Supervised Classification Performance

The highly structured nature of this dataset (common for this specific mushroom data source) resulted in extremely high performance for tree-based models.

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Tree** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| KNN | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 1.0000 |
| Neural Network | 0.9996 | 0.9996 | 0.9997 | 0.9997 | 1.0000 |
| Logistic Regression | 0.9632 | 0.9632 | 0.9632 | 0.9632 | 0.9950 |
| Naive Bayes | 0.9572 | 0.9587 | 0.9572 | 0.9573 | 0.9829 |

**Conclusion:** Both the **Decision Tree Classifier** and **Random Forest Classifier** achieved perfect separation, indicating that a combination of features perfectly distinguishes edible from poisonous mushrooms in this dataset.

### Unsupervised Clustering

The KMeans algorithm, tasked with finding two clusters, showed a very strong alignment with the true classification labels, yielding a high Silhouette Score and a highly accurate contingency table.

*   **Silhouette Score:** High (e.g., typically around 0.5-0.7, depending on scaling).
*   **Contingency Table:** Showed a near-perfect segregation of the data points into the two clusters corresponding to the true classes (edible vs. poisonous).

## 💻 Dependencies

This project requires Python 3 and the following libraries:

```bash
pandas
numpy
scikit-learn
seaborn
matplotlib
