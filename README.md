# Sampling Techniques and Model Performance on an Imbalanced Credit Card Dataset

## Overview

This repository contains an academic implementation that evaluates how different sampling techniques affect the predictive accuracy of multiple machine learning models on a highly imbalanced credit card dataset.

The workflow includes:

* Balancing the dataset
* Computing a statistically motivated sample size
* Generating multiple samples using distinct sampling strategies
* Benchmarking model accuracy on a held-out test set

---

## Dataset

**Source (GitHub):**
https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

**Local file used in this repository:**
`Creditcard_data.csv`

**Target column:** `Class` (binary classification)

---

## Methodology

### 1. Loading and Balancing the Dataset

The dataset is initially highly imbalanced. To reduce class bias during training, the notebook balances the dataset using **random oversampling of the minority class**.

* Majority class: `Class = 0`
* Minority class: `Class = 1`
* Technique: `sklearn.utils.resample` with replacement
* Final step: dataset shuffling

---

### 2. Train-Test Split

After balancing, the dataset is split into:

* **Test set:** 20% (stratified)
* **Pool set:** 80% (used as the sampling pool)

This ensures:

* Evaluation is performed on a **fixed hold-out test set**
* Sampling strategies operate only on the **pool dataset**

---

### 3. Sample Size Calculation

Sample size is computed using **Cochran’s Formula**:

* Confidence Level: 95%
* Margin of Error: 5%
* p = 0.5

**Calculated Sample Size: 385**

---

### 4. Sampling Techniques Implemented

Five sampling strategies are applied to the pool dataset:

1. Simple Random Sampling
2. Stratified Sampling
3. Systematic Sampling
4. Cluster Sampling (KMeans with 10 clusters)
5. Bootstrap Sampling

Each technique generates a sample of **size 385**.

---

### 5. Machine Learning Models Evaluated

The following models are trained on each sampled dataset:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* Gradient Boosting Classifier

---

### 6. Evaluation Metric

* **Metric:** Accuracy (%)
* **Evaluation:** Fixed hold-out test set

---

## Results (Accuracy Percentage)

| Model               | Simple Random | Stratified | Systematic | Cluster   | Bootstrap |
| ------------------- | ------------- | ---------- | ---------- | --------- | --------- |
| Logistic Regression | 93.137255     | 93.790850  | 92.483660  | 78.758170 | 93.464052 |
| Decision Tree       | 98.039216     | 99.019608  | 97.385621  | 80.392157 | 97.712418 |
| Random Forest       | 99.346405     | 99.673203  | 99.673203  | 81.699346 | 99.673203 |
| SVM                 | 68.627451     | 67.973856  | 69.607843  | 60.130719 | 68.954248 |
| Gradient Boosting   | 98.692810     | 99.019608  | 99.346405  | 82.026144 | 98.692810 |

---

## Best Sampling Technique per Model

* Logistic Regression → Stratified Sampling
* Decision Tree → Stratified Sampling
* Random Forest → Stratified / Systematic / Bootstrap (tie)
* SVM → Systematic Sampling
* Gradient Boosting → Systematic Sampling

---

## Key Observations

* Stratified Sampling performs strongly for **Logistic Regression, Decision Tree, and Random Forest**.
* Systematic Sampling performs best for **SVM and Gradient Boosting**.
* Cluster Sampling shows consistently lower performance in this implementation.
* KMeans-based cluster selection may reduce dataset representativeness.

---

## Notes

This implementation uses **accuracy** as the evaluation metric to match assignment requirements.

For real-world imbalanced classification problems, additional metrics should be considered:

* Precision
* Recall
* F1-score
* ROC-AUC

---

## Author

**Vansh Singla**
