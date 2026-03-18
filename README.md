# sibyl-of-cumae
# Machine Learning Portfolio

A collection of applied machine learning projects focused on predictive modelling, feature engineering, and real-world datasets.

---

## Featured Project: Student Dropout Prediction

Built supervised machine learning models to predict whether a student will drop out, using staged datasets that progressively introduce richer features.

### Objective
Predict student dropout with high accuracy and identify the key drivers of model performance.

### Approach
- Feature engineering including module progression, attendance, and engagement metrics
- Preprocessing: encoding, scaling, and staged dataset construction
- Models:
  - XGBoost
  - Neural Networks (baseline, tuned, deep)
- Evaluation:
  - Accuracy, Precision, Recall, F1 Score, ROC-AUC

### Key Results
- Stage 3 models achieved AUC ≈ 0.999
- XGBoost slightly outperformed neural networks on tabular data
- Performance improvements were driven primarily by:
  - Academic progression (modules passed)
  - Attendance patterns (authorised/unauthorised absences)

### Key Insight
Model performance was driven significantly more by feature quality than model complexity or hyperparameter tuning.

---

## Project: Customer Segmentation with Clustering

Applied unsupervised learning techniques to segment customers from a large-scale e-commerce dataset, enabling targeted marketing strategies.

### Objective
Identify meaningful customer segments based on behavioural purchasing patterns.

### Approach
- Dataset: 951,669 transactions aggregated into ~63,800 customers
- Feature engineering:
  - Aggregation to customer-level behavioural features (frequency, recency, CLV, etc.)
- Methods:
  - Elbow Method (WCSS)
  - Silhouette Score
  - Hierarchical Clustering (dendrogram)
  - K-Means clustering
- PCA and t-SNE used for dimensionality reduction and cluster visualisation

### Key Results
- Optimal cluster range identified as 4–5 clusters across all diagnostic methods
- Silhouette scores indicated moderate cluster separation (peak ≈ 0.265)  
- 5-cluster solution improved segmentation granularity, particularly within high-value customers  
- Clear behavioural differentiation observed across clusters (e.g. high-frequency/high-CLV vs low-frequency segments)

### Key Insight
Customer behaviour exists on a continuum rather than in sharply defined groups. Clustering captures meaningful structure, but segmentation should be interpreted as probabilistic rather than strictly discrete.

---

## Tech Stack

Python, pandas, scikit-learn, XGBoost, TensorFlow/Keras, matplotlib

---

## Repository Structure

- `student_dropout_prediction.ipynb` — supervised modelling (classification)
- `anomaly_detection_engine.ipynb` — unsupervised anomaly detection
- `customer_segmentation.ipynb` — clustering and segmentation

---

## Next Projects

- Time series forecasting
- Recommendation systems
- Feature engineering pipelines at scale
---

## Project: Anomaly Detection in Ship Engine Data

Developed an anomaly detection system to identify abnormal engine behaviour in a shipping fleet, supporting predictive maintenance and operational reliability.

### Objective
Detect anomalous engine activity in the absence of labelled data.

### Approach
- Dataset: 19,535 observations across six engine features
- Methods:
  - Interquartile Range (IQR) for univariate outlier detection
  - One-Class SVM (OCSVM)
  - Isolation Forest
- Feature scaling applied where required (OCSVM)
- PCA used for dimensionality reduction and anomaly visualisation

### Key Results
- Univariate IQR flagged ~21.6% of observations as partially anomalous but failed at row-level classification
- Multivariate thresholding identified ~2.1% of observations as true anomalies  
- OCSVM and Isolation Forest successfully captured anomalies within the expected 1–5% range  
- Model agreement identified ~3.3% overlapping anomalies across methods

### Key Insight
Unsupervised machine learning methods significantly outperform statistical approaches by capturing multivariate relationships. Isolation Forest proved most practical due to robustness, interpretability, and direct control over anomaly rates.
