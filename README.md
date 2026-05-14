# sibyl-of-cumae
# Machine Learning Portfolio

A collection of applied machine learning and NLP projects focused on predictive modelling, unsupervised learning, feature engineering, and real-world analytical systems.

---

## Featured Project: PureGym Customer Review NLP Analysis

Developed an end-to-end NLP pipeline to analyse customer reviews from Google and Trustpilot, identify operational pain points, and generate actionable recommendations for improving customer experience.

### Objective
Identify the key drivers of negative customer sentiment across PureGym locations using advanced NLP and topic modelling techniques.

### Approach
- Dataset: ~40,000 reviews across Google and Trustpilot
- Data cleansing and standardisation:
  - text cleaning
  - tokenisation
  - stopword removal
  - sentiment filtering
- Methods:
  - Word frequency analysis and word clouds
  - BERT emotion classification
  - BERTopic topic modelling
  - Gensim LDA topic modelling
  - Phi-4 LLM topic extraction and summarisation
- Geographic hotspot analysis across gym locations
- Topic aggregation and operational recommendation generation

### Key Results
- Negative sentiment concentrated heavily around:
  - cleanliness and hygiene
  - staff behaviour
  - equipment availability
  - overcrowding
- BERTopic produced the clearest operational themes when applied to anger-filtered reviews
- LLM summarisation successfully converted raw customer complaints into actionable operational recommendations
- London and other high-footfall urban locations showed the highest concentration of negative reviews

### Key Insight
Combining transformer-based topic modelling with LLM summarisation creates a highly effective workflow for converting large-scale unstructured customer feedback into operationally actionable business intelligence.

---

## Project: Student Dropout Prediction

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

---

## Tech Stack

Python, pandas, scikit-learn, XGBoost, TensorFlow/Keras, BERTopic, Hugging Face Transformers, Gensim, matplotlib

---

## Repository Structure

- `puregym_nlp_analysis.ipynb` — NLP, topic modelling, and LLM analysis
- `student_dropout_prediction.ipynb` — supervised modelling (classification)
- `anomaly_detection_engine.ipynb` — unsupervised anomaly detection
- `customer_segmentation.ipynb` — clustering and segmentation

---

## Next Projects

- Time series forecasting
- Recommendation systems
- Graph-based ML and probabilistic models
- Feature engineering pipelines at scale

---
