# sibyl-of-cumae

## Machine Learning Portfolio

A collection of applied machine learning, deep learning, time-series forecasting, and NLP projects focused on predictive modelling, feature engineering, unsupervised learning, and real-world analytical systems.

---

## Featured Project: PureGym Customer Review NLP Analysis

Developed an end-to-end NLP pipeline to analyse customer reviews from Google and Trustpilot, identify operational pain points, and generate actionable recommendations for improving customer experience.

### Objective

Identify the key drivers of negative customer sentiment across PureGym locations using advanced NLP and topic-modelling techniques.

### Approach

- Analysed approximately 40,000 Google and Trustpilot reviews
- Cleaned, tokenised, and standardised review text
- Applied sentiment filtering and word-frequency analysis
- Used BERT for emotion classification
- Compared BERTopic and Gensim LDA topic models
- Used Phi-4 for topic extraction and summarisation
- Conducted geographic hotspot analysis across gym locations

### Key Results

- Negative sentiment concentrated around cleanliness, staff behaviour, equipment availability, and overcrowding
- BERTopic produced the clearest operational themes when applied to anger-filtered reviews
- LLM summarisation converted customer complaints into actionable operational recommendations
- London and other high-footfall urban locations showed the greatest concentration of negative reviews

### Key Insight

Combining transformer-based topic modelling with LLM summarisation provides an effective workflow for converting large-scale customer feedback into actionable business intelligence.

[View notebook](topic_modelling_for_pure-gym.ipynb)

---

## Project: Book Sales and Demand Forecasting

Developed and compared statistical, machine-learning, deep-learning, and hybrid models to forecast Nielsen BookScan sales for *The Alchemist* and *The Very Hungry Caterpillar*.

### Objective

Forecast future book sales and identify suitable modelling strategies to support procurement, reordering, stock control, and reprinting decisions.

### Approach

- Resampled weekly sales data and handled missing observations
- Analysed trend, seasonality, stationarity, and autocorrelation
- Applied time-series decomposition, ADF, KPSS, ACF, and PACF analysis
- Compared Auto ARIMA/SARIMA, XGBoost, and LSTM models
- Developed sequential and parallel SARIMA–LSTM hybrid models
- Tuned models using grid search, expanding-window validation, and KerasTuner
- Evaluated weekly 32-week and monthly eight-month forecasts using MAE and MAPE

### Key Results

- For *The Alchemist*, tuned XGBoost achieved the lowest MAE of **138.41**
- The sequential SARIMA–LSTM hybrid achieved the best proportional accuracy for *The Alchemist*, with a MAPE of **21.62%**
- For *The Very Hungry Caterpillar*, the parallel SARIMA–LSTM hybrid achieved the lowest MAE of **442.75**
- A stacked LSTM achieved the best proportional accuracy for *The Very Hungry Caterpillar*, with a MAPE of **22.60%**
- Machine-learning and hybrid approaches substantially outperformed Auto ARIMA on the more irregular sales series
- Monthly aggregation generally weakened performance by reducing training data and smoothing important demand spikes

### Key Insight

No single model performed best across both books. Book-specific model selection produced the strongest results, with hybrid SARIMA–LSTM models offering the most robust overall approach and XGBoost providing a strong, maintainable benchmark.

[View notebook](book_sales_demand_forecasting.ipynb)

---

## Project: Student Dropout Prediction

Built supervised machine-learning models to predict student dropout using staged datasets that progressively introduced richer academic and engagement features.

### Objective

Predict student dropout accurately and identify the factors driving model performance.

### Approach

- Engineered academic progression, attendance, and engagement features
- Applied encoding, scaling, and staged dataset construction
- Compared XGBoost with baseline, tuned, and deep neural networks
- Evaluated performance using accuracy, precision, recall, F1 score, and ROC-AUC

### Key Results

- Stage 3 models achieved an ROC-AUC of approximately **0.999**
- XGBoost slightly outperformed the neural networks on tabular data
- Academic progression and attendance patterns were the strongest predictors

### Key Insight

Feature quality had a substantially greater effect on performance than model complexity or hyperparameter tuning.

[View notebook](predicting_student_drop-out.ipynb)

---

## Project: Customer Segmentation with Clustering

Applied unsupervised learning to segment customers from a large-scale e-commerce dataset, supporting more targeted marketing strategies.

### Objective

Identify meaningful customer segments based on purchasing behaviour.

### Approach

- Aggregated 951,669 transactions into approximately 63,800 customer profiles
- Engineered behavioural features including frequency, recency, and customer lifetime value
- Used the elbow method, silhouette scores, hierarchical clustering, and K-Means
- Applied PCA and t-SNE for dimensionality reduction and visualisation

### Key Results

- Diagnostic methods indicated an optimal range of four to five clusters
- The five-cluster solution improved segmentation of high-value customers
- Clear differences emerged between high-frequency, high-value, and low-frequency customers
- The peak silhouette score was approximately **0.265**, indicating moderate cluster separation

### Key Insight

Customer behaviour exists on a continuum rather than in sharply separated groups. Segments should therefore be interpreted as useful behavioural profiles rather than rigid categories.

[View notebook](customer_segmentation_with_clustering.ipynb)

---

## Project: Anomaly Detection in Ship Engine Data

Developed an unsupervised anomaly-detection system to identify abnormal engine behaviour and support predictive maintenance.

### Objective

Detect anomalous engine activity without labelled training data.

### Approach

- Analysed 19,535 observations across six engine features
- Used the interquartile range for univariate outlier detection
- Applied One-Class SVM and Isolation Forest
- Scaled features where required
- Used PCA to visualise anomalies

### Key Results

- Univariate IQR analysis flagged approximately 21.6% of observations but was unsuitable for row-level classification
- Multivariate thresholding identified approximately 2.1% of observations as anomalies
- One-Class SVM and Isolation Forest produced anomaly rates within the expected 1–5% range
- Approximately 3.3% of observations were identified as anomalies by both models

### Key Insight

Unsupervised machine-learning methods outperformed univariate statistical rules by capturing relationships between engine variables. Isolation Forest provided the most practical balance of robustness, interpretability, and anomaly-rate control.

[View notebook](anomalies_in_ship_engine.ipynb)

---

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- TensorFlow and Keras
- KerasTuner
- statsmodels and pmdarima
- sktime
- BERTopic
- Hugging Face Transformers
- Gensim
- matplotlib and seaborn

---

## Repository Structure

| File | Description |
|---|---|
| [`topic_modelling_for_pure-gym.ipynb`](topic_modelling_for_pure-gym.ipynb) | NLP, topic modelling, emotion classification, and LLM analysis |
| [`book_sales_demand_forecasting.ipynb`](book_sales_demand_forecasting.ipynb) | Statistical, machine-learning, deep-learning, and hybrid time-series forecasting |
| [`predicting_student_drop-out.ipynb`](predicting_student_drop-out.ipynb) | Supervised classification and student dropout prediction |
| [`customer_segmentation_with_clustering.ipynb`](customer_segmentation_with_clustering.ipynb) | Customer clustering and behavioural segmentation |
| [`anomalies_in_ship_engine.ipynb`](anomalies_in_ship_engine.ipynb) | Unsupervised anomaly detection for ship engine data |

---

## Planned Projects

- Recommendation systems
- Graph-based machine learning
- Probabilistic modelling
- Feature-engineering pipelines at scale
