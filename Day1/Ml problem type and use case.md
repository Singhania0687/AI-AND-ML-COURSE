# Machine Learning Problem Types, Algorithms & Use Cases

---

## 1. Supervised Learning

**Goal:** Learn a mapping from inputs to known outputs (labels).

| Subtype         | Description                 | Algorithms                                                                                         | Use Cases                                                |
|-----------------|-----------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------|
| Classification  | Predict discrete class labels | Logistic Regression, Decision Trees, Random Forest, SVM, KNN, Naive Bayes, XGBoost, Neural Nets | Spam detection, Image recognition, Fraud detection       |
| Regression      | Predict continuous values    | Linear Regression, Ridge/Lasso, SVR, Decision Trees, Random Forest Regressor, Neural Nets         | House price prediction, Sales forecasting, Stock prices  |

---

## 2. Unsupervised Learning

**Goal:** Find patterns or structure without labels.

| Subtype                | Description                      | Algorithms                              | Use Cases                                  |
|------------------------|---------------------------------|---------------------------------------|--------------------------------------------|
| Clustering             | Group similar data points        | K-Means, DBSCAN, Hierarchical, GMM    | Customer segmentation, Anomaly detection  |
| Dimensionality Reduction | Reduce features while preserving info | PCA, t-SNE, LDA, Autoencoders          | Visualization, Noise reduction, Compression|

---

## 3. Semi-Supervised Learning

**Goal:** Use small labeled + large unlabeled data.

| Description                | Algorithms                          | Use Cases                          |
|----------------------------|-----------------------------------|----------------------------------|
| Mix of supervised & unsupervised | Self-training, Graph-based, Semi-supervised SVM | Medical imaging, Text classification |

---

## 4. Reinforcement Learning (RL)

**Goal:** Learn decision-making to maximize reward.

| Description              | Algorithms                                    | Use Cases                               |
|--------------------------|-----------------------------------------------|----------------------------------------|
| Agent learns optimal actions | Q-Learning, Deep Q-Networks (DQN), Policy Gradient, Actor-Critic | Robotics, Games, Self-driving cars, Recommendations |

---

## 5. Anomaly Detection

**Goal:** Identify unusual patterns.

| Description           | Algorithms                        | Use Cases                             |
|-----------------------|---------------------------------|-------------------------------------|
| Detect rare/unusual events | Isolation Forest, One-Class SVM, Autoencoders, Statistical methods | Fraud detection, Intrusion detection |

---

## 6. Natural Language Processing (NLP)

**Goal:** Understand and generate text data.

| Subtype                 | Algorithms                          | Use Cases                               |
|-------------------------|-----------------------------------|----------------------------------------|
| Text Classification     | Naive Bayes, Logistic Regression, Transformers (BERT, GPT) | Spam filtering, Sentiment analysis     |
| Named Entity Recognition | CRF, Transformers                 | Extracting names, places, organizations|
| Language Modeling       | RNNs, Transformers                | Chatbots, Summarization, Translation   |

---

## 7. Time Series Forecasting

**Goal:** Predict future based on past time-ordered data.

| Algorithms                                  | Use Cases                               |
|---------------------------------------------|----------------------------------------|
| ARIMA, Exponential Smoothing, LSTM, Prophet | Stock prices, Weather, Demand forecasting |

---

# Summary Table

| Problem Type                  | Algorithms                              | Use Cases                               |
|------------------------------|---------------------------------------|----------------------------------------|
| Supervised (Classification)  | Logistic Regression, Random Forest, SVM | Email spam, Image recognition          |
| Supervised (Regression)      | Linear Regression, SVR, Decision Trees | House prices, Sales forecasting         |
| Unsupervised (Clustering)    | K-Means, DBSCAN, GMM                  | Customer segmentation, Anomaly detection|
| Unsupervised (Dim. Reduction)| PCA, t-SNE, Autoencoders              | Visualization, Noise reduction          |
| Semi-Supervised              | Self-training, Semi-supervised SVM     | Medical imaging, Text classification    |
| Reinforcement Learning       | Q-Learning, DQN, Policy Gradients     | Robotics, Game AI, Recommendations      |
| Anomaly Detection            | Isolation Forest, One-Class SVM       | Fraud, Intrusion detection              |
| NLP                         | Naive Bayes, Transformers             | Sentiment analysis, Chatbots            |
| Time Series Forecasting      | ARIMA, LSTM, Prophet                  | Stock forecasting, Weather prediction  |

---


