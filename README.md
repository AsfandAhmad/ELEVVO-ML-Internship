# 🚀 ELEVVO ML Internship Projects

This repository contains **5 Machine Learning projects** covering regression, recommendation systems, clustering, deep learning, and tree-based models.
Each project demonstrates a different ML concept with real datasets.

---

## 📂 Projects Overview & Run Instructions

### 1️⃣ 🎓 Student Score Prediction

* **Goal:** Predict exam scores based on study hours and lifestyle factors.
* **Approach:** Linear Regression, Polynomial Regression, Multi-feature Regression.
* **Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
* **Run:**

```bash
python3 src/student_score.py
```

---

### 2️⃣ 🎬 Movie Recommendation System

* **Goal:** Recommend movies using the **MovieLens 100K dataset**.
* **Approach:** User-based CF, Item-based CF, and Matrix Factorization (SVD).
* **Tools:** Python, Pandas, Surprise/Scikit-learn, Matplotlib.
* **Run:**

```bash
# User-based CF
python3 main.py --data_dir data/ml-100k --model user --K 10 --min_relevant 4.0 --k_neighbors 50

# Item-based CF
python3 main.py --data_dir data/ml-100k --model item --K 10 --k_sim 50

# Matrix Factorization (SVD)
python3 main.py --data_dir data/ml-100k --model svd --K 10 --svd_components 50

# Compare all models
python3 evaluate_models.py
```

---

### 3️⃣ 🚦 Traffic Sign Recognition

* **Goal:** Classify traffic sign images into 43 classes.
* **Approach:** CNNs with TensorFlow/Keras.
* **Tools:** Python, TensorFlow/Keras, Matplotlib, NumPy.
* **Run:**

```bash
# Train
python -m src.training.train --data_dir data/processed --img_size 64 --batch_size 64 --epochs 20

# Evaluate
python -m src.training.evaluate --data_dir data/processed --model_path outputs/checkpoints/best_model.keras

# Predict single image
python predict.py
```

---

### 4️⃣ 🛍️ Mall Customer Clustering

* **Goal:** Segment mall customers based on income & spending score.
* **Approach:** K-Means & DBSCAN with Silhouette Score for cluster selection.
* **Tools:** Python, Pandas, Scikit-learn, Matplotlib.
* **Run:**

```bash
python3 src/cluster.py --data data/Mall_Customers.csv
```

---

### 5️⃣ 🌲 Forest Cover Type Classification

* **Goal:** Predict forest cover types using the **UCI Covertype dataset**.
* **Approach:** Random Forest & XGBoost with hyperparameter tuning.
* **Tools:** Python, Pandas, Scikit-learn, XGBoost, Matplotlib.
* **Run:**

```bash
# Train both models
python3 train_covertype.py --model both

# Train only Random Forest
python3 train_covertype.py --model rf

# Train only XGBoost
python3 train_covertype.py --model xgb

# With hyperparameter tuning
python3 train_covertype.py --tune --model both
```

---

## 🛠️ Common Setup Instructions

Clone the repo:

```bash
git clone https://github.com/AsfandAhmad/ELEVVO-ML-Internship.git
cd ELEVVO-ML-Internship
```

Create environment & install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate   # (use .venv\Scripts\activate on Windows)
pip install -r requirements.txt
```

---

## 📊 Tools Used Across Projects

* **Python** 🐍
* **Libraries:** Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Matplotlib, Seaborn, XGBoost
* **Concepts Covered:** Regression, Collaborative Filtering, CNNs, Clustering, Tree-based Models

---

✨ This repo demonstrates a **wide range of ML techniques**, from simple regressions to deep learning and recommendation systems.
