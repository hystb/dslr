# DSLR – Data Science Logistic Regression

A from-scratch implementation of a multi-class logistic regression model to classify Hogwarts students into their respective houses based on academic performance.

## 📊 Project Overview

This project aims to predict a student's **Hogwarts House** based on their scores in various courses using a self-built logistic regression model (One-vs-All strategy). It emphasizes understanding the full ML pipeline without relying on high-level machine learning libraries.

## 🔧 Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Unix/macOS
venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

### Train the model

```bash
python logreg_train.py datasets/dataset_train.csv
```

Trains a One-vs-All logistic regression model and exports it to `dslr.model`.

### Make predictions

```bash
python logreg_predict.py datasets/dataset_test.csv dslr.model
```

Outputs predictions in a `houses.csv` file.

### Generate statistics

```bash
python describe.py datasets/dataset_train.csv
```

### Visualizations

```bash
python histogram.py datasets/dataset_train.csv
python scatter_plot.py datasets/dataset_train.csv
python pair_plot.py datasets/dataset_train.csv
```

## 📈 Features

- Pure Python implementation of One-vs-All logistic regression
- Mean-based imputation for missing data
- Data normalization before training
- CLI tools for training, prediction, and visualization
- Dataset-agnostic design (assuming similar structure)

## 📦 Dataset

You will need:

- `datasets/dataset_train.csv` — training data with house labels
- `datasets/dataset_test.csv` — test data without labels

Each row contains a student's name, house, and course grades. Only numeric features are used for training (some columns are excluded).

## 🧠 Model

The model uses a **One-vs-All logistic regression** approach, trained on standardized numeric features while excluding non-informative ones such as `"Birthday"`, `"Best Hand"`, etc.
