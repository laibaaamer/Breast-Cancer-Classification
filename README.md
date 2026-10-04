# 🩺 Breast Cancer Classification

A machine learning project for classifying breast tumors as **Benign** or **Malignant** using the Breast Cancer Wisconsin dataset. The project covers data preprocessing, feature scaling, model training, hyperparameter tuning, and performance evaluation.

## 📌 Project Overview

The goal of this project is to build a reliable machine learning classification model that can distinguish between benign and malignant breast tumors based on extracted tumor features.

The project also compares different hyperparameter optimization approaches to evaluate model performance.

## 🎯 Objectives

* Load and explore the breast cancer dataset
* Perform data preprocessing
* Scale numerical features
* Split data into training and testing sets
* Train a machine learning classification model
* Perform hyperparameter tuning using:

  * GridSearchCV
  * RandomizedSearchCV
* Evaluate model performance using classification metrics
* Compare the performance of different tuning strategies

## 📊 Dataset

This project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, which contains numerical features calculated from digitized images of breast mass samples.

The target variable represents:

* **0 — Malignant**
* **1 — Benign**

The features describe characteristics of the cell nuclei, such as:

* Radius
* Texture
* Perimeter
* Area
* Smoothness
* Compactness
* Concavity
* Concave points
* Symmetry
* Fractal dimension

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ├── GridSearchCV
   └── RandomizedSearchCV
   ↓
Model Evaluation
   ↓
Performance Comparison
```

## ⚙️ Hyperparameter Tuning

Two optimization techniques were explored:

### GridSearchCV

GridSearchCV evaluates all specified combinations of hyperparameters using cross-validation to find the best-performing configuration.

### RandomizedSearchCV

RandomizedSearchCV evaluates a selected number of random hyperparameter combinations, making it more efficient when the search space is large.

## 📈 Results

The models were evaluated using cross-validation and test-set performance.

| Method             | CV Score | Test Score |
| ------------------ | -------: | ---------: |
| GridSearchCV       |   0.9582 |     0.9561 |
| RandomizedSearchCV |   0.9604 |     0.9474 |

### Key Observation

GridSearchCV achieved a test accuracy of approximately **95.61%**, while RandomizedSearchCV achieved approximately **94.74%**.

The results demonstrate how hyperparameter optimization can affect the generalization performance of a machine learning model.

## 📏 Evaluation Metrics

The project uses standard classification metrics, including:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Cross-validation score

These metrics provide a broader understanding of model performance beyond accuracy alone.

## 📁 Project Structure

```text
breast-cancer-classification/
│
├── breast_cancer_classification.py
├── requirements.txt
├── README.md
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/breast-cancer-classification.git
```

Move into the project directory:

```bash
cd breast-cancer-classification
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

If using the Python script:

```bash
python breast_cancer_classification.py
```

## 📚 Key Learning Outcomes

Through this project, I practiced:

* Exploratory Data Analysis
* Data preprocessing
* Feature scaling
* Supervised machine learning
* Classification
* Cross-validation
* Hyperparameter optimization
* GridSearchCV
* RandomizedSearchCV
* Model evaluation
* Performance comparison
* Interpreting classification metrics

## 🔮 Future Improvements

Possible improvements include:

* Testing additional classification algorithms
* Applying ensemble learning methods
* Performing more extensive feature selection
* Using advanced hyperparameter optimization techniques
* Adding ROC-AUC and Precision-Recall curves
* Deploying the trained model as a web application

## 👩‍💻 Author

**Laiba Aamer**

BS Artificial Intelligence Student


## 📄 License

This project is created for educational and learning purposes.
