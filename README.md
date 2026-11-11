# Credit Card Fraud Detection Using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.14-blue)
![License](https://img.shields.io/badge/License-IIT_RPR-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

Classifies credit card transactions as legitimate (`0`) or fraudulent (`1`) on an imbalanced dataset using Logistic Regression and Python (`pandas`, `scikit-learn`, `seaborn`, `imbalanced-learn`).

---

## Team & Roles

| Member Name | Entry No. | Role | GitHub Handle |
| :--- | :--- | :--- | :--- |
| **Ravicharan** | `2026ESB1479` | Documentation Lead | [@Ravicharan02](https://github.com/Ravicharan02) |
| **Krishna** | `2026EEB1368` | Execution Lead | [@Krxna04](https://github.com/Krxna04) |
| **Roshini** | `2026MCB1578` | PPT Lead | [@rose-2030](https://github.com/rose-2030) |
| **Sidhart** | `2026AIB1093` | Video Lead | [@ravansidharth-cpu](https://github.com/ravansidharth-cpu) |

---

## 1. Project Description

Credit Card Fraud Detection is a machine learning project that aims to identify fraudulent credit card transactions using **Logistic Regression**.

The project uses transaction data to classify each transaction into one of two categories:
* **0** – Legitimate Transaction
* **1** – Fraudulent Transaction

The main objective is to build a machine learning model that can detect fraudulent transactions accurately while minimizing incorrect predictions.

---

## 2. Problem Statement

Credit card fraud is a major concern in digital financial transactions. Detecting fraudulent transactions is challenging because fraudulent transactions represent only a very small portion of all transactions (~0.17%).

This project uses machine learning techniques to analyze transaction patterns and identify potentially fraudulent activities.

---

## 3. Objectives

* To analyze and understand the credit card transaction dataset.
* To perform Exploratory Data Analysis (EDA).
* To preprocess and prepare the data for machine learning.
* To handle the class imbalance present in the dataset.
* To train a Logistic Regression classification model.
* To evaluate the model using appropriate performance metrics.
* To identify fraudulent transactions effectively.

---

## 4. Dataset

* **Dataset Name:** Credit Card Fraud Detection Dataset
* **Source:** [Kaggle - Credit Fraud: Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook)
* **File Name:** `creditcard.csv`
* **Target Column:** `Class` (`0` = Normal transaction, `1` = Fraudulent transaction)
* **Features:** Contains `Time`, `Amount`, and anonymized numerical features (`V1`–`V28`).

> **Note:** The dataset is not included in this repository because of its large file size. Download `creditcard.csv` from Kaggle and place it inside the `data/` directory before running the scripts.

---

## 5. Technologies Used

* Python 3.14
* Jupyter Notebook
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Git and GitHub
* nbformat
* os library
---

## 6. Machine Learning Algorithm

**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm used for binary classification. In this project, it is used to classify credit card transactions as legitimate or fraudulent. Class balancing techniques (e.g., undersampling / SMOTE) are applied to address the severe imbalance between classes.

---

## 7. Project Workflow

1. Dataset collection and loading.
2. Data exploration and statistical analysis.
3. Exploratory Data Analysis (EDA).
4. Data preprocessing and feature scaling (`Time` and `Amount`).
5. Splitting the dataset into training and testing sets.
6. Training the Logistic Regression model.
7. Predicting transaction classes.
8. Evaluating model performance.
9. Visualizing and exporting evaluation results into `outputs/` or `results/plots/`.

---
## 8. How to Run the Project

Using Jupyter Notebook
1. Open the project directory in VS Code or Jupyter Notebook.
2. Download the dataset from the given drive link in the data folder and save it in the same folder of the cloned repository.
3. Open the src folder
4. Open fraud_detection.ipynb.
5. Make sure the dataset path is correct.
6. Select the Python kernel.
7. Run all cells sequentially.
8. Observe the model training, predictions, and evaluation results.
--- 

## 9. Model Evaluation

The model is evaluated using the following performance metrics:
* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC-AUC Score

---

## 10. Project Structure

```text
AI101/
├── data/                      # Dataset folder
│   └── README.txt             # Instructions to download creditcard.csv
├── outputs/                   # Saved visual figures and generated outputs
│   └── placeholder
├── src/                       # Source scripts and notebooks
│   ├── data_exploration.ipynb # EDA notebook
│   ├── evaluation.ipynb       # Model evaluation notebook
│   ├── main.py                # Main pipeline entry point
│   ├── model_training.ipynb   # Model training notebook
│   └── preprocessing.ipynb    # Preprocessing notebook
├── .gitignore                 # Git ignore file
├── README.md                  # Main project documentation
└── requirements.txt           # Pinned dependencies
```
## 📂 Folder Descriptions

| Folder / File | Description |
| :--- | :--- |
| **`data/`** | Contains dataset location setup and `README.txt` instructions to download `creditcard.csv`. |
| **`outputs/`** | Stores generated plots, confusion matrices, and model evaluation charts. |
| **`src/`** | Contains Python pipeline scripts (`main.py`) and exploratory analysis Jupyter Notebooks. |
| **`requirements.txt`** | Lists all required Python dependencies and package versions. |
| **`README.md`** | Main repository documentation containing project details, setup, and team roles. |
| **`.gitignore`** | Specifies untracked files (e.g., dataset files, virtual environments) to ignore in Git. |

---
## 10. Installation and Setup

### Step 1: Clone the repository
```bash 
git clone https://github.com/Krxna04/AI101.git
```
### Step 2: Navigate to the project directory
```
cd AI101
```
### Step 3: Create and activate a virtual environment
```
python -m venv venv
```
# On Windows:
venv\Scripts\activate
```
# On macOS/Linux:
source venv/bin/activate
```
### Step 4: Install the required libraries
```
pip install -r requirements.txt
```
### Step 5: Download the dataset
```
Download creditcard.csv from Kaggle and place it inside the data/ directory:
AI101/data/readme.txt
```
---


