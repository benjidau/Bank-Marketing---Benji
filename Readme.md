# **Bank Marketing Prediction Project**

Using Machine Learning to Predict Customer Term Deposit Subscription

by **Benji Dau** (UID010044292)

---

## **Project Summary**

This project analyzes the **Bank Marketing Dataset** from the UCI Machine Learning Repository to predict whether a client will subscribe to a **term deposit** after a direct marketing campaign. Using statistical exploration and machine learning models, this project aims to identify key customer patterns and improve marketing decision-making.

---

## **Objectives**

* Understand the structure, distribution, and characteristics of client, campaign, and economic features.
* Perform data cleaning, preprocessing, and exploratory data analysis (EDA).
* Build and compare machine learning models to classify client subscription outcomes (`yes`/`no`).
* Identify the most important features affecting customer decision-making.
* Provide insights to help optimize future marketing strategies.

---

## **Dataset Description**

This project uses the **Bank Marketing Dataset** from the **UCI Machine Learning Repository**.

### **Dataset Source**

UCI Machine Learning Repository
 [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)

### **Files Included**

| File               | Description                       |
| ------------------ | --------------------------------- |
| **bank-full.csv**  | Full dataset (45,211 rows)        |
| **bank.csv**       | Smaller version (4,521 rows)      |
| **bank-names.txt** | Detailed description of variables |

### **Feature Groups**

#### **Client Attributes**

* Age, job, marital status, education
* Default, balance, housing loan, personal loan
#### **Marketing Campaign Attributes**

* Contact communication type (cellular/telephone)
* Month, day of week
* Number of contacts performed
* Days passed since last contact

#### **Economic Indicators**

* Employment variation rate
* Consumer price index
* Consumer confidence index
* Euribor 3-month rate
* Number of employees

#### **Target Variable**

* **`y`** → Whether the client subscribed to a term deposit (`yes` or `no`)

---

## **Tools and Libraries**

### **Programming Language**

* Python 3.x

### **Core Libraries**

* **NumPy** – numerical operations
* **Pandas** – data manipulation
* **Matplotlib / Seaborn** – data visualization
* **Scikit-Learn** – machine learning models & preprocessing
* **Pytorch** - deep learning package to build neural networks 
* **SHAP** - explain the output of machine learning model 
* **Jupyter Notebook** – development environment

### **Machine Learning Techniques**

* Logistic Regression
* Decision Trees
* Random Forest
* Gradient Boosting (XGBoost / LightGBM optional)
* Hyperparameter tuning using GridSearchCV

## The final report

The PDF file for final project: [Final Report](reports/ISOM835 Final Report - Benji.pdf)

## Link to Colab Notebook

[Colab Notebook](https://colab.research.google.com/drive/1p-rFkB44KdSyFoYSB4FExIRlCYahRyZp?usp=sharing)

## Instructions 

Follow the steps below to reproduce the full end-to-end analysis, from data loading to model training, evaluation, and interpretation.


## **1. Open the Google Colab Notebook**

The full workflow is implemented in a single Colab notebook.

**Open notebook:**
`https://colab.research.google.com/drive/1p-rFkB44KdSyFoYSB4FExIRlCYahRyZp?usp=sharing`

* copy to your Google Drive if applicable

## **2. Run All Cells (Recommended)**

In Colab:

**Runtime → Run all**

The notebook automatically performs:

* Data loading
* Data inspection and EDA
* Preprocessing & feature engineering
* Model training (Logistic Regression, Random Forest, XGBoost, MLP)
* Hyperparameter tuning
* SHAP interpretability
* Business insights & recommendations

All required libraries are installed within the notebook

## **3. Dataset Loading (Automatic)**

No manual download is required.

Running the notebook executes:

```python
from ucimlrepo import fetch_ucirepo
bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets
```


## **4. Data Preprocessing Pipeline**

Executing the preprocessing section will:

- Handle missing values
- Encode categorical features (One-Hot + Label Encoding)
- Normalize numerical features
- Engineer new features (date, weekend flag, age groups)
- Remove leakage variables (duration, balance)
- Create a stratified train/test split

No manual steps are needed.

## **5. Train the Machine Learning Models**

The notebook automatically trains:

* Logistic Regression
* Random Forest
* XGBoost (baseline and tuned)

To evaluate a model, the notebook runs commands like:

```python
evaluate_model("XGBoost", xgb_wrapper, X_test, y_test)
```

Metrics are printed directly in the output.

---

## **6. Interpret the Model with SHAP**

SHAP visualizations are generated automatically:

* SHAP summary plot (beeswarm)
* Per-feature impact
* Feature importance (Gain)

Example code executed:

```python
explainer = shap.TreeExplainer(xgb_best)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

## **7. Train Neural Network Models (Optional)**

Two PyTorch models are included:

1. **Original MLP** (unweighted BCE)
2. **Weighted MLP** (BCEWithLogitsLoss + pos_weight)

Simply run the NN section to:

* Train the models
* View loss curves
* Evaluate accuracy, precision, recall, F1
* Generate ROC-AUC plots

No additional configuration required.


## **8. Review Final Outputs**

At the end of the notebook, you will find:

- Model comparison tables
- Best-performing model summary
- Customer behavior insights
- Marketing recommendations
- Ethical and Responsible AI reflections

These outputs generate automatically when you run all cells.
