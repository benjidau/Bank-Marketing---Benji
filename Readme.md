# **Bank Marketing Prediction Project**

Using Machine Learning to Predict Customer Term Deposit Subscription

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
* **Joblib** – model saving
* **Jupyter Notebook** – development environment

### **Machine Learning Techniques**

* Logistic Regression
* Decision Trees
* Random Forest
* Gradient Boosting (XGBoost / LightGBM optional)
* Hyperparameter tuning using GridSearchCV