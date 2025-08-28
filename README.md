# 📊 Employee Salary Classification

This project uses machine learning models to predict whether an employee earns more or less than $50K/year based on census data (adult dataset). It includes preprocessing, model comparison, outlier handling, and a Streamlit web app interface for both individual and batch predictions.

## 📁 Dataset

- **Source:** [adult 3.csv](./adult%203.csv)
- **Description:** Contains demographic and employment-related attributes (age, education, occupation, etc.) used for salary prediction.
- **Target:** Whether income is `<=50K` or `>50K`.

- ## ⚙️ Features Used

- Age  
- Education  
- Occupation  
- Hours-per-week  
- Years of experience  


## 🧪 Exploratory Data Analysis

### 📌 With Outliers  
![Outlier Boxplot](https://github.com/Athulyakrishna-k1312/salary-predictor/blob/1440a50c6eba3aa017ac0749ef1fa13508931fd8/images/pic1.png)
### 📌 Without Outliers  
![Cleaned Boxplot](https://github.com/Athulyakrishna-k1312/salary-predictor/blob/b370299f11d01539eeeca34c228338eceaa3de7e/images/pic2.png)

---

## 🤖 Model Performance

### 📈 Accuracy Comparison  
![Model Comparison Bar Chart](https://github.com/Athulyakrishna-k1312/salary-predictor/blob/0308b329cad37bdbcdee031715196469b4f4c8b1/images/pic3.png)

### 📊 Classification Reports  
![Classification Report](https://github.com/Athulyakrishna-k1312/salary-predictor/blob/0308b329cad37bdbcdee031715196469b4f4c8b1/images/pic4.png)

### ✅ Best Performing Model  
![Best Model Highlight](https://github.com/Athulyakrishna-k1312/salary-predictor/blob/0308b329cad37bdbcdee031715196469b4f4c8b1/images/pic5.png)

---
## 🖥️ Web App Interface

This project includes a **Streamlit** app to classify employee salaries:

- Single Prediction via form input
- Batch Prediction via file upload

### 🔍 App Screenshot  
![App UI](https://github.com/Athulyakrishna-k1312/salary-predictor/blob/fd8d3ba194d9640d3ce553eff578a560fd374e61/images/Screenshot%20from%202025-07-22%2012-00-56.png)

---
 
## 🌐 Live App

🔗 [Click here to try it online](https://salary-predictor-cvfpndnysxwgbibtgtmpmc.streamlit.app/)

---



