# 🚲 Bike Rental Demand Prediction using Machine Learning

This project aims to predict the number of bikes rented in Seoul using weather and temporal data. Accurate forecasting helps in optimizing inventory, improving customer satisfaction, and enhancing operational planning for bike-sharing services.

---

## 📌 Project Objective

The objective is to build an end-to-end machine learning pipeline that:
- Predicts hourly bike rental demand.
- Helps businesses manage resources and anticipate customer needs.
- Evaluates different regression models and chooses the best one for production use.

---

## 📊 Dataset Summary

- **Dataset Name:** Seoul Bike Sharing Demand
- **Total Rows:** 8,760
- **Total Columns:** 24
- **Target Variable:** `Rented_Bike_Count`
- **Features Include:**
  - Temperature, Humidity, Wind Speed, Rainfall, Snowfall
  - Solar Radiation, Visibility
  - Seasons, Holiday, Functioning Day
  - Hour, Date, and derived time features (Month, Day, Weekday)

---

## 🧪 ML Workflow

### 1. Data Preprocessing
- Checked for missing values & outliers.
- Feature manipulation:
  - Extracted `Hour`, `Month`, `Day`, `Weekday`, `Weekend`.
  - Created weather condition flags (rain, snow, sun).
  - Binned temperature into categories.

### 2. Feature Selection
- Removed irrelevant or highly correlated features.
- Selected features using correlation heatmaps and VIF.

### 3. Data Transformation
- Applied log transformation on the target to reduce skewness.
- Scaled numerical features using `StandardScaler` and `MinMaxScaler`.

### 4. Model Building
Tested the following regression models:
- Linear Regression
- Ridge Regression (with GridSearchCV)
- Random Forest Regressor (with GridSearchCV)
- **XGBoost Regressor** (final selected model)

---

## ✅ Final Model: XGBoost Regressor

### 🔧 Best Parameters (after GridSearchCV):
{
  'learning_rate': 0.1,
  'max_depth': 7,
  'n_estimators': 200,
  'subsample': 0.8
}

### **📈 Evaluation Metrics:**

| Metric                    | Score         |
| ------------------------- | ------------- |
| R² (R-squared)            | **0.9318**    |
| MAE (Mean Absolute Error) | **85.20**     |
| MSE (Mean Squared Error)  | **26,810.82** |


## **Business Impact**
- High Accuracy: XGBoost predicted with ~93% accuracy.

- MAE of 85.20: On average, the model predicts within 85 bike counts of the actual value.

- Usefulness: Helps reduce bike shortages/overages by optimizing demand prediction hour-by-hour.

### **Feature Importance (Using SHAP)**
SHAP values were used to interpret the final model:

- Top features influencing rental demand:

  - Hour

  - Temperature_C

  - Solar_Radiation_MJ_per_m2

  - Seasons_Summer

  - Holiday

SHAP visualizations helped make the model interpretable and explain decisions to stakeholders.

## **📂 Repository Structure**

├── data/
│   └── seoul_bike_rentals.csv
├── notebooks/
│   └── exploratory_analysis.ipynb
│   └── model_building.ipynb
├── src/
│   └── preprocessing.py
│   └── model_utils.py
├── main.py
├── README.md
├── requirements.txt
└── .gitignore


### **🛠 Tools & Libraries Used**
- Python 3.x

- pandas, numpy, matplotlib, seaborn

- scikit-learn

- xgboost

- shap

- GridSearchCV

### **Conclusion**
The XGBoost model outperformed all others and gave highly accurate predictions with R² ≈ 0.93. The pipeline is production-ready and can be deployed for real-time predictions, aiding smart city transportation planning and efficient bike rental service management.

### **👤 Author**
Gaurav Jangir
📧 jangirgaurav705@gmail.com
🎓 MSc Mathematics | AI/ML & Data Science Enthusiast




