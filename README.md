# 📊 Customer Churn Prediction Dashboard

A Machine Learning web application built using **Python**, **Scikit-learn**, and **Streamlit** to predict customer churn. The dashboard provides churn predictions, probability scores, customer details, business recommendations, model insights, and downloadable customer information through an interactive interface.

---

## 🚀 Features

- 🔍 Predict customer churn using a trained Random Forest model
- 📈 Display probability of staying and churning
- 👤 View customer details by selecting a Customer ID
- 📋 Download customer details
- 🖨️ Print prediction report (Save as PDF using browser print)
- 💡 Business recommendations based on churn risk
- 📊 View model accuracy
- 📉 Visualize Top 10 Important Features
- 🎨 Interactive Streamlit dashboard

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn_model.pkl
├── accuracy.pkl
├── feature_importance.pkl
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Machine Learning Workflow

1. Load the telecom customer churn dataset
2. Handle missing values
3. Encode categorical variables
4. Train a Random Forest Classifier
5. Evaluate model performance
6. Save the trained model using Joblib
7. Build an interactive Streamlit dashboard

---

## 💻 Dashboard Features

### Customer Prediction
- Predict whether a customer will churn
- Probability of Staying
- Probability of Churning

### Customer Information
- Display selected customer details
- Download customer details

### Recommendation
- Business recommendations based on customer risk level

### Model Insights
- Random Forest Accuracy
- Top 10 Important Features

---

## ▶️ How to Run

Clone the repository

```bash
git clone https://github.com/saisarvani16/Customer-Churn-Prediction.git
```

Move into the project folder

```bash
cd Customer-Churn-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Dashboard Preview

> Add screenshots of your dashboard here after uploading them.

---

## 📌 Future Enhancements

- Deploy on Streamlit Community Cloud
- Compare multiple machine learning models
- Advanced customer analytics dashboard
- Enhanced visualizations
- Search customer by multiple attributes

---

## 👩‍💻 Author

**Sai Sarvani Kota**

GitHub: https://github.com/saisarvani16

---

⭐ If you found this project useful, feel free to star the repository!
