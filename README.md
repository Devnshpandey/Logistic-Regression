# 🚢 Titanic Survival Predictor Web App

This is a Streamlit-based web application that uses logistic regression to predict the survival of Titanic passengers. The model is trained on historical data and allows users to input passenger attributes to receive a survival prediction and corresponding probability.

## 🔍 Features

- Predict survival using **logistic regression**
- Interactive **Streamlit sliders** for inputting passenger data
- **ROC Curve** and **AUC score** for model performance evaluation
- **Confusion Matrix** for understanding prediction results

## 📁 Files in This Repository

- `app.py` – Streamlit application script
- `Titanic_train.csv` – Training dataset
- `requirements.txt` – Python dependencies
- `README.md` – Project documentation

## 📦 Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/titanic-predictor-app.git
   cd titanic-predictor-app
   ```

2. (Optional but recommended) Set up a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Running the App Locally

```bash
streamlit run app.py
```

## 🌐 Live Demo

Access the deployed app here:  
**[insert your Streamlit Cloud URL once deployed]**

## 📊 Input Features Used

- Pclass (Ticket class)
- Sex (Gender)
- Age
- SibSp (Siblings/Spouses aboard)
- Parch (Parents/Children aboard)
- Fare
- Embarked (Port of embarkation)

## ✅ Model & Performance

- Model: Logistic Regression (`scikit-learn`)
- ROC AUC Score displayed on web UI
- Visual diagnostics included for interpretability

## 👨‍💻 Author

**Devansh** – passionate about accessible machine learning, data storytelling, and deploying clean, interactive apps for real-world impact.
