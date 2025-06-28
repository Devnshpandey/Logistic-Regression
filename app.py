# app.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import io

# Load data
@st.cache_data
def load_data(path='Titanic_train.csv'):
    df = pd.read_csv(path)
    return df

# Preprocess data
def preprocess(df):
    df = df.copy()
    df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], inplace=True)
    df.fillna({'Age': df['Age'].median(), 'Embarked': df['Embarked'].mode()[0]}, inplace=True)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = LabelEncoder().fit_transform(df['Embarked'])
    return df

# Train model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# App title
st.title("🚢 Titanic Survival Predictor")

# Load and prepare data
df_raw = load_data()
df = preprocess(df_raw)
X, y = df.drop('Survived', axis=1), df['Survived']
model, X_test, y_test = train_model(X, y)

# Sidebar for user inputs
st.sidebar.header("Input Passenger Features")
input_features = {}
for col in X.columns:
    min_val, max_val, med_val = X[col].min(), X[col].max(), X[col].median()
    input_features[col] = st.sidebar.slider(col, float(min_val), float(max_val), float(med_val))

input_df = pd.DataFrame([input_features])

# Prediction
prediction = model.predict(input_df)[0]
prob = model.predict_proba(input_df)[0][1]
st.subheader("Prediction Result")
st.markdown(f"**Prediction:** {'🟩 Survived' if prediction == 1 else '🟥 Did Not Survive'}")
st.markdown(f"**Survival Probability:** {prob:.2f}")

# ROC Curve
st.subheader("Model Evaluation")
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

fig, ax = plt.subplots()
sns.lineplot(x=fpr, y=tpr, label=f'AUC = {roc_auc:.2f}', ax=ax)
sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', color='red', ax=ax)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
st.pyplot(fig)

# Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test))
st.write("**Confusion Matrix:**")
st.dataframe(pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1']))