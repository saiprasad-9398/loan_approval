import streamlit as st

st.set_page_config(
    page_title="Loan Approval Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 Loan Approval Prediction System")

st.markdown("""
### End-to-End Machine Learning Project

This application predicts whether a loan will be *Approved* or *Rejected* using
classification models trained on historical banking data.

Use the sidebar to navigate through:
- 📌 Overview  
- 📊 EDA  
- 📈 Model Metrics  
- 🔮 Prediction  
""")
import streamlit as st
import pandas as pd

st.title("📌 Project Overview")

st.markdown("""
### Problem Statement
Banks need to decide whether a loan applicant is likely to repay the loan.
Manual decisions are subjective and slow.  
This project uses Machine Learning to automate the *Loan Approval Prediction* process.

### Objective
Predict whether a loan will be:
- *Approved (1)*
- *Rejected (0)*
""")

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

st.subheader("Dataset Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.subheader("Dataset Preview (Top 20 Rows)")
st.dataframe(df.head(20))

st.subheader("Summary Statistics")
st.dataframe(df.describe())
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis")

df = pd.read_csv("loan_approval_dataset.csv")
df.columns = df.columns.str.strip()

st.subheader("Loan Status Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x="loan_status", data=df, ax=ax1)
st.pyplot(fig1)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(10,6))
sns.heatmap(df.drop("loan_status", axis=1).corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

st.subheader("Key Insights")
st.markdown("""
- Higher *CIBIL score* leads to higher loan approval probability  
- Higher *income and asset values* increase approval chances  
- *Education* and *self-employment* have moderate impact  
- Asset features are correlated, justifying tree-based models
""")
import streamlit as st
import pandas as pd
from PIL import Image

st.title("📈 Model Performance & Evaluation")

metrics = pd.read_csv("model_metrics.csv")
st.subheader("Model Comparison Table")
st.dataframe(metrics)

st.subheader("Confusion Matrix")
st.image(Image.open("confusion_matrix.png"), use_column_width=True)

st.subheader("ROC Curve")
st.image(Image.open("roc_curve.png"), use_column_width=True)
import streamlit as st
import joblib
import numpy as np

st.title("🔮 Loan Approval Prediction")

model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.subheader("Enter Applicant Details")

user_input = []
for feature in features:
    value = st.number_input(feature, value=0.0)
    user_input.append(value)

if st.button("Predict Loan Status"):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.write(f"### Approval Probability: {probability:.2f}")

    if prediction == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")
