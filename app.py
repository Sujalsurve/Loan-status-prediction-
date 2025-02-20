import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
with open("loan_kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load dataset for reference (optional for visualization)
df = pd.read_csv("loan_data.csv")

# Streamlit App Title
st.title("Loan Status Prediction App")
st.write(
    "This app predicts whether a loan application will be **Approved** or **Rejected** using KMeans clustering."
)

# Input Section
st.sidebar.header("Input Applicant Details")
income = st.sidebar.slider("Applicant Income (in $):", 2000, 10000, 5000)
loan_amount = st.sidebar.slider("Loan Amount (in $):", 50, 500, 250)
credit_history = st.sidebar.selectbox(
    "Credit History (0: No, 1: Yes):", [0, 1], index=1
)

# Display raw data on demand
if st.sidebar.checkbox("Show Raw Dataset"):
    st.subheader("Loan Dataset Preview")
    st.write(df.head())

# Prediction Button
if st.button("Predict Loan Status"):
    # Prepare input features
    features = np.array([[income, loan_amount, credit_history]])
    
    # Predict the cluster
    prediction = model.predict(features)[0]
    status = "Approved" if prediction == 1 else "Rejected"
    
    # Display prediction
    st.success(f"The predicted loan status is: **{status}**")

    # Additional explanation
    if status == "Approved":
        st.write(
            "Congratulations! Based on the provided details, your loan application is likely to be approved."
        )
    else:
        st.write(
            "Unfortunately, based on the provided details, your loan application is likely to be rejected. Consider improving your credit history or reducing the loan amount."
        )

# Visualizations Section
st.sidebar.header("Visualizations")
if st.sidebar.checkbox("Show Loan Status Distribution"):
    st.subheader("Loan Status Distribution")
    sns.countplot(x="Loan_Status", data=df, palette="viridis")
    st.pyplot(plt.gcf())  # Display the chart in Streamlit

if st.sidebar.checkbox("Show Income vs Loan Amount"):
    st.subheader("Income vs Loan Amount by Loan Status")
    sns.scatterplot(
        x="ApplicantIncome",
        y="LoanAmount",
        hue="Loan_Status",
        data=df,
        palette="coolwarm",
    )
    st.pyplot(plt.gcf())  # Display the chart in Streamlit
