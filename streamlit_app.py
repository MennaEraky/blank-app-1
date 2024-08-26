import streamlit as st
import pickle
import numpy as np

# Load the model
st.sidebar.header("Model Loading")
try:
    with open('logistic.pkl', 'rb') as file:
        model = pickle.load(file)
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Streamlit App Title and Description
st.title("Diabetes Prediction App")
st.markdown("""
Welcome to the **Diabetes Prediction App**! This tool helps predict the likelihood of diabetes based on various health metrics. 
Please input the following information to receive your prediction.
""")

# Input features
st.sidebar.header("Input Features")
feature_1 = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=2, help="Number of times pregnant")
feature_2 = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=120, help="Glucose level in blood")
feature_3 = st.sidebar.number_input("Insulin", min_value=0, max_value=900, value=85, help="Insulin level in blood")
feature_4 = st.sidebar.number_input("BMI", min_value=0.0, max_value=70.0, value=32.0, help="Body Mass Index (weight in kg/(height in m)^2)")
feature_5 = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, help="Diabetes Pedigree Function")
feature_6 = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30, help="Age in years")

# Prepare the input data
input_data = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5, feature_6]])

# Check if the input data has missing values
if np.any(np.isnan(input_data)):
    st.error("Input data contains missing values. Please provide valid input.")
else:
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display the prediction result
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.error("The model predicts that the individual is likely to have diabetes.")
    else:
        st.success("The model predicts that the individual is unlikely to have diabetes.")

# Additional Features: Reset Button and Footer
if st.sidebar.button("Reset Input"):
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Menna Hesham Eraky")
