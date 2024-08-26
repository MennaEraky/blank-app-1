import streamlit as st
import pickle
import numpy as np

# Load the model
try:
    with open('model6.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit App
st.title("Decision Tree Prediction")

# Input features
st.sidebar.header("Input Features")
feature_1 = st.sidebar.number_input("Pregnancies",value=10.0)
feature_2 = st.sidebar.number_input("Glucose",value=10.0)
feature_3 = st.sidebar.number_input("Insulin")
feature_4 = st.sidebar.number_input("BMI")
feature_5 = st.sidebar.number_input("DiabetesPedigreeFunction")
feature_6 = st.sidebar.number_input("Age")

# Prepare the input data
#input_data = np.array([[feature_1, feature_2,feature_3,feature_4,feature_5,feature_6]])
# Test input
input_data = np.array([[10.0, 10.0, 50.0, 30.0, 0.5, 25.0]])  # Adjust values accordingly

# Make prediction
#prediction = model.predict(input_data)
#print("Predicted Output:", prediction[0])

# Check if the input data has missing values
if np.any(np.isnan(input_data)):
    st.error("Input data contains missing values. Please provide valid input.")
else:

    # Make prediction
    prediction = model.predict(input_data)
    st.write("### Predicted Output")
    st.write(prediction[0])

