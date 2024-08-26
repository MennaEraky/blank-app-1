
import streamlit as st
import pickle
import numpy as np

# load pkl file
with open('model (2).pkl', 'rb') as file:
    model = pickle.load(file)


# Streamlit App
st.title("Decision Tree Prediction")

# Input features
st.sidebar.header("Input Features")
feature_1 = st.sidebar.number_input("Pregnancies")
feature_2 = st.sidebar.number_input("Glucose")
feature_3 = st.sidebar.number_input("Insulin")
feature_4 = st.sidebar.number_input("BMI")
feature_5 = st.sidebar.number_input("DiabetesPedigreeFunction")
feature_6 = st.sidebar.number_input("Age")


# Prepare the input data
input_data = np.array([[feature_1, feature_2, feature_3,feature_4,feature_5,feature_6]])

# Check if the input data has the correct number of features
if input_data.shape[1] != n_features:
    st.error(f"The model expects {n_features} features, but received {input_data.shape[1]} features.")
else:
    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write("### Predicted Output")
    st.write(prediction[0])
