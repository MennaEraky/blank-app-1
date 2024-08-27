
import streamlit as st
import pickle
import numpy as np

# load pkl file
with open('tree_clf (3).pkl', 'rb') as file:
    model = pickle.load(file)


# Check the number of features the model expects
n_features = model.n_features_in_

# Streamlit App
st.title("Linear Regression Prediction")

# Input features
st.sidebar.header("Input Features")
Pregnancies = st.sidebar.number_input("Pregnancies ", value=10.0)
Glucose = st.sidebar.number_input("Glucose", value=20.0)
Insulin = st.sidebar.number_input("Insulin", value=30.0)
BMI = st.sidebar.number_input("BMI", value=30.0)
DiabetesPedigreeFunction = st.sidebar.number_input("DiabetesPedigreeFunction", value=30.0)
Age = st.sidebar.number_input("Age ", value=30.0)

# Prepare the input data
input_data = np.array([[Pregnancies, Glucose, Insulin,BMI,DiabetesPedigreeFunction,Age]])

# Check if the input data has the correct number of features
if input_data.shape[1] != n_features:
    st.error(f"The model expects {n_features} features, but received {input_data.shape[1]} features.")
else:
    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write("### Predicted Output")
    st.write(prediction[0])
