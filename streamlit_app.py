import streamlit as st
import pickle
import numpy as np

# Load the model
model_path = 'logistic.pkl'
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Check the number of features the model expects
try:
    n_features = model.n_features_in_
except AttributeError:
    st.error("Model does not have 'n_features_in_' attribute. Check the model object.")
    n_features = None

# Streamlit App
st.title("Decision Tree Prediction")

# Input features
st.sidebar.header("Input Features")
Pregnancies = st.sidebar.number_input("Pregnancies", value=10.0)
Glucose = st.sidebar.number_input("Glucose", value=20.0)
Insulin = st.sidebar.number_input("Insulin", value=30.0)
BMI = st.sidebar.number_input("BMI", value=30.0)
DiabetesPedigreeFunction = st.sidebar.number_input("DiabetesPedigreeFunction", value=30.0)
Age = st.sidebar.number_input("Age", value=30.0)

# Prepare the input data
input_data = np.array([[Pregnancies, Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age]])

# Check if the input data has the correct number of features
if n_features is None or input_data.shape[1] != n_features:
    st.error(f"The model expects {n_features} features, but received {input_data.shape[1]} features.")
else:
    # Make prediction
    try:
        # Check if model supports the predict method
        if hasattr(model, 'predict'):
            prediction = model.predict(input_data)
            st.write("### Predicted Output")
            st.write(prediction[0])
        else:
            st.error("The model does not have a 'predict' method.")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
