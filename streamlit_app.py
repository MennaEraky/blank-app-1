
import streamlit as st
import pickle
import numpy as np

# load pkl file
with open('tree_clf (1).pkl', 'rb') as file:
    model = pickle.load(file)


# Check the number of features the model expects
n_features = model.n_features_in_

# Streamlit App
st.title("Linear Regression Prediction")

# Input features
st.sidebar.header("Input Features")
Glucose = st.sidebar.number_input("Feature 1", value=1.0)
Glucose = st.sidebar.number_input("Feature 2", value=2.0)
Insulin = st.sidebar.number_input("Feature 3", value=3.0)
BMI = st.sidebar.number_input("Feature 4", value=3.0)
DiabetesPedigreeFunction = st.sidebar.number_input("Feature 5", value=3.0)
Age = st.sidebar.number_input("Feature 6", value=3.0)

# Prepare the input data
input_data = np.array([[Glucose, Glucose, Insulin,BMI,DiabetesPedigreeFunction,Age]])

# Check if the input data has the correct number of features
if input_data.shape[1] != n_features:
    st.error(f"The model expects {n_features} features, but received {input_data.shape[1]} features.")
else:
    # Make prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write("### Predicted Output")
    st.write(prediction[0])
