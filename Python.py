import streamlit as st
import numpy as np
import tensorflow as tf
import joblib  # Import joblib to load the scaler

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Load the scaler object used for preprocessing
scaler = joblib.load('scaler.joblib')  # Ensure this is the correct path to your saved scaler
scaler = joblib.load('D:/AI-learning-resources/Workshop/pythonProject/scaler.joblib')  # Replace 'path/to/' with the actual path to your scaler file

# Function to preprocess user input to match model expectations
def preprocess_input(user_input):
    # Assuming user_input is a numpy array with the correct shape
    user_input_scaled = scaler.transform(user_input)
    user_input_scaled = user_input_scaled.astype('float32')  # Cast to float32
    return user_input_scaled

# Function to make predictions with the TFLite model
def make_prediction(interpreter, input_data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Streamlit UI
st.title('Cancer Detection Tool')
st.write('Please enter the following patient details:')

# List of feature names from the Breast Cancer Wisconsin (Diagnostic) dataset
feature_names = [
    'Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area',
    'Mean Smoothness', 'Mean Compactness', 'Mean Concavity',
    'Mean Concave Points', 'Mean Symmetry', 'Mean Fractal Dimension',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE',
    'Smoothness SE', 'Compactness SE', 'Concavity SE',
    'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
    'Worst Radius', 'Worst Texture', 'Worst Perimeter', 'Worst Area',
    'Worst Smoothness', 'Worst Compactness', 'Worst Concavity',
    'Worst Concave Points', 'Worst Symmetry', 'Worst Fractal Dimension'
]

# Create input boxes for user input using real feature names
user_input = np.array([st.number_input(feature_name, step=0.01, format="%.2f") for feature_name in feature_names]).reshape(1, -1)


# Button to make predictions
if st.button('Predict'):
    # Preprocess the input to match the model's training data
    input_data = preprocess_input(user_input)

    # Make prediction
    prediction = make_prediction(interpreter, input_data)

    # Output the result
    st.write('Cancer Detection Result:')
    st.write('Probability of cancer:', prediction[0][0])  # Display the raw probability
    st.write('Predicted label:', prediction[0][0] > 0.5)  # Assuming a binary classification with a threshold of 0.5
