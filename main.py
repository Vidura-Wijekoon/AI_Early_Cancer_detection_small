import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os  # Import the os module
# Function to generate synthetic data
def create_data(n_samples=1000):
    np.random.seed(0)
    # Age and maintenance status
    age = np.random.randint(1, 30, size=n_samples)  # Age of the equipment in years
    maintenance = np.random.choice(['poor', 'average', 'good'], size=n_samples)

    # Overloading conditions
    capacity = np.random.uniform(70, 150, n_samples)  # Capacity of the system
    demand = np.random.uniform(50, 200, n_samples)  # Demand on the system

    # Weather conditions
    weather = np.random.choice(['clear', 'stormy', 'extreme'], size=n_samples)

    # Cybersecurity threat level
    cyber_threat = np.random.choice(['low', 'moderate', 'high'], size=n_samples)

    # Physical security threat level
    physical_threat = np.random.choice(['low', 'moderate', 'high'], size=n_samples)

    # Simplified rule for failure
    failure = ((age > 20) | (maintenance == 'poor') | (demand > capacity) |
               (weather == 'extreme') | (cyber_threat == 'high') | (physical_threat == 'high'))

    return pd.DataFrame({
        'age': age,
        'maintenance': maintenance,
        'capacity': capacity,
        'demand': demand,
        'weather': weather,
        'cyber_threat': cyber_threat,
        'physical_threat': physical_threat,
        'failure': failure
    })


# Train model function
def train_model(data):
    X = data.drop('failure', axis=1)
    y = data['failure']

    # Encoding categorical data
    categorical_features = ['maintenance', 'weather', 'cyber_threat', 'physical_threat']
    X = pd.get_dummies(X, columns=categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Using RandomForest for better accuracy with complex data
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Display accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Model accuracy: {accuracy:.2f}')

    # Return the model and the feature names
    return model, X_train.columns


# Main Streamlit app
def main():
    st.title("Advanced Power Grid Stability Predictor")

    # Add a description
    st.write(
        """
        This application predicts the stability of a power grid using various parameters 
        such as equipment age, maintenance condition, system capacity, demand, and environmental factors. 
        Adjust the input parameters and hit 'Predict' to see the results.
        """
    )

    # Display an image
    st.image("D:/AI-learning-resources/Workshop/pythonProject/venv/Scripts/powergrid.jpg", use_column_width=True)
    # Specify the relative path to your image
    image_path = "D:/AI-learning-resources/Workshop/pythonProject/venv/Scripts/powergrid.jpg"  # ensure this file is in your project directory



    # Check if the image file exists and display it; otherwise, show an error.
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.error(f"Failed to find image at: {image_path}")


    # User input
    st.sidebar.header('User Input Parameters')
    age = st.sidebar.slider('Age of equipment (years)', 1, 30, 15)
    maintenance = st.sidebar.selectbox('Maintenance condition', ['poor', 'average', 'good'])
    capacity = st.sidebar.slider('System capacity (MW)', 70, 150, 100)
    demand = st.sidebar.slider('Demand on system (MW)', 50, 200, 100)
    weather = st.sidebar.selectbox('Weather conditions', ['clear', 'stormy', 'extreme'])
    cyber_threat = st.sidebar.selectbox('Cybersecurity threat level', ['low', 'moderate', 'high'])
    physical_threat = st.sidebar.selectbox('Physical security threat level', ['low', 'moderate', 'high'])

    data = create_data()
    model, feature_names = train_model(data)  # Receive feature names here

    # Visualize feature importances
    plot_feature_importances(model, feature_names)  # Pass the correct feature names

    if st.sidebar.button('Predict'):
        input_data = pd.DataFrame({
            'age': [age],
            'maintenance': [maintenance],
            'capacity': [capacity],
            'demand': [demand],
            'weather': [weather],
            'cyber_threat': [cyber_threat],
            'physical_threat': [physical_threat]
        })

        # Encoding categorical data for prediction
        input_data = pd.get_dummies(input_data, columns=['maintenance', 'weather', 'cyber_threat', 'physical_threat'])

        # Ensure all columns in the input are in the same order as the training data
        missing_cols = set(feature_names) - set(input_data.columns)
        for c in missing_cols:
            input_data[c] = 0
        input_data = input_data[feature_names]

        prediction = model.predict(input_data)[0]
        st.write('Grid Failure Predicted:' if prediction else 'No Grid Failure Predicted')


def plot_feature_importances(model, features):
    # Retrieve feature importances from the model and pair them with feature names
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Create a bar chart
    plt.figure(figsize=(12, 8))  # Increase the vertical size of the figure
    plt.title("Feature Importances")
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [features[i] for i in indices], rotation=45, ha='right')

    plt.subplots_adjust(bottom=0.3)  # Adjust the bottom margin to make room for the rotated x-axis labels

    # Show the plot in Streamlit
    st.pyplot(plt)


if __name__ == "__main__":
    main()
