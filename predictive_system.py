# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:42:05 2024

@author: KALYANAM ABHINAV
"""

import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('./trained_model.sav', 'rb'))

# Prediction function
def prediction(input_data):
    # Convert input data to a NumPy array and reshape for model input
    input_data = np.array(input_data, dtype=float).reshape(1, -1)
    
    # Predict using the loaded model
    prediction_result = loaded_model.predict(input_data)
    
    # Return the predicted class
    return prediction_result[0]

def main():
    # Title of the app
    st.title('Decision Tree Prediction Model')
    
    # Input fields for user
    sepal_length = st.text_input('Sepal Length:')
    sepal_width = st.text_input('Sepal Width:')
    petal_length = st.text_input('Petal Length:')
    petal_width = st.text_input('Petal Width:')
    
    # Code for prediction
    diagnosis = ''
    
    # Prediction button
    if st.button('Get Prediction Result'):
        try:
            # Collect input data
            input_data = [sepal_length, sepal_width, petal_length, petal_width]
            diagnosis = prediction(input_data)
            st.success(f'Predicted Class: {diagnosis}')
        except Exception as e:
            st.error(f'Error in prediction: {e}')
    
if __name__ == '__main__':
    main()
