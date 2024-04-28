from datetime import datetime
import io
import streamlit as st
import pandas as pd

# This assumes LinReg is defined properly in the models/linear_regression file
from models.linear_regression import LinReg

# Streamlit app code
st.title('Predict Sphere')

# Initialize session state for the linear regression model if it's not already set
if 'linear_regression' not in st.session_state:
    st.session_state.linear_regression = LinReg()

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=['xlsx'])
    # Function to apply a background color to cells in a DataFrame
def colorize(val):
    color = f'background-color: #{val}' if pd.notnull(val) else ''
    return color


if uploaded_file:
    # Process and display the Excel file
    df = st.session_state.linear_regression.preprocess_excel(uploaded_file)
    st.write("DataFrame preview (with background color codes as values):")
    st.dataframe(df)


    # Button to train the model that uses session state
    if st.button('Train Model & predict'):
        long_df  = st.session_state.linear_regression.process_excel(uploaded_file)
        model_trained = st.session_state.linear_regression.fit(long_df)
        if model_trained:
            st.success('The model has been successfully trained!')
            st.write(f'Mean Squared Error: {st.session_state.linear_regression.mse:.2f}')
            st.write(f'Accuracy Percentage: {st.session_state.linear_regression.accuracy_percentage * 100:.2f}%')
        predicted_df = st.session_state.linear_regression.predict()
        if not predicted_df.empty:
            st.write("Predicted DataFrame with next_color codes:")
            # Apply the coloring function to the 'next_color' column
            st.dataframe(predicted_df.style.applymap(colorize, subset=['next_color']))
        else:
            st.error('No predictions to display. Ensure model is trained and data is available.')

        




