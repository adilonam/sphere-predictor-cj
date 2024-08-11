from datetime import datetime
import io
import streamlit as st
import pandas as pd

# from models.linear_regression import LinReg
# from models.tensorflow_model import TensorFlowModel
from models.tensorflow_model import TensorFlowModel

# Streamlit app code
st.title('Predict Sphere CJ')








if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False



if 'model' not in st.session_state:
    st.session_state.model = TensorFlowModel()
    st.session_state.model.color_mapping = {}



# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=['xlsx'])
    # Function to apply a background color to cells in a DataFrame
def colorize(val):
    # Assuming 'val' is between 0 and 1 for the purpose of creating a gradient 
    # If your values have a different range, you need to normalize them first
    green_value = int(val * 255)
    color = f'00FF00'  # Adjusted green channel based on 'val' proximity to 0
    style = f'background-color: #{color}'
    return style





if uploaded_file:
    # Process and display the Excel file
    df = st.session_state.model.preprocess_excel(uploaded_file)
    st.write("DataFrame preview (with background color codes as values):")
    st.dataframe(df)
    long_df  = st.session_state.model.process_excel(uploaded_file)
    X, y  = st.session_state.model.train_test_split(long_df)

    load_button_clicked = st.button('Load Model')
    if load_button_clicked:
        try:
            st.session_state.model.load()
            st.success(f'Model and scaler have been loaded. Last saved on: {st.session_state.model.last_save_time}')
        except FileNotFoundError as e:
            st.error('Model and scaler could not be found. Please train a model first.')
    # Button to train the model that uses session state
    if st.button('Train Model'):

        # Display a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()  # Placeholder to display text below progress bar
        
        # Some arbitrary divisions of progress (you will adjust these based on the actual progress of your steps)
        progress_bar.progress(10)
        progress_text.text("Processing Excel file...")
        
        
        progress_bar.progress(30)
        progress_text.text("Splitting data for training and testing...")
        

        progress_bar.progress(50)
        progress_text.text("Training the model...")
        model_trained = st.session_state.model.fit(X, y)
        
        if model_trained:
            st.session_state.model_trained = True
            progress_bar.progress(100)
            progress_text.text("Model training completed successfully!")
        
        else:
            progress_text.text("Model training failed.")
            st.error('Model training was unsuccessful. Please check your data and try again.')

        # Clean up the temporary UI components
        progress_bar.empty()
        progress_text.empty()
    if st.session_state.model_trained:
        st.success('The model has been successfully trained!')
        color_count = len(st.session_state.model.color_mapping)
        st.write(f'Mean Squared Error: {st.session_state.model.mse:.2f}')
        save_button_clicked = st.button('Save Model')
        if save_button_clicked:
            st.session_state.model.save()
            st.success(f'Model and scaler have been saved. Last saved on: {st.session_state.model.last_save_time}')

    # Display the DataFrame with `next_color` column colored accordingly
    if st.button('Show Predictions'):
        predicted_df = st.session_state.model.predict_last()
        if not predicted_df.empty:
            st.write("Predicted DataFrame with next_color codes:")
            st.write('Prediction close of 1 mean: Red, Green, Yellow')
            st.write('Prediction close of 0 mean: Orange, Purple, Blue')
            # Apply the coloring function to the 'next_color' column
            predicted_df = predicted_df.sort_values(by='next_color_code', ascending=False)
            st.dataframe(predicted_df)
        else:
            st.error('No predictions to display. Ensure model is trained and data is available.')




