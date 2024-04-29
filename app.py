from datetime import datetime
import io
import streamlit as st
import pandas as pd

# from models.linear_regression import LinReg
# from models.tensorflow_model import TensorFlowModel
from models.random_forest import RandomForest

# Streamlit app code
st.title('Predict Sphere')








if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False



if 'model' not in st.session_state:
    st.session_state.model = RandomForest()
    st.session_state.model.color_mapping = {}



# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=['xlsx'])
    # Function to apply a background color to cells in a DataFrame
def colorize(val):
    color = f'background-color: #{val}' if pd.notnull(val) else ''
    return color





if uploaded_file:
    # Process and display the Excel file
    df = st.session_state.model.preprocess_excel(uploaded_file)
    st.write("DataFrame preview (with background color codes as values):")
    st.dataframe(df)


    # Button to train the model that uses session state
    if st.button('Train Model'):

        # Display a progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()  # Placeholder to display text below progress bar
        
        # Some arbitrary divisions of progress (you will adjust these based on the actual progress of your steps)
        progress_bar.progress(10)
        progress_text.text("Processing Excel file...")
        long_df  = st.session_state.model.process_excel(uploaded_file)
        
        progress_bar.progress(30)
        progress_text.text("Splitting data for training and testing...")
        X, y  = st.session_state.model.train_test_split(long_df)

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
        st.write(f'Accuracy Percentage : {st.session_state.model.accuracy * 100:.2f}%')
        st.write(f'Accuracy to have a correct color from purple orange blue: {st.session_state.model.preferred_accuracy * 100:.2f}%')


    # Display the DataFrame with `next_color` column colored accordingly
    if st.button('Show Predictions'):
        predicted_df = st.session_state.model.predict_last()
        if not predicted_df.empty:
            st.write("Predicted DataFrame with next_color codes:")
            # Apply the coloring function to the 'next_color' column
            st.dataframe(predicted_df.style.map(colorize, subset=['next_color']))
        else:
            st.error('No predictions to display. Ensure model is trained and data is available.')




