from datetime import datetime
import streamlit as st
import pandas as pd
# from models.linear_regression import LinReg
# from models.tensorflow_model import TensorFlowModel
from models.lib_cj import CjModel
import numpy as np
# Streamlit app code
st.title('Predict Sphere CJ')






if 'model' not in st.session_state:
    st.session_state.model = CjModel()


cj_number = st.number_input("Enter the cj number to predict:", min_value=0, max_value=100, step=1)

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=['xlsx'])
    # Function to apply a background color to cells in a DataFrame





if uploaded_file and cj_number:
    # Process and display the Excel file
    long_df  = st.session_state.model.process_excel(uploaded_file)
    st.write("DataFrame preview (with background color codes as values):")
    st.dataframe(long_df)
    st.session_state.model.load()
    X, y, last_X = st.session_state.model.make_train_data(long_df, int(cj_number))
    



    # Display the DataFrame with `next_color` column colored accordingly
    if st.button('Show Predictions'):
        prediction_label , prediction = st.session_state.model.predict_last(last_X)
        if prediction_label is not None and prediction is not None:
            st.write(f"### Predicted DataFrame with Next Color Codes for CJ{cj_number}:")
            
            
            # Create a DataFrame for better visualization
            color_labels = ['Red', 'Green', 'Yellow', 'Blue', 'Orange', 'Purple', 'Maroon']
            df = pd.DataFrame(prediction, columns=color_labels)
            st.dataframe(df)
            
            # Display the conversion of prediction labels
            st.write("**Convert this Code to Winner Color Using:**")
            color_mapping = {1: 'Red', 2: 'Green', 3: 'Yellow', 4: 'Blue', 5: 'Orange', 6: 'Purple', 7: 'Maroon'}
            winner_color = color_mapping[prediction_label[0,0]]
            st.markdown(f"<p style='color:green;'>Prediction Label (Converted): {winner_color}</p>", unsafe_allow_html=True)
        else:
            st.error('No predictions to display. Ensure model is trained and data is available.')




