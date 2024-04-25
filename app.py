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

if uploaded_file:
    # Process and display the Excel file
    df = st.session_state.linear_regression.preprocess_excel(uploaded_file)
    st.write("DataFrame preview (with background color codes as values):")
    st.dataframe(df)

    long_df = st.session_state.linear_regression.process_excel(uploaded_file)
    st.write("DataFrame preprocessed preview:")
    st.dataframe(long_df)

    # Display DataFrame information using StringIO buffer
    buffer = io.StringIO()
    long_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text("DataFrame Information:")
    st.code(s)

    # Button to train the model that uses session state
    if st.button('Train Model'):
        model_trained = st.session_state.linear_regression.fit(uploaded_file)

        if model_trained:
            st.success('The model has been successfully trained!')
            st.write(f'Mean Squared Error: {st.session_state.linear_regression.mse:.2f}')
            st.write(f'Accuracy Percentage: {st.session_state.linear_regression.accuracy_percentage * 100:.2f}%')

# Prediction form
with st.form(key='predict_form'):
    st.write("Enter details for color value prediction:")
    input_name = st.text_input('NAME (e.g. "GJ1")')
    input_date = st.date_input('Date')

    submit_button = st.form_submit_button(label='Predict Color Value')

    if submit_button and input_name and input_date:
        input_datetime = datetime.combine(input_date, datetime.min.time())

        # Predict data
        data = {
            'NAME': [input_name],
            'date': [input_datetime]
        }
        df = pd.DataFrame(data)

        # Display DataFrame and information
        st.write("DataFrame to predict with:")
        st.dataframe(df)

        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text("DataFrame Information:")
        st.code(s)

        # Make prediction using session state
        predicted_color_value = st.session_state.linear_regression.predict(df)
        st.write(f'Predicted Color Value: {predicted_color_value}')

        st.color_picker('Pick a color', value="#" + predicted_color_value, key='color_picker')
