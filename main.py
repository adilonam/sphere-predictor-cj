import streamlit as st
import pandas as pd
import io

from models.utils import process_excel



# Streamlit app
st.title('Cell Values to Background Color Converter')

# File uploader widget
uploaded_file = st.file_uploader("Upload your Excel file (.xlsx)", type=['xlsx'])

if uploaded_file:
    # Process the Excel file to change the cell values to their background color hex codes
    df = process_excel(uploaded_file)

    # Let the user download the updated file
    st.download_button(
        label='Download updated Excel file',
        data=uploaded_file,
        file_name='updated_excel_file.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Once the file_data is ready, load it into a DataFrame to display


    # Display the DataFrame
    st.write("DataFrame preview (with background color codes as values):")
    st.dataframe(df)

    # Display DataFrame information
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    
    st.text("DataFrame Information:")
    st.code(s)
