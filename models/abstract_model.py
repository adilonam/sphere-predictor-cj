import pandas as pd
import openpyxl
import io
import tempfile




class AbstractModel:
    

    
    color_code_to_hex_mapping = {}

    def preprocess_excel(self , uploaded_file):
        # Load the workbook
        wb = openpyxl.load_workbook(filename=uploaded_file)
        ws = wb.active

        # Iterate through the cells, skipping the first row and the first column
        for row in ws.iter_rows(min_row=2, min_col=2):
            for cell in row:
                # Set the cell value to the background color's hex code
                cell.value = cell.fill.start_color.index[2:] if cell.fill.start_color.index else None

        # Save the updated workbook to a temporary file and return it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            return pd.read_excel(io.BytesIO(tmp.read()))
        
    

    def process_data(self,df ):

        # Assuming `df` is the DataFrame with the color data
        color_series = df.drop('NAME', axis=1).stack()  # Drop the 'NAME' column and stack to create a single series
        color_codes, unique_colors = pd.factorize(color_series)  # Factorize the entire series to get unique codes

        # Map the color codes back into the original DataFrame shape
        coded_df = pd.DataFrame(color_codes.reshape(df.shape[0], -1), columns=df.columns[1:])
        coded_df.insert(0, 'NAME', df['NAME'])  # Insert the 'NAME' column back into the DataFrame

        self.color_code_to_hex_mapping = dict(enumerate(unique_colors))

        

        # Processing the DataFrame 'data' to have "date", "name", "color_value" columns
        long_df = pd.melt(coded_df, id_vars=['NAME'], var_name='date', value_name='color_value')


        # Convert dates and name to a numerical value, 
        long_df['date'] = pd.to_datetime(long_df['date'])  
        long_df['day_of_year'] = long_df['date'].dt.dayofyear
        long_df['name_as_number'] = long_df['NAME'].str.extract('(\d+)').astype(int)



        return long_df 
    

    def process_excel(self , uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)