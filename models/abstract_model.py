import pandas as pd
import openpyxl
import io
import tempfile




class AbstractModel:
    
    color_mapping = {}

    def preprocess_excel(self , uploaded_file):
        # Load the workbook
        wb = openpyxl.load_workbook(filename=uploaded_file)
        ws = wb.active

        # Iterate through the cells, skipping the first row and the first column
        for row in ws.iter_rows(min_row=2, min_col=2):
            for cell in row:
                # Check for background color's hex code
                color_hex = cell.fill.start_color.index[2:] if cell.fill.start_color.index else 'None'

                # Combine the value with the color hex code within the same cell
                cell.value = f"{cell.value} | {color_hex}"

        # Save the updated workbook to a temporary file and return it
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            df = pd.read_excel(io.BytesIO(tmp.read()))
            # Get the first and last columns as Series
            first_column = df.iloc[:, 0]
            last_column = df.iloc[:, -1]

            # Concatenate the Series into a new DataFrame
            self.last_df = pd.concat([first_column, last_column], axis=1)

            
            return df
        


    def process_data(self ,df):


        # Processing the DataFrame 'data' to have "date", "name", "color_value" columns
        long_df = pd.melt(df, id_vars=['NAME'], var_name='date', value_name='color_and_value')


        # Convert dates and name to a numerical value, 
        long_df['date'] = pd.to_datetime(long_df['date'])  
        long_df['day_of_year'] = long_df['date'].dt.dayofyear
        long_df['name_as_number'] = long_df['NAME'].str.extract('(\d+)').astype(int)
        long_df[['value', 'color']] = long_df['color_and_value'].str.split(' \| ', expand=True)
        long_df['value'] =  long_df['value'].astype(float)

        codes, uniques = pd.factorize(long_df['color'])
        long_df['color_code'] = codes

        # Create a color mapping from the factorize operation
        self.color_mapping = dict(enumerate(uniques))

        long_df['next_color_code'] = long_df['color_code'].shift(-1)
        long_df = long_df.dropna(axis=0)

        return long_df 


    def process_excel(self ,uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)