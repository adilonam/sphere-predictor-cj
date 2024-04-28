import pandas as pd
import openpyxl
import io
import tempfile
from sklearn.model_selection import train_test_split



class AbstractModel:
    
    color_mapping = {}
    features = ['date', 'name_code' , 'value', 'color_code']
    target = 'next_color_code'
    X_predict = None

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
            self.row_count =  df.shape[0]
            
            return df
        


    def process_data(self ,df):
        if self.color_mapping:
            raise Exception("data process has already processed")
        long_df = df
        first_col_header = long_df.columns[0]
        long_df.columns = [first_col_header] + [i for i in range(1, len(long_df.columns))]
        long_df['name_code'] = long_df.index +1
        # Processing the DataFrame 'data' to have "date", "name", "color_value" columns
        long_df = pd.melt(df, id_vars=['NAME' , 'name_code'], var_name='date', value_name='color_and_value')
        # Convert dates and name to a numerical value, 
        long_df['date'] = long_df['date'].astype(int)  
        long_df['name_code'] = long_df['name_code'].astype(int)
        long_df[['value', 'color']] = long_df['color_and_value'].str.split(' \| ', expand=True)
        long_df['value'] =  long_df['value'].astype(float)
        # Assume long_df is your pre-loaded pandas DataFrame.
        codes, uniques = pd.factorize(long_df['color'])
        # Add 1 to codes to start numbering from 1 instead of 0
        long_df['color_code'] = codes + 1
        # Create a color mapping from the factorize operation, starting with 1
        self.color_mapping = {i + 1: uniques[i] for i in range(len(uniques))}
        long_df['next_color_code'] = long_df.groupby('name_code')['color_code'].shift(-1)
        return long_df 


    def process_excel(self ,uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)
    
    def train_test_split(self, long_df):
        if self.X_predict:
            raise Exception("Split has already processed")
        # Prepare the dataset for Linear Regression
        # Updated to include 'name_as_number' as an additional feature
        long_df = long_df.dropna(axis=0)
        X = long_df[self.features].values # Features
        y = long_df[self.target].values  # Target
        self.X_predict = X[-self.row_count:]
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return  X_train, X_test, y_train, y_test