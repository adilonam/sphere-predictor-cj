import pandas as pd
import openpyxl
import io
import tempfile
from sklearn.metrics import mean_squared_error  
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error  
from sklearn.model_selection import train_test_split
import numpy as np 

class AbstractModel:
    
    color_mapping = {}
    features = ['date', 'name_code' , 'value', 'color_code']
    target = 'next_color_code'
    X_predict = None
    is_preferred_color = False
    preferred_color = ["D5A6BD" ,"FF9900" ] 
    preferred_color_code = []

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
        



    def color_change(self , x):
        if x in self.preferred_color:
            return x 
        else:
            return "FFFFFF"

    def process_data(self ,df):
        if self.color_mapping:
            raise Exception("data process has already processed")
        long_df = df
        first_col_header = long_df.columns[0]
        long_df.columns = [first_col_header] + [i for i in range(1, len(long_df.columns))]
        long_df['name_code'] = long_df.index 
        # Processing the DataFrame 'data' to have "date", "name", "color_value" columns
        long_df = pd.melt(df, id_vars=['NAME' , 'name_code'], var_name='date', value_name='color_and_value')
        # Convert dates and name to a numerical value, 
        long_df['date'] = long_df['date'].astype(int)  
        long_df['name_code'] = long_df['name_code'].astype(int)
        long_df[['value', 'color']] = long_df['color_and_value'].str.split(' \| ', expand=True)
        # check working with three color
        if self.is_preferred_color:
            long_df['color'] = long_df['color'].map(lambda x : self.color_change(x))

        long_df['value'] =  long_df['value'].astype(float)
        # Assume long_df is your pre-loaded pandas DataFrame.
        codes, uniques = pd.factorize(long_df['color'])
        # Add 1 to codes to start numbering from 1 instead of 0
        long_df['color_code'] = codes 
        # Create a color mapping from the factorize operation, starting with 1
        for i in range(len(uniques)):
            self.color_mapping [i]  = uniques[i] 
            if uniques[i] in self.preferred_color:
                self.preferred_color_code.append(i)
        long_df['next_color_code'] = long_df.groupby('name_code')['color_code'].shift(-1)

        return long_df 


    def process_excel(self ,uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)
    

    def set_metrics(self, predictions , y_test):
       
        predicted_color_code =  self.decode(predictions)
        true_color_code = self.decode(y_test)

        self.mse = mean_squared_error(true_color_code, predicted_color_code)

        self.mse = mean_squared_error(true_color_code, predicted_color_code)
        # Calculate the number of correct predictions
        correct_predictions = sum(true_color_code == predicted_color_code)
        # Calculate the total number of predictions
        total_predictions = len(predicted_color_code)
        # Calculate the percentage of correct predictions
        self.accuracy = (correct_predictions / total_predictions)

        # get prefered accuracy 
        preferred_correct_predictions = 0

        # Check each pair of true and predicted colors
        for t, p in zip(true_color_code, predicted_color_code):
            # If the predicted color is not in the allowed values, count it (regardless of it being correct)
            # OR
            # If the predicted color is an allowed value and matches the true color, count it
            if p not in self.preferred_color_code or (p in self.preferred_color_code and t == p):
                preferred_correct_predictions += 1

        self.preferred_accuracy = (preferred_correct_predictions / total_predictions)

    
    
    def decode(self , predictions):
        predicted_labels = predictions.argmax(axis=1)
        one_hot_predictions = np.zeros((predicted_labels.shape[0], len(self.encoder.categories_[0])))

        # Set the predicted labels to 1
        for i, label in enumerate(predicted_labels):
            one_hot_predictions[i, label] = 1

        # Step 2: Use the encoder to inverse transform the one-hot encoded predictions
        y_test_pred = self.encoder.inverse_transform(one_hot_predictions)
        y_test_pred = y_test_pred.reshape(y_test_pred.shape[0]).astype(int)
        return y_test_pred
    
    def color_mapper(self , x):
        if x in self.color_mapping:
            return self.color_mapping[x]
        else:
            last_key = sorted(self.color_mapping.keys())[-1]
            return self.color_mapping[last_key]

    def predict(self):
        if  self.X_predict is None:
            raise Exception('Must be trained')
        predictions = self.model.predict(self.X_predict)
        predicted_colors =  self.decode(predictions)

        predicted_df = self.last_df
        predicted_df['next_color_code'] =  predicted_colors
        predicted_df['next_color_code'] = predicted_df['next_color_code'].round().astype(int)
        predicted_df['next_color'] = predicted_df['next_color_code'].map( lambda x: self.color_mapper(x) )
        return predicted_df
    

    def train_test_split(self, long_df):
        # Prepare the dataset for Linear Regression
        # Updated to include 'name_as_number' as an additional feature
        long_df = long_df.dropna(axis=0)
        X = long_df[self.features].values # Features
        y = long_df[self.target].values  # Target

        # Convert labels to one-hot encoding
        self.encoder = OneHotEncoder()
        y = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()

        self.X_predict = X[-self.row_count:]
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return  X_train, X_test, y_train, y_test
    
    