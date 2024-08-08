import pandas as pd
import openpyxl
import io
import tempfile
from sklearn.metrics import mean_squared_error  
import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error  
import numpy as np 
from sklearn.metrics import accuracy_score

class AbstractModel:
    
    color_mapping = {}
    features = ["date_code"  ,'name_code' , 'day_of_year', 'day_of_month', 'day_of_week' ,"color_code" , 'value'  ] #  'day_of_year', 'day_of_month', 'day_of_week'
    target = 'next_color_binary'
    last_long_df = None
    preferred_color = ['00FF00' , 'FFFF00'  , 'FF0000']      
    # ['#FF0000' red , 'FF9900' orange , 'D5A6BD' purple, '00FF00' green  , 'FFFF00' yellow, '00FFFF' blue]
    last_df = None
    is_sheet2 = True
    color_group_dict = {
        "FF0000": 3, 
        "00FF00": 1, 
        "FFFF00": 1, 
        "00FFFF": 2, 
        "FF9900": 2, 
        "D5A6BD": 2,
    }

    def preprocess_excel(self, uploaded_file):
        # Load the workbook
        wb = openpyxl.load_workbook(filename=uploaded_file)
        
        # Process Sheet1
        ws_sheet1 = wb['Sheet1']
        for row in ws_sheet1.iter_rows(min_row=2, min_col=2):
            for cell in row:
                color_hex = cell.fill.start_color.index[2:] if cell.fill.start_color.index else 'None'
                cell.value = f"{cell.value} | {color_hex}"
        
        # Create a DataFrame from Sheet1 with updates
        with tempfile.NamedTemporaryFile() as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            df_sheet1 = pd.read_excel(tmp, engine='openpyxl', sheet_name='Sheet1')
        
        # Create a DataFrame from Sheet2 without changes
        if self.is_sheet2:
            df_sheet2 = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name='Sheet2')
            print((df_sheet1.shape , df_sheet2.shape))
            assert (df_sheet1.shape == df_sheet2.shape), "Shape in df_sheet1 and df_sheet2 do not match."
            for i in range(1, df_sheet1.shape[1]):
                df_sheet1.iloc[:, i] = df_sheet1.iloc[:, i].astype(str) + ' | ' + df_sheet2.iloc[:, i].astype(str)

        return df_sheet1
        



    def color_change(self , x):
        if x in self.preferred_color:
            return 1
        else:
            return 0
    
 
       

    def process_data(self ,df):
        if self.color_mapping:
            raise Exception("data process has already processed")
        
        # Get the first and last columns as Series
        first_column = df.iloc[:, 0]
        last_column = df.iloc[:, -1]

        # Concatenate the Series into a new DataFrame
        self.last_df = pd.concat([first_column, last_column], axis=1)
        self.row_count =  df.shape[0]

        # Processing the DataFrame 'data' to have "date", "name", "color_value" columns
        long_df = pd.melt(df, id_vars=['NAME'], var_name='date', value_name='color_and_value')

        # Convert dates and name to a numerical value, 
        long_df['name_code'] = long_df['NAME'].str.extract(r'(\d+)').astype(int)

        long_df['date'] = pd.to_datetime(long_df['date'])
        # Get day of the month
        long_df['day_of_month'] = long_df['date'].dt.day

        # Get day of the week (Monday=0, Sunday=6)
        long_df['day_of_week'] = long_df['date'].dt.dayofweek


        codes, uniques = pd.factorize(long_df['date'])
        long_df['date_code'] = codes

        long_df['day_of_year'] = long_df['date'].dt.dayofyear.astype(int)

        long_df['date'] = long_df['date'].astype(int) 

        if self.is_sheet2:
            long_df[['value', 'color' , 'extra']] = long_df['color_and_value'].str.split(' \| ', expand=True)
            long_df['extra'] =  long_df['extra'].astype(float)
            if "extra" not in self.features:
                self.features.append("extra")
        else:
            long_df[['value', 'color']] = long_df['color_and_value'].str.split(' \| ', expand=True)
        long_df['value'] =  long_df['value'].astype(float)
        

        codes, uniques = pd.factorize(long_df['color'])
        self.current_colors = uniques
        # Add 1 to codes to start numbering from 1 instead of 0
        long_df['color_code'] = codes 
        long_df['next_color_code'] = long_df.groupby('name_code')['color_code'].shift(-1)
        long_df['previous_color_code'] = long_df.groupby('name_code')['color_code'].shift(1)

        long_df['color_binary'] = long_df['color'].map(lambda x : self.color_change(x)).astype(int)
        
        long_df['next_color_binary'] = long_df.groupby('name_code')['color_binary'].shift(-1)
        
        long_df['color_group'] = long_df['color'].map(self.color_group_dict).astype(int)
        
        long_df['next_color_group'] = long_df.groupby('name_code')['color_binary'].shift(-1)
        

        
        return long_df 


    def process_excel(self ,uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)
    

    def set_metrics(self, predictions , y_test):
       
        self.mse = mean_squared_error(y_test, predictions)
        self.accuracy  = accuracy_score(y_test, predictions)
        correct = 0
        self.predictions_count = 0
        self.y_test_count = 0
        for t, p in zip(y_test, predictions):
            # If the predicted color is not in the allowed values, count it (regardless of it being correct)
            # OR
            # If the predicted color is an allowed value and matches the true color, count it
            if  t == p and t == 1:
                correct += 1
            if p  == 1:
                self.predictions_count += 1
            if t == 1:
                self.y_test_count +=1 

        self.preferred_accuracy = correct / self.predictions_count if  self.predictions_count != 0 else 0 


       
    
    
    
    def color_mapper(self , x):
        if x in self.color_mapping:
            return self.color_mapping[x]
        else:
            last_key = sorted(self.color_mapping.keys())[-1]
            return self.color_mapping[last_key]

    def predict_last(self):
        if  self.last_long_df is None:
            raise Exception('Must be trained')
        predictions = self.predict(self.last_long_df)
        predicted_df = self.last_df
        predicted_df['color_and_value'] =  self.last_long_df['color_and_value'].values
        predicted_df['next_color_code'] =  predictions
        return predicted_df
    

    def train_test_split(self, long_df):
        # Prepare the dataset for Linear Regression
        # Updated to include 'name_as_number' as an additional feature
        self.last_long_df = long_df[-self.row_count:]
        long_df = long_df.dropna(axis=0)
        X = long_df[self.features].values # Features
        y = long_df[self.target].values  # Target

        
        return  X, y
    def predict(self , X):
        raise Exception('Not emplemented')
    
    