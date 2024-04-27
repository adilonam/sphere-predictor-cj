import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .abstract_model import AbstractModel
from sklearn.metrics import mean_squared_error  



class LinReg(AbstractModel):


    def train_test_split(self, df):
        # Prepare the dataset for Linear Regression
        # Updated to include 'name_as_number' as an additional feature
        X = df[['day_of_year', 'name_as_number' , 'value', 'color_code']].values # Features
        y = df['next_color_code'].values  # Target

        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        return  X_train, X_test, y_train, y_test

    def fit(self , uploaded_file):
        long_df = self.process_excel(uploaded_file)
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = self.train_test_split(long_df)
        self.regressor = LinearRegression()
        self.regressor.fit(X_train, y_train)
        # get accuracy
        predicted_colors = self.regressor.predict(X_test)
        predicted_colors =  [round(x) for x in predicted_colors]
        self.mse = mean_squared_error(y_test, predicted_colors)
        # Calculate the number of correct predictions
        correct_predictions = sum(y_test == predicted_colors)
        # Calculate the total number of predictions
        total_predictions = len(predicted_colors)
        # Calculate the percentage of correct predictions
        self.accuracy_percentage = (correct_predictions / total_predictions)
        return True
    
    def predict(self , long_df):
        long_df['day_of_year'] = long_df['date'].dt.dayofyear
        long_df['name_as_number'] = long_df['NAME'].str.extract('(\d+)').astype(int)
        long_df['value'] = long_df['value'].astype(int)

        X = long_df[['day_of_year', 'name_as_number', 'value']].values  # Features
        predicted_colors = self.regressor.predict(X)
        color_code = round(predicted_colors[0]) if round(predicted_colors[0]) < len(self.color_mapping) else len(self.color_mapping) -1
        
        return self.color_mapping[color_code]



    




