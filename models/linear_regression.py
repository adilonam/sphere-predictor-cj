import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from .abstract_model import AbstractModel
from sklearn.metrics import mean_squared_error  



class LinReg(AbstractModel):

    def fit(self , long_df):
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
    
    def color_mapper(self , x):
        if x in self.color_mapping:
            return self.color_mapping[x]
        else:
            last_key = sorted(self.color_mapping.keys())[-1]
            return self.color_mapping[last_key]
    def predict(self):
        if  self.X_predict is None:
            raise Exception('Must be trained')
        predicted_colors = self.regressor.predict(self.X_predict)
        predicted_df = self.last_df
        predicted_df['next_color_code'] =  predicted_colors
        predicted_df['next_color_code'] = predicted_df['next_color_code'].round().astype(int)
        predicted_df['next_color'] = predicted_df['next_color_code'].map( lambda x: self.color_mapper(x) )
        return predicted_df



    




