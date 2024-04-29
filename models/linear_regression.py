import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from .abstract_model import AbstractModel




class LinReg(AbstractModel):


    def fit(self ,  X, y):
        # Splitting the dataset into the Training set and Test set

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        # get accuracy
        predicted_colors = self.model.predict(X_test)
        predicted_colors = [round(x) for x in predicted_colors]
        self.set_metrics(predicted_colors , y_test )
        return True
    
    def predict(self, long_df):
        X = long_df[self.features].values
        predictions = self.model.predict(X)
        return predictions
    



    




