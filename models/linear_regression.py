import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from .abstract_model import AbstractModel




class LinReg(AbstractModel):


    def fit(self ,  X_train, X_test, y_train, y_test):
        # Splitting the dataset into the Training set and Test set
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)
        # get accuracy
        predicted_colors = self.model.predict(X_test)

        self.set_metrics(predicted_colors , y_test )
        return True
    
    
    



    




