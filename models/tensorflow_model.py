import json
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from .abstract_model import AbstractModel
from sklearn.metrics import mean_squared_error  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np



class TensorFlowModel(AbstractModel):
    epochs = 3

    def __init__(self) -> None:
        self.encoder = OneHotEncoder()
        super().__init__()



    
 

    def reshape_input(self , X):
        return X.reshape((X.shape[0], X.shape[1], 1))



    
    def fit(self ,  X, y):
        
        # No need for one-hot encoding in binary classification:
        # Remove the encoder fitting line

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        X = self.reshape_input(X)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        # No need to reshape X_train since we are assuming
        # it has already been reshaped for LSTM layers appropriately

        # Build the model
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        # Output layer for binary classification
        self.model.add(Dense(1, activation='sigmoid'))

        # Compile the model with binary crossentropy loss function and an optimizer
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        self.model.fit(
            X, 
            y, 
            epochs=self.epochs,  # Replace self.epochs with the actual number of epochs you want
            batch_size=32,       # The batch size
            verbose=1
        )

        predictions = self.model.predict(X_test)


        self.set_metrics((predictions > 0.5).astype("int32") , y_test)
        return True




    def predict(self, long_df):
        X = long_df[self.features].values
        X = self.scaler.transform(X)
        X = self.reshape_input(X)
        
        predictions = self.model.predict(X)
        return (predictions > 0.5).astype("int32")
    


    




