from datetime import datetime
import json
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from .abstract_model import AbstractModel
from sklearn.metrics import mean_squared_error  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization ,Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import joblib




class TensorFlowModel(AbstractModel):
    epochs = 30
    prob = 0.61
    last_save_time = None

    def __init__(self) -> None:
        self.encoder = OneHotEncoder()
        super().__init__()

    def save(self, path = './.models'):
        # Save the TensorFlow model
        self.model.save(f'{path}/tensorflow/model.h5')
        
        # Save the scaler
        joblib.dump(self.scaler, f'{path}/tensorflow/scaler.pkl')

        # Get the current time as a string
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.last_save_time = current_time
        
        # Store the timestamp in a separate text file
        with open(f'{path}/tensorflow/time.txt', 'w') as f:
            f.write(current_time)

    def load(self, path ='./.models' ):
        # Load the TensorFlow model
        self.model = load_model(f'{path}/tensorflow/model.h5')
        
        # Load the scaler
        self.scaler = joblib.load( f'{path}/tensorflow/scaler.pkl')
        # For the plain text file
        with open(f'{path}/tensorflow/time.txt', 'r') as f:
            self.last_save_time = f.read()



    
 

    def reshape_input(self , X):
        return X.reshape((X.shape[0], X.shape[1], 1))



    
    def fit(self ,  X, y):
        
        # No need for one-hot encoding in binary classification:
        # Remove the encoder fitting line

        self.scaler = MinMaxScaler()
        X = self.scaler.fit_transform(X)
        
        
        X = self.reshape_input(X)
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # No need to reshape X_train since we are assuming
        # it has already been reshaped for LSTM layers appropriately

        # Build the model
        self.model = Sequential([
    Flatten(input_shape=(X.shape[1], X.shape[2])),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

        # Compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  

        # Train the model
        self.model.fit(
            X_train, 
            y_train, 
            epochs=self.epochs,  # Replace self.epochs with the actual number of epochs you want
            batch_size=32,       # The batch size
            verbose=1
        )

        predictions = self.model.predict(X_test)

        self.set_metrics((predictions > self.prob).astype("int32") , y_test)
        self.model.fit( X_test, 
            y_test, 
            epochs=self.epochs,  # Replace self.epochs with the actual number of epochs you want
            batch_size=32,       # The batch size
            verbose=1)
        return True




    def predict(self, long_df):
        X = long_df[self.features].values
        X = self.scaler.transform(X)
        X = self.reshape_input(X)
        
        predictions = self.model.predict(X)
        return predictions
    


    




