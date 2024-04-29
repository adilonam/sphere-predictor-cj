import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from .abstract_model import AbstractModel
from sklearn.metrics import mean_squared_error  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np



class TensorFlowModel(AbstractModel):
    epochs = 10

    def __init__(self) -> None:
        self.encoder = OneHotEncoder()
        super().__init__()
        
    
 

    def reshape_input(self , X):
        return X.reshape((X.shape[0], X.shape[1], 1))



    
    def fit(self ,  X, y):
        # Splitting the dataset into the Training set and Test set
        X = self.reshape_input(X)
        # Convert labels to one-hot encoding
        y = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Assuming x_train is already reshaped appropriately for LSTM layers and y_train is one-hot encoded
        num_classes = y_train.shape[1]  # Assuming y_train is one-hot encoded with shape (num_samples, num_classes)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        self.model = Sequential()
        self.model.add(LSTM(128, return_sequences=True, input_shape=( X_train.shape[1], 1)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(LSTM(64, return_sequences=False))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))

        # Output layer with 'num_classes' neurons and softmax activation function for multi-class classification
        self.model.add(Dense(num_classes, activation='softmax'))

        # Compile the model with categorical crossentropy loss function and an optimizer of your choice
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(
                X_train, 
                y_train, 
                epochs=self.epochs,                 # The number of epochs to train for
                batch_size=32,             # The batch size
                verbose = 1
            )

        # Model summary
        # get accuracy
        predictions = self.model.predict(X_test)

        self.set_metrics(self.decode(predictions) , self.decode(y_test) )
        return True


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
    

    def predict(self, long_df):
        X = long_df[self.features].values
        X = self.reshape_input(X)
        predictions = self.model.predict(X)
        predictions = self.decode(predictions)
        return predictions
    


    




