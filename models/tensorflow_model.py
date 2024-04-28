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



class TensoFlowModel(AbstractModel):
    epochs = 20

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

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        return  X_train, X_test, y_train, y_test
    

    def fit(self ,  X_train, X_test, y_train, y_test):
        # Splitting the dataset into the Training set and Test set

        

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
            )

        # Model summary
        # get accuracy
        predictions = self.model.predict(X_test)

        predicted_labels = predictions.argmax(axis=1)
        predicted_colors =  self.decode(predicted_labels)

        true_labels = np.argmax(y_test, axis=1)
        true_colors = self.decode(true_labels)

        self.mse = mean_squared_error(true_colors, predicted_colors)
        # Calculate the number of correct predictions
        correct_predictions = sum(true_colors == predicted_colors)
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
        predictions = self.model.predict(self.X_predict)
        predicted_labels = predictions.argmax(axis=1)
        predicted_colors =  self.decode(predicted_labels)

        predicted_df = self.last_df
        predicted_df['next_color_code'] =  predicted_colors
        predicted_df['next_color_code'] = predicted_df['next_color_code'].round().astype(int)
        predicted_df['next_color'] = predicted_df['next_color_code'].map( lambda x: self.color_mapper(x) )
        return predicted_df
    
    def decode(self , predicted_labels):
        one_hot_predictions = np.zeros((predicted_labels.shape[0], len(self.encoder.categories_[0])))

        # Set the predicted labels to 1
        for i, label in enumerate(predicted_labels):
            one_hot_predictions[i, label] = 1

        # Step 2: Use the encoder to inverse transform the one-hot encoded predictions
        y_test_pred = self.encoder.inverse_transform(one_hot_predictions)
        y_test_pred = y_test_pred.reshape(y_test_pred.shape[0]).astype(int)
        return y_test_pred



    




