import pandas as pd
import openpyxl
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional, Flatten

class CjModel:
    features = ["date_as_int", 'day_of_year',"name_code", 'day_of_month', 'day_of_week', "v1", 'v2', "v3","v4"]
    target = 'v2'
    timesteps = 10
    color_code = {
        "FF0000": 1,  # red
        "00FF00": 2,  # green
        "FFFF00": 3, # yellow
        "00FFFF": 4,  # blue
        "FF9900": 5,  # orange
        "D5A6BD": 6,  # purple
         "CC4125": 7, # red
    }
    
    epochs = 50
    def __init__(self) -> None:
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = StandardScaler()
        super().__init__()
    
    def preprocess_excel(self, uploaded_file):
        # data structure of cell sheet 1 sheet 2 sheeet 3 sheet 4
        
        
        #get sheet1 
        df_sheet1 = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name='Sheet1')

        #get color from sheet 2
        wb = openpyxl.load_workbook(filename=uploaded_file)
        ws_sheet2 = wb['Sheet2']
        for row in ws_sheet2.iter_rows(min_row=2, min_col=2):
            for cell in row:
                color_hex = cell.fill.start_color.index[2:] if cell.fill.start_color.index else 'None'
                cell.value = color_hex
                
        with tempfile.NamedTemporaryFile() as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            df_sheet2 = pd.read_excel(tmp, engine='openpyxl', sheet_name='Sheet2')       
            df_sheet2 = df_sheet2.replace(self.color_code)

        
        #sheet3
        df_sheet3 = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name='Sheet3')
        #sheet4
        df_sheet4 = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name='Sheet4')
        
        
        combined_df = df_sheet1.astype(str) + ' | ' + df_sheet2.astype(str) + ' | ' + df_sheet3.astype(str) + ' | ' + df_sheet4.astype(str)


        combined_df.iloc[:, 0] = df_sheet1.iloc[:, 0]

        return combined_df

    

    def process_data(self, combined_df):
        
        long_df = pd.melt(combined_df, id_vars=['DATE'], var_name='date', value_name='all_values')


        long_df['name_code'] = long_df['DATE'].str.extract(r'(\d+)').astype(int)

        long_df['date'] = pd.to_datetime(long_df['date'])

        long_df['date_as_int'] = long_df['date'].astype(int) // 10**9 

        long_df['day_of_month'] = long_df['date'].dt.day

        long_df['day_of_week'] = long_df['date'].dt.dayofweek

        long_df['day_of_year'] = long_df['date'].dt.dayofyear.astype(int)


        long_df[['v1', 'v2', 'v3', "v4"]] = long_df['all_values'].str.split(' \| ', expand=True).astype(float)

        return long_df

    def process_excel(self, uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)



    def create_sequences(self , X,y, timesteps):
        sequences = []
        labels = []
        for i in range(len(X) - timesteps):
            sequences.append(X[i:i + timesteps])
            labels.append(y[i + timesteps])
            
        last_sequence = [X[len(X) - timesteps:len(X)]]
        
        return np.array(sequences), np.array(labels), np.array(last_sequence)

    def make_train_data(self, long_df, name_code):
        
        filtred_df = long_df[long_df['name_code'] == name_code] 
        X = filtred_df[self.features].values
        y = filtred_df[self.target].values
        
        X = self.scaler.fit_transform(X)
        y = self.encoder.fit_transform(y.reshape(-1, 1))
        
        X, y , last_X= self.create_sequences(X , y , self.timesteps)

        return X, y, last_X


    def fit(self , X, y):
        

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        input_shape = (self.timesteps, X.shape[2])
        # Define the LSTM model
        self.model =    Sequential([
    # Flatten the input before feeding into Dense layers
    Flatten(input_shape=input_shape),

    # First dense layer with Batch Normalization and Dropout
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),  # Dropout for regularization

    # Second dense layer with Batch Normalization and Dropout
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),  # Dropout for regularization

    # Third dense layer with Batch Normalization and Dropout
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),  # Dropout for regularization

    # Fourth dense layer with Batch Normalization and Dropout
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),  # Dropout for regularization

    # Output layer
    Dense(y_test.shape[1], activation='softmax')
])
               

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the self.model
        self.model.fit(X_train, y_train, epochs=self.epochs, verbose=1)

        # Evaluate on training data
        train_loss, train_accuracy = self.model.evaluate(X_train, y_train, verbose=0)
        print(f'Training Accuracy: {train_accuracy:.4f}')
        
        predictions = self.model.predict(X_test)
        predictions_labels = self.encoder.inverse_transform(predictions)

        # Evaluate on test data
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f'Test Accuracy: {test_accuracy:.4f}')
        
        self.model.fit(X_test, y_test, epochs=self.epochs, verbose=0)
        
        y_test_labels = self.encoder.inverse_transform(y_test)
        
        return predictions , y_test_labels

        

    def predict_last(self, last_X):
        prediction = self.model.predict(last_X)
        prediction = self.encoder.inverse_transform(prediction)
        return prediction
