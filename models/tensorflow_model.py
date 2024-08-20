from datetime import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from .abstract_model import AbstractModel

class TensorFlowModel(AbstractModel):
    epochs = 50
    prob = 0.61
    last_save_time = None
    encoder = None
    scaler = None

    def __init__(self) -> None:
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = MinMaxScaler()
        super().__init__()

    def save(self, path='./.models'):
        self.model.save(f'{path}/tensorflow/model.h5')
        joblib.dump(self.scaler, f'{path}/tensorflow/scaler.pkl')
        joblib.dump(self.encoder, f'{path}/tensorflow/encoder.pkl')

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.last_save_time = current_time

        with open(f'{path}/tensorflow/time.txt', 'w') as f:
            f.write(current_time)

    def load(self, path='./.models'):
        self.model = load_model(f'{path}/tensorflow/model.h5')
        self.scaler = joblib.load(f'{path}/tensorflow/scaler.pkl')
        self.encoder = joblib.load(f'{path}/tensorflow/encoder.pkl')

        with open(f'{path}/tensorflow/time.txt', 'r') as f:
            self.last_save_time = f.read()

    def reshape_input(self, X):
        return X.reshape((X.shape[0], X.shape[1], 1))

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        y = self.encoder.fit_transform(y.reshape(-1, 1))

        self.model = Sequential([
            Dense(256, input_dim=X.shape[1], activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(y.shape[1], activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model.fit(X, y, epochs=self.epochs, batch_size=32, verbose=1)
        return True

    def predict(self, long_df):
        X = long_df[self.features].values
        X = self.scaler.transform(X)
        predictions = self.model.predict(X)
        return predictions

    def predict_last(self):
        if self.last_long_df is None:
            raise Exception('Must be trained')
        predictions = self.predict(self.last_long_df)
        predicted_df = self.last_df
        predicted_df['color_and_value'] = self.last_long_df['color_and_value'].values
        predicted_df['next_color_code'] = self.encoder.inverse_transform(predictions)
        print(predictions)
        return predicted_df
