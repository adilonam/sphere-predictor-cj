import pandas as pd
import openpyxl
import tempfile
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

class AbstractModel:
    color_mapping = {}
    features = ["date_code", "name_code", 'day_of_year', 'day_of_month', 'day_of_week', "color_group", 'value', "extra"]
    target = 'next_color_group'
    last_long_df = None
    preferred_color = ['00FF00', 'FFFF00', 'FF0000']
    last_df = None
    is_sheet2 = True
    color_group_dict = {
        "FF0000": 1,  # red
        "00FF00": 2,  # green
        "FFFF00": 3, # yellow
        "00FFFF": 4,  # blue
        "FF9900": 5,  # orange
        "D5A6BD": 6,  # purple
    }

    def preprocess_excel(self, uploaded_file):
        wb = openpyxl.load_workbook(filename=uploaded_file)
        ws_sheet1 = wb['Sheet1']
        for row in ws_sheet1.iter_rows(min_row=2, min_col=2):
            for cell in row:
                color_hex = cell.fill.start_color.index[2:] if cell.fill.start_color.index else 'None'
                cell.value = f"{cell.value} | {color_hex}"

        with tempfile.NamedTemporaryFile() as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            df_sheet1 = pd.read_excel(tmp, engine='openpyxl', sheet_name='Sheet1')

        if self.is_sheet2:
            df_sheet2 = pd.read_excel(uploaded_file, engine='openpyxl', sheet_name='Sheet2')
            assert df_sheet1.shape == df_sheet2.shape, "Shape in df_sheet1 and df_sheet2 do not match."
            for i in range(1, df_sheet1.shape[1]):
                df_sheet1.iloc[:, i] = df_sheet1.iloc[:, i].astype(str) + ' | ' + df_sheet2.iloc[:, i].astype(str)

        return df_sheet1

    def color_change(self, x):
        return 1 if x in self.preferred_color else 0

    def process_data(self, df):
        if self.color_mapping:
            raise Exception("Data has already been processed")

        first_column = df.iloc[:, 0]
        last_column = df.iloc[:, -1]
        self.last_df = pd.concat([first_column, last_column], axis=1)
        self.row_count = df.shape[0]

        long_df = pd.melt(df, id_vars=['NAME'], var_name='date', value_name='color_and_value')
        long_df['name_code'] = long_df['NAME'].str.extract(r'(\d+)').astype(int)
        long_df['date'] = pd.to_datetime(long_df['date'])
        long_df['day_of_month'] = long_df['date'].dt.day
        long_df['day_of_week'] = long_df['date'].dt.dayofweek
        long_df['date_code'] = pd.factorize(long_df['date'])[0]
        long_df['day_of_year'] = long_df['date'].dt.dayofyear.astype(int)

        if self.is_sheet2:
            long_df[['value', 'color', 'extra']] = long_df['color_and_value'].str.split(' \| ', expand=True)
            long_df['extra'] = long_df['extra'].astype(float)
        else:
            long_df[['value', 'color']] = long_df['color_and_value'].str.split(' \| ', expand=True)
        
        long_df['value'] = long_df['value'].astype(float)
        long_df['color_code'] = pd.factorize(long_df['color'])[0]
        long_df['next_color_code'] = long_df.groupby('name_code')['color_code'].shift(-1)
        long_df['previous_color_code'] = long_df.groupby('name_code')['color_code'].shift(1)
        long_df['color_binary'] = long_df['color'].map(self.color_change).astype(int)
        long_df['next_color_binary'] = long_df.groupby('name_code')['color_binary'].shift(-1)
        long_df['color_group'] = long_df['color'].map(self.color_group_dict).astype(int)
        long_df['next_color_group'] = long_df.groupby('name_code')['color_group'].shift(-1)

        return long_df

    def process_excel(self, uploaded_file):
        df = self.preprocess_excel(uploaded_file)
        return self.process_data(df)

    def set_metrics(self, predictions, y_test):
        self.mse = mean_squared_error(y_test, predictions)
        self.accuracy = accuracy_score(y_test, predictions)
        correct = 0
        self.predictions_count = 0
        self.y_test_count = 0

        for t, p in zip(y_test, predictions):
            if t == p and t == 1:
                correct += 1
            if p == 1:
                self.predictions_count += 1
            if t == 1:
                self.y_test_count += 1

        self.preferred_accuracy = correct / self.predictions_count if self.predictions_count != 0 else 0

    def color_mapper(self, x):
        if x in self.color_mapping:
            return self.color_mapping[x]
        else:
            last_key = sorted(self.color_mapping.keys())[-1]
            return self.color_mapping[last_key]

    def predict_last(self):
        if self.last_long_df is None:
            raise Exception('Must be trained')
        predictions = self.predict(self.last_long_df)
        predicted_df = self.last_df
        predicted_df['color_and_value'] = self.last_long_df['color_and_value'].values
        predicted_df['next_color_code'] = predictions
        return predicted_df

    def train_test_split(self, long_df):
        self.last_long_df = long_df[-self.row_count:]
        long_df = long_df.dropna(axis=0)
        X = long_df[self.features].values
        y = long_df[self.target].values

        return X, y

    def predict(self, X):
        raise NotImplementedError('Not implemented')
