import pandas as pd
import openpyxl
import tempfile
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np

from models.abstract_model import AbstractModel





class TimeSeries(AbstractModel):
    def __init__(self) -> None:
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = MinMaxScaler()
        super().__init__()
        
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
        "FFFF00": 3,
        "00FFFF": 4,
        "FF9900": 5,
        "D5A6BD": 6,
    }
    timesteps = 20

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
    
    def split_data(self, df):
        
        X = df[self.features].values
        y = df[self.target].values
        
        X = self.scaler.fit_transform(X)
        y = self.encoder.fit_transform(y.reshape(-1, 1))

        X, y = self.create_sequences(X, y , self.timesteps)
        return X, y
    
    def create_sequences(self , X,y, timesteps):
        sequences = []
        labels = []
        for i in range(len(X) - timesteps):
            sequences.append(X[i:i + timesteps])
            labels.append(y[i + timesteps-1])
        
        
        return np.array(sequences), np.array(labels)
        
        
    
    