import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class linear_regression:

    def fit(data):
        # Assuming 'data' is a pandas DataFrame containing your data as provided in the original format
        # Convert the date columns (excluding the 'NAME' column)
        for col in data.columns[1:]:
            data[col] = data[col].apply(lambda x: int(x, 16))

        # Processing the DataFrame 'data' to have "date", "name", "color_value" columns
        data_long = pd.melt(data, id_vars=['NAME'], var_name='date', value_name='color_value')
        data_long['date'] = pd.to_datetime(data_long['date'])

        # Convert dates to a numerical value, such as the day of the year
        data_long['day_of_year'] = data_long['date'].dt.dayofyear

        # Prepare the dataset for Linear Regression
        X = data_long['day_of_year'].values.reshape(-1, 1)  # Feature
        y = data_long['color_value'].values  # Target

        # Splitting the dataset into the Training set and Test set
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the model
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting new values (You would add new 'day_of_year' values here to make predictions)
        predicted_colors = regressor.predict(X_test)

        # If you want to see the predicted hexadecimal color value
        predicted_hex_colors = [hex(int(value))[2:].upper().zfill(8) for value in predicted_colors]

        # The `predicted_hex_colors` now contains the predicted colors in hexadecimal format.
