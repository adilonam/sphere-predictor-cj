from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from .abstract_model import AbstractModel
import numpy as np



class RandomForest(AbstractModel):
    window_size = 4
    def create_sliding_window_features(self , data, window_size):
            new_data = []
            for i in range(window_size, len(data)):
                new_data.append(data[i-window_size:i+1])
            return np.array(new_data) 
    
    def fit(self ,  X , y ):
        # Splitting the dataset into the Training set and Test set
        # Now split the windowed dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
       

      
        # Train the classifier using the training data
        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)
        self.set_metrics(predictions , y_test )

        self.model.fit(X_test, y_test)
        return True
    
    
    def predict(self, long_df):
        X = long_df[self.features].values
        X = X.reshape(X.shape[0], -1)
        predictions = self.model.predict(X) 
        return predictions



    




