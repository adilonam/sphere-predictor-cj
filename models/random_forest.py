from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .abstract_model import AbstractModel




class RandomForest(AbstractModel):


    def fit(self ,  X , y ):
        # Splitting the dataset into the Training set and Test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train the classifier using the training data
        self.model.fit(X_train, y_train)

        predicted_colors = self.model.predict(X_test)

        self.set_metrics(predicted_colors , y_test )
        return True
    
    
    def predict(self, long_df):
        X = long_df[self.features].values
        predictions = self.model.predict(X)
        return predictions



    




