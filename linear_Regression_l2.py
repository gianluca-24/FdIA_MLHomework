#da caricare su colab

import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.metrics import mean_squared_error, r2_score

from linear_regression import LinearRegression

df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[: , 2:]

X = df.drop('shares', axis = 1)
y = df['shares']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#funzione RMSE
def ridge(pred, y, samp):
    return np.sqrt((1 / samp) * np.sum((y - pred) ** 2))

class LinearRegression():

    def __init__(self, iters = 1000, alpha = 0.01, lmbd_2 = 0.01):
        self.w = None
        self.b = None
        self.alpha = alpha
        self.iters = iters
        self.lmbd_2 = lmbd_2

    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iters):
            predictions = np.dot(X,self.w) + self.b

            #da verificare formule aggiornamento
            dw = (1 / samples) * np.dot(X.T, (predictions - y)) + self.lmbd_2 * (np.sign(self.w) ** 2)
            db = (1 / samples) * np.sum(predictions - y)

            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

    def predict(self, X):
        a = np.dot(X,self.w) + self.b
        return a
    
rl = LinearRegression()
rl.fit(X_train, y_train)
pred = rl.predict(X_test)

ret_rmse = ridge(pred, y_test, X_test.shape[0])

print(ret_rmse)