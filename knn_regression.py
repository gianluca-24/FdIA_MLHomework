import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def k_nearest_neighbors(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(X_test.shape[0]):
        distances = np.sqrt(np.sum((X_train - X_test[i])**2, axis=1))
        indices = np.argsort(distances)[:k]
        y_pred.append(np.mean(y_train[indices]))
    return np.array(y_pred)


df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[:, 2:]

X = df.drop('shares', axis=1).values
y = df['shares'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_pred = k_nearest_neighbors(X_train, y_train, X_test, k=5)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("RMSE:", rmse)
