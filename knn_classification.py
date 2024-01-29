import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[:, 2:]

X = df.drop('shares', axis=1).values
y = (df['shares'].values > 1400).astype(int) # Converto la variabile target in binaria (0 o 1) a seconda che il numero di condivisioni sia inferiore o superiore a 1400

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


def k_nearest_neighbors(X_train, y_train, X_test, k):
    y_pred = []
    for i in range(X_test.shape[0]):
        distances = np.sqrt(np.sum((X_train - X_test[i]) ** 2, axis=1))
        indices = np.argsort(distances)[:k]
        y_pred.append(np.bincount(y_train[indices]).argmax())
    return np.array(y_pred)

y_pred = k_nearest_neighbors(X_train, y_train, X_test, k=5)

accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)
