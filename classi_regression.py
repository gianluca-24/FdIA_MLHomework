import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[: , 2:]

X = df.drop('shares', axis = 1)
y = df['shares'].apply(lambda x: 1 if x > 1400 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)

def threshold(z):
    return [1 if i >= 0 else 0 for i in z]

def accuracy(y_pred, y_test):
     return np.sum(y_pred == y_test) / len(y_test)

class LinearClassifier():

    def __init__(self, alpha = 0.01, iters = 1000):
        self.w = None
        self.b = None
        self.alpha = alpha
        self.iters = iters
    
    def fit(self, X, y):
        samples, features = X.shape
        self.w = np.zeros(features)
        self.b = 0

        for _ in range(self.iters):
            predictions = np.dot(X,self.w) + self.b
            pred_n = threshold(predictions)

            dw = (1/samples) * np.dot(X.T, (pred_n - y))
            db = (1/samples) * np.sum(pred_n - y)

            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

    def predict(self,X):
        pred = np.dot(X,self.w) + self.b
        l_pred = threshold(pred)
        return l_pred

classi = LinearClassifier()
classi.fit(X_train, y_train)
pred = classi.predict(X_test)
acc = accuracy(pred, y_test)
print(acc)

#sklearn linear classifier-------------------------
# clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
#---------------------------------------------

    

    
