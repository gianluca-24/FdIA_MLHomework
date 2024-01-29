#da caricare su colab
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

#la logistic regression usa la cross entropy invece dell'MSE
df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[: , 2:]

X = df.drop('shares', axis = 1)
y = df['shares'].apply(lambda x: 1 if x > 1400 else 0)

# X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 290601)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#logistic regression with scikit learn
# ////////////////////////////////////
# logisticRegr = LogisticRegression()
# logisticRegr.fit(X_train, y_train)
# predictions = logisticRegr.predict(X_test)

# score = logisticRegr.score(X_test, y_test)
# print(score)
# ////////////////////////////////////

#sigmoid restituisce un vettore [numpy.ndarray]
def sigmoid(z):
  return (1 / (1+np.exp(-z)))

class Logistic_Regression():

    def __init__(self, alpha = 0.01, iters = 1000):
        self.alpha = alpha
        self.iters = iters
        self.w = None
        self.b = None

    def fit(self,X,y):
        samples, features = X.shape
        self.w = np.zeros(features) #crea un array di dimensione data riempito di zeri
        self.b = 0

        for i in range(self.iters):
            #si effettua la predizione mediante regressione lineare e quello che si 
            #ottiene lo si usa per calcolare la logistic function
            lin_pred = np.dot(X,self.w) + self.b #predizione lineare
            predictions = sigmoid(lin_pred)
            # print(predictions)

            #calcola la variazione
            dw = (1 / samples) * np.dot(X.T, (predictions - y))
            db = (1 / samples) * np.sum(predictions - y)

            #update di pesi e bias
            self.w = self.w - self.alpha * dw
            self.b = self.b - self.alpha * db

    def predict(self, X):
        lin_pred = np.dot(X,self.w) + self.b #predizione lineare
        y_pred = sigmoid(lin_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
    
clf = Logistic_Regression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
     return np.sum(y_pred == y_test) / len(y_test)

acc = accuracy(y_pred, y_test)
print(acc)