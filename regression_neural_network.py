import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing 

df = pd.read_csv('OnlineNewsPopularity/OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[:, 2:]

X = df.drop('shares',axis=1)
y = df['shares'].apply(lambda x: 1 if x > 1400 else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

def accuracy(y_pred, y_test):
  return np.sum(y_pred == y_test) / len(y_test)

class NeuralNetwork:
    def __init__(self, hidden, lr):
        self.lr = lr
        self.hidden = hidden
        self.weights = []

    def my_forward(self, X):
        self.lista = []
        self.lista_2 = []
        for i in self.weights:
            prod = np.dot(X, i)
            self.lista_2.append(prod)
            lista = self.sigmoid(prod / len(X))
            self.lista.append(lista)
            X = lista
        self.out = self.lista_2[-1]

    def my_backward(self, X, Y):
        self.error = self.out - Y
        self.delta = [self.error * self.lista[-1]]
        for i in range(len(self.weights)-2, -1, -1):
            delta = np.dot(self.delta[0], self.weights[i+1].T) * self.sigmoid_derivative(self.lista[i])
            self.delta.insert(0, delta)
        for x in range(len(self.weights)):
            if x != 0:
                value = self.lista[x-1]
            else:
                value = X
            self.weights[x] -= self.lr * value.reshape(-1, 1) @ self.delta[x].reshape(1, -1)
      
    def prepare_data(self, X, Y):
        X_copy = X.copy()
        X_copy.insert(loc=0, column='bias', value=[1 for x in range(X_copy.shape[0])])
        X_mat = X_copy.to_numpy()
        self.X_len = X_mat.shape[0]
        y_mat = Y.to_numpy().reshape(-1, 1)
        self.weights = []
        len_value = X_mat.shape[1]
        for x in self.hidden:
            self.weights.append(np.random.randn(len_value, x))
            len_value = x
        self.weights.append(np.random.randn(len_value, 1))
        return X_mat, y_mat

    def wrapper(self, X, Y, iterazioni):
        X_mat, y_mat = self.prepare_data(X, Y)
        for x in range(iterazioni):
            for i in range(X_mat.shape[0]):
                var_1 = X_mat[i]
                var_2 = y_mat[i]
                self.my_forward(var_1)
                self.my_backward(var_1, var_2)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def predict(self, X):
        X_copy = X.copy()
        X_copy.insert(loc = 0, column='bias', value=[1 for x in range(X_copy.shape[0])])
        X_mat = X_copy.to_numpy()
        y_pred_test = []
        for x in range(X_mat.shape[0]):
            self.my_forward(X_mat[x])
            y_pred_test.append(int(self.out > 0.5))
        return y_pred_test

nn_clas = NeuralNetwork([59, 30, 22], 0.01)
nn_clas.wrapper(X_train, y_train, 100)
y_pred = nn_clas.predict(X_test)

accuracy_NN = accuracy(y_pred,y_test)

print("Accuracy: ", accuracy_NN)