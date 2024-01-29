import numpy as np
from collections import Counter
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 

# load your dataset and split it into training and testing sets
df = pd.read_csv('OnlineNewsPopularity.csv')
df = df.rename(columns=lambda x: x.strip())
df = df.iloc[: , 2:]

X = df.drop('shares', axis = 1)
y = df['shares'].apply(lambda x: 1 if x > 1400 else 0).values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# define your neural network parameters
input_layer_size = X_train.shape[1] #gli input sono le feature
hidden_layer_size = 2
output_layer_size = 1

# define the activation functions
def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)


# # initialize the weights and biases of the neural network
W1 = np.random.randn(input_layer_size, hidden_layer_size) * 0.01
b1 = np.zeros((1, hidden_layer_size))
W2 = np.random.randn(hidden_layer_size, output_layer_size) * 0.01
b2 = np.zeros((1, output_layer_size))

# set your learning rate and number of epochs
learning_rate = 0.01
epochs = 1000

# train your neural network using backpropagation
for epoch in range(epochs):
    # forward propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    Y_pred = sigmoid(Z2)
#     # compute the loss
    loss = np.mean((y_train - Y_pred)**2)

#     # backward propagation
    dZ2 = Y_pred - y_train
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # update the weights and biases
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

# make predictions on the testing set
Z1 = np.dot(X_test, W1) + b1
print(Z1)
A1 = relu(Z1)
print(A1)
Z2 = np.dot(A1, W2) + b2
print(Z2)
Y_pred = sigmoid(Z2)
print(Y_pred)
# convert the continuous predictions into binary classes using a threshold value
Y_pred = np.where(Y_pred >= 0.5, 1, 0)
Y_pred = np.reshape(Y_pred,-1)


# evaluate the performance of your binary classifier using standard classification metrics

acc = accuracy_score(y_test, Y_pred)

print('Accuracy:', acc)