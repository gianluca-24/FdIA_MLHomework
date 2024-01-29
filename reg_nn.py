import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

# Lettura del dataset
data = pd.read_csv('OnlineNewsPopularity.csv')
data = data.rename(columns=lambda x: x.strip())
data = data.iloc[: , 2:]

# Preprocessing dei dati
X = data.drop('shares', axis = 1)
y = data['shares']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class NeuralNetwork():
    def __init__(self, num_input_neurons, num_hidden_neurons, num_output_neurons):
        self.num_input_neurons = num_input_neurons
        self.num_hidden_neurons = num_hidden_neurons
        self.num_output_neurons = num_output_neurons
        self.hidden_weights = np.random.rand(self.num_input_neurons, self.num_hidden_neurons)
        self.output_weights = np.random.rand(self.num_hidden_neurons, self.num_output_neurons)
        self.hidden_bias = np.random.rand(1, self.num_hidden_neurons)
        self.output_bias = np.random.rand(1, self.num_output_neurons)

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def forward_propagate(self, input_data):
        self.hidden_layer_output = self.sigmoid(np.dot(input_data, self.hidden_weights) 
                                                + self.hidden_bias)
        self.output_layer_output = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        return self.output_layer_output

    def back_propagate(self, input_data, target_output, learning_rate):
        output_error = target_output - self.output_layer_output
        output_delta = output_error
        hidden_error = np.dot(output_delta, self.output_weights.T)
        hidden_delta = hidden_error * self.hidden_layer_output * (1 - self.hidden_layer_output)
        self.hidden_weights += learning_rate * np.dot(input_data.T, hidden_delta)
        self.output_weights += learning_rate * np.dot(self.hidden_layer_output.T, output_delta)
        self.hidden_bias += learning_rate * np.sum(hidden_delta, axis=0)
        self.output_bias += learning_rate * np.sum(output_delta, axis=0)

    def train(self, input_data, target_output, learning_rate, epochs):
        for epoch in range(epochs):
            error = 0
            for i in range(len(input_data)):
                output = self.forward_propagate(input_data[i])
                self.back_propagate(input_data[i], target_output[i], learning_rate)
                error += np.mean(np.square(target_output[i] - output))
            if epoch % 100 == 0:
                print("Epoch",epoch," with error",error)

    def predict(self, input_data):
        return self.forward_propagate(input_data)
    

num_samples = 1000
input_data = np.random.uniform(-10, 10, size=(num_samples, 1))
target_output = np.sin(input_data) + np.random.normal(0, 0.5, size=(num_samples, 1))

num_input_neurons = 1
num_hidden_neurons = 10
num_output_neurons = 1
learning_rate = 0.01
epochs = 5000

nn = NeuralNetwork(num_input_neurons, num_hidden_neurons, num_output_neurons)
nn.train(input_data, target_output, learning_rate, epochs)
test_input = np.random.uniform(-10, 10, size=(100, 1))
predicted_output = nn.predict(test_input)