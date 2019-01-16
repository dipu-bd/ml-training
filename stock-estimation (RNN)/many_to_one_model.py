# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

epochs = 100
timesteps = 180
batch_size = 32
folder = 'lstm256x4-t180-adam-e100'

os.makedirs(folder, exist_ok=True)

# Preparing the dataset
dataset = pd.read_csv('GOOGL.csv')
training_len = len(dataset) - timesteps

dataset_train = dataset.iloc[:training_len, 1:2].values
np.save(os.path.join(folder, 'dataset_train.npy'), dataset_train)

dataset_test = dataset.iloc[training_len - timesteps:, 1:2].values
np.save(os.path.join(folder, 'dataset_test.npy'), dataset_test)

# ... feature scaling
sc = MinMaxScaler()
training_set = sc.fit_transform(dataset_train)
testing_set = sc.transform(dataset_test)

# Populate the input and outputs for training
y_train = []
X_train = []
for i in range(timesteps, len(training_set)):
    X_train.append(training_set[i - timesteps: i, 0])
    y_train.append(training_set[i, 0])

y_train = np.array(y_train)
X_train = np.array(X_train)
X_train = np.reshape(X_train, (len(X_train), timesteps, 1))

# Populate the input and outputs for testing
X_test = []
for i in range(timesteps, len(testing_set)):
    X_test.append(testing_set[i - timesteps: i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the RNN model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

regressor = Sequential()

regressor.add(LSTM(256, return_sequences=True, input_shape=(timesteps, 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(256, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(256, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(256))
regressor.add(Dropout(0.2))

regressor.add(Dense(1))

regressor.compile('adam', loss='mean_squared_error')
with f as open(os.path.join(folder, 'model.yaml')):
    f.write(regressor.to_yaml())

regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
regressor.save_weights(os.path.join(folder, 'weights.hdf5'))

# Predict the results
predicted_stock = regressor.predict(X_test)
predicted_stock = sc.inverse_transform(predicted_stock)

predicted_stock = predicted_stock.flatten()
real_stock = dataset.iloc[training_len:, 1].values

rmse = np.sqrt(np.mean((predicted_stock - real_stock) ** 2))
print('RMSE =', rmse)

# Visualize the prediction
import matplotlib.pyplot as plt
plt.plot(real_stock, color='red', label='Real Stock Price')
plt.plot(predicted_stock, color='blue', label='Predicted Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time (days)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
