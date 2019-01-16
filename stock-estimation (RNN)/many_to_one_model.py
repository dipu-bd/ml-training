# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Configurations
epochs = 300
batch_size = 32
algorithm = 'adam'
timesteps = 180
dataset_file = 'GOOGL.csv'
folder = 'lstm256x5'

# Folder to save
folder += '-t%d-%s-e%d (%s)' % (timesteps, algorithm, epochs, dataset_file)
os.makedirs(folder, exist_ok=True)

# Preparing the dataset
dataset = pd.read_csv(os.path.join('data', dataset_file))
training_len = len(dataset) - timesteps

dataset_train = dataset.iloc[:training_len, 1:2].values
np.save(os.path.join(folder, 'dataset_train.npy'), dataset_train)

dataset_test = dataset.iloc[training_len - timesteps:, 1:2].values
np.save(os.path.join(folder, 'dataset_test.npy'), dataset_test)

# Feature scaling
sc = MinMaxScaler()
training_set = sc.fit_transform(dataset_train)
testing_set = sc.transform(dataset_test)

# Populate the input and outputs for training
from utils import generate_inputs
y_train = training_set[timesteps:].flatten()
X_train = generate_inputs(training_set, timesteps)
X_test = generate_inputs(testing_set, timesteps)

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

regressor.add(LSTM(256, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(256))
regressor.add(Dropout(0.2))

regressor.add(Dense(1))

regressor.compile(algorithm, loss='mean_squared_error')

# Save the model config
with open(os.path.join(folder, 'model.yaml'), 'w') as f:
    f.write(regressor.to_yaml())

# Train the model
regressor.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# Save the trained weights
regressor.save_weights(os.path.join(folder, 'weights.hdf5'))

# Predict the results
predicted_stock = regressor.predict(X_test)
predicted_stock = sc.inverse_transform(predicted_stock)

predicted_stock = predicted_stock.flatten()
real_stock = dataset.iloc[training_len:, 1].values

# Visualize the prediction
from utils import visualize_results
visualize_results(real_stock, predicted_stock)
