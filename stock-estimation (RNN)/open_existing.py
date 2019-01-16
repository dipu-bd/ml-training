# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from keras.models import model_from_yaml
from sklearn.preprocessing import MinMaxScaler
from utils import generate_inputs, visualize_results

# The source folder
folder = 'lstm256x4-t180-adam-e1'

if not os.path.exists(folder):
    print('Folder does not exists')
    sys.exit(1)

# Open the model from the folder
with open(os.path.join(folder, 'model.yaml')) as f:
    yaml = f.read()

regressor = model_from_yaml(yaml)
regressor.compile('adam', loss='mean_squared_error')
timesteps = regressor.get_config()['layers'][0]['config']['batch_input_shape'][1]

# Load the weights
regressor.load_weights(os.path.join(folder, 'weights.hdf5'))

# Open datasets from the folder
dataset_train = np.load(os.path.join(folder, 'dataset_train.npy'))
dataset_test = np.load(os.path.join(folder, 'dataset_test.npy'))

sc = MinMaxScaler()
training_set = sc.fit_transform(dataset_train)
testing_set = sc.transform(dataset_test)

# Prepare the dataset
y_train = training_set[timesteps:].flatten()
X_train = generate_inputs(training_set, timesteps)
X_test = generate_inputs(testing_set, timesteps)

# Retrain the network
# regressor.fit(X_train, y_train, batch_size=32, epochs=10)

# Predict the results
predicted_stock = regressor.predict(X_test)
predicted_stock = sc.inverse_transform(predicted_stock)

predicted_stock = predicted_stock.flatten()
real_stock = dataset_test[-len(predicted_stock):].flatten()

# Visualize the prediction
visualize_results(real_stock, predicted_stock)
