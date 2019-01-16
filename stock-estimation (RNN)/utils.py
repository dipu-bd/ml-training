# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def generate_inputs(dataset, timesteps):
    X = []
    for i in range(timesteps, len(dataset)):
        X.append(dataset[i - timesteps:i])
    # end for
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X


def visualize_results(real, predicted):
    plt.plot(real, color='red', label='Real Stock Price')
    plt.plot(predicted, color='blue', label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time (days)')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    print('RMSE =', np.sqrt(np.mean((predicted - real) ** 2)))
