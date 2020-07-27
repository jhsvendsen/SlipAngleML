# RNN - LSTM

###############################################################################
# Importing key libraries
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##############################################################################
# Preprocessing Data
##############################################################################

# Importing the training set
dataset = pd.read_csv('D:\GitHub\DataPostProcess-ML\Python\Models\SlipAngleModel\FirstGen\Data\LotusR12520LapsMugello.csv')

# Removing NaN
dataset = dataset.dropna()


# Creating the X variables
training_set = dataset.drop(columns= ["rTyreSlipFL",
                           "rTyreSlipRL",
                           "rTyreSlipRR",
                           "aTyreSlipFR",
                           "aTyreSlipFL",
                           "aTyreSlipRL",
                           "aTyreSlipRR",
                           "ndSlipFL",
                           "ndSlipFR",
                           "ndSlipRL",
                           "ndSlipRR",
                           "isInPit",
                           "vCar_kph",
                           "vCar_mph",
                           "tLap",
                           "tLastLap",
                           "tBestLap",
                           "nLap",
                           "isTcEnabled",
                           "isTcInAction",
                           "isAbsEnabled",
                           "aContactPatchSlipFL",
                           "aContactPatchSlipFR",
                           "aContactPatchSlipRL",
                           "aContactPatchSlipRR",
                           "aPitch",
                           "tyreRadiusFL",
                           "tyreRadiusFR",
                           "tyreRadiusRL",
                           "tyreRadiusRR"])


# Encoding True/False into 0 and 1
arraynames = ['isEngineLimiterOn','isAbsInAction']
for name in arraynames:
    training_set[name] = training_set[name].astype(int)
    
# Plot some data 

# Checking head
training_set_head = training_set.head()
training_set_X = training_set.drop(columns= ['rTyreSlipFR'])
training_set_y = training_set['rTyreSlipFR'].values
training_set_y = training_set_y.reshape(-1,1)

# Feature Scaling X and transfer to numpy array
min_ts_X = training_set_X.min(axis = 0)
max_ts_X = training_set_X.max(axis = 0)
training_set_X_scaled = (training_set_X - min_ts_X)/(max_ts_X - min_ts_X)
has_nan = pd.isnull(training_set_X_scaled).values.any()
assert has_nan == False
training_set_X_scaled = training_set_X_scaled.to_numpy()

# Feature Scaling y
min_ts_y = training_set_y.min(axis = 0)
max_ts_y = training_set_y.max(axis = 0)
training_set_y_scaled = (training_set_y - min_ts_y)/(max_ts_y - min_ts_y)
training_set_y_scaled.reshape(-1,1)

# Creating a data structure with 60(can be changed) timesteps and 1 output
lookback = 30 # Decide how far to look back
X_train = []
y_train = []
for i in range(lookback, training_set_X_scaled.shape[0]):
    X_train.append(training_set_X_scaled[i-lookback:i, 0])
    y_train.append(training_set_y_scaled[i, 0])
        
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

##############################################################################
# Training Recurrent Neural Network
##############################################################################

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import CuDNNLSTM
from keras.layers import Dropout


# Initialising the RNN
regressor = Sequential()

# adding first layer
regressor.add(LSTM(100, return_sequences = True,
                   input_shape = ( X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# adding third layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# adding third layer
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# adding fourth layer last layer so return = false
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units = 1))

# Compilling the RNN
regressor.compile(optimizer = 'rmsprop', # try rmsprop
                  loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 20, batch_size = 200)


##############################################################################
# Downloading Test Data
##############################################################################

# Importing the training set
dataset_test = pd.read_csv('D:\GitHub\DataPostProcess-ML\Python\Models\SlipAngleModel\FirstGen\Data\LotusR1252LapsMugello.csv')

# Removing NaN
dataset_test = dataset_test.dropna()

# Creating the y variabel
test_set_y = dataset_test["rTyreSlipFR"]
gvert = dataset_test["gVert"]
glat = dataset_test["gLat"]
test_set_y_scaled = (test_set_y - min_ts_y)/(max_ts_y - min_ts_y)

test_set_X = dataset_test.drop(columns= ["rTyreSlipFR",
                                         "rTyreSlipFL",
                                         "rTyreSlipRL",
                                         "rTyreSlipRR",
                                         "aTyreSlipFR",
                                         "aTyreSlipFL",
                                         "aTyreSlipRL",
                                         "aTyreSlipRR",
                                         "ndSlipFL",
                                         "ndSlipFR",
                                         "ndSlipRL",
                                         "ndSlipRR",
                                         "isInPit",
                                         "vCar_kph",
                                         "vCar_mph",
                                         "tLap",
                                         "tLastLap",
                                         "tBestLap",
                                         "nLap",
                                         "isTcEnabled",
                                         "isTcInAction",
                                         "isAbsEnabled",
                                         "aContactPatchSlipFL",
                                         "aContactPatchSlipFR",
                                         "aContactPatchSlipRL",
                                         "aContactPatchSlipRR",
                                         "aPitch",
                                         "tyreRadiusFL",
                                         "tyreRadiusFR",
                                         "tyreRadiusRL",
                                         "tyreRadiusRR"])


# Encoding True/False into 0 and 1
for name in arraynames:
    test_set_X[name] = test_set_X[name].astype(int)


# Getting the predicted values
dataset_total = pd.concat((training_set_X , test_set_X), axis = 0)
inputs = dataset_total[len(dataset_total) - len(test_set_X) - lookback:]

# Scaling input data for model
inputs = (inputs - min_ts_X)/(max_ts_X - min_ts_X)
inputs = inputs.to_numpy()

# Creating a data structure
X_test = []
for i in range(lookback, inputs.shape[0]):
    X_test.append(inputs[i-lookback:i, 0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

##############################################################################
# Prediction and Visualising the data
##############################################################################

predicted_rTyreSlipFR_scaled = regressor.predict(X_test)
predicted_rTyreSlipFR = pd.DataFrame(data = predicted_rTyreSlipFR_scaled )
predicted_rTyreSlipFR = (predicted_rTyreSlipFR * (max_ts_y - min_ts_y)) + min_ts_y

# Visualising the results
plt.plot(test_set_y_scaled, color = 'red', label = 'Real scaled')
plt.plot(predicted_rTyreSlipFR_scaled , color = 'blue', label = 'Pred scaled')
plt.title('Scaled')
plt.xlabel('Time')
plt.ylabel('r slip angle')
plt.legend()
plt.show()

# Visualising the results
plt.plot(test_set_y, color = 'red', label = 'Real')
plt.plot(predicted_rTyreSlipFR, color = 'blue', label = 'Pred')
plt.title('RNN Prediction of Slip angle FR from Assetto Corsa')
plt.xlabel('Time')
plt.ylabel('r slip angle')
plt.legend()
plt.show()

# Show gLat vs predicted slip angle
plt.plot(test_set_y, color = 'red', label = 'slip')
plt.show()
plt.plot(glat, color = 'blue', label = 'gvert')
plt.show()


