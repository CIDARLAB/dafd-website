"""
Created on Thurs Oct 20 2:39 PM 2022

@author: dpmcintyre
"""
import keras.models
import numpy as np #array & sqrt stats
from keras.layers import Dense
from keras.models import Sequential

from keras import backend
from keras import optimizers
import os
from keras import regularizers
import xgboost as xgb


class NeuralNetModel_DAFD3:
    model = None

    def __init__(self, load=True):
        if load:
            self.load_model()
        else:
            self.build_model()

    def build_model(self):
        # Initializing NN
        self.model = Sequential()
        # more layers result in high over-fitting, a simple 2 layer model is used here.
        # Adding the input layer and the first hidden layer
        self.model.add(Dense(units=512, activation='relu', input_dim=7, name='scratch_dense_1',
                        kernel_regularizer=regularizers.l2(0.001)))  # more options to avoid over-fitting

        self.model.add(Dense(units=16, activation='relu', name='scratch_dense_2',
                        kernel_regularizer=regularizers.l2(0.001)))  # more options to avoid over-fitting


        self.model.add(Dense(units=1, name='scratch_dense_6'))
                ### Optimizer
        adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)
                ### Compiling the NN
        self.model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error', rmse, r_square])


    def train_model(self, X_train, Y_train, X_test, Y_test):
        ### Fitting the model to the train set
        if self.model is None:
            self.build_model()

        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32,
                           epochs=5000)  # , callbacks=[earlystopping])

    def load_model(self, model_name="20231001_DAFD3_NN_7weights"):
        # load file
        file = os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name
        self.model = keras.models.load_model(file, custom_objects={"rmse": rmse, "r_square":r_square})

    def predict(self, features):
        return self.model.predict(np.asarray(features).reshape(1, -1))[0][0]


    def save_model(self, model_name = "my_model"):
        file = os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name
        self.model.save(file)

class XGBoost_DAFD3:
    model = None

    def __init__(self, load=True):
        if load:
            self.load_model()
        else:
            self.build_model()

    def build_model(self):
        self.model = xgb.XGBRegressor(tree_method="hist")

    def train_model(self, X_train, Y_train):
        if self.model is None:
            self.build_model()
        ### Fitting the model to the train set
        self.model.fit(X_train, Y_train)

    def save_model(self, name):
        self.model.save_model(name)

    def load_model(self, model_name="20231001_DAFD3_XGB.json"):
        # load file
        file = os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name
        if self.model is None:
            self.build_model()
        self.model.load_model(file)

    def predict(self, features):
        return self.model.predict(np.asarray(features).reshape(1, -1))[0]



# root mean squared error (rmse) for regression
def rmse(y_obs, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_obs), axis=-1))

# mean squared error (mse) for regression
def mse(y_obs, y_pred):
    return backend.mean(backend.square(y_pred - y_obs), axis=-1)

# coefficient of determination (R^2) for regression
def r_square(y_obs, y_pred):
    SS_res =  backend.sum(backend.square(y_obs - y_pred))
    SS_tot = backend.sum(backend.square(y_obs - backend.mean(y_obs)))
    return (1 - SS_res/(SS_tot + backend.epsilon()))

def mean_absolute_percentage_error(y_obs, y_pred):
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    y_obs=y_obs.reshape(-1,1)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100

def mean_absolute_percentage_error2(y_obs, y_pred): #for when the MAPE doesnt need reshaping
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100