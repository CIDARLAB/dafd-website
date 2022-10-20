"""
Created on Thurs Oct 20 2:39 PM 2022

@author: dpmcintyre
"""


import numpy as np #array & sqrt stats
import pandas as pd #data frame excel for python
import matplotlib.pyplot as plt #plots
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from keras.callbacks import EarlyStopping
from keras import backend
from keras import optimizers
import os
import h5py
import sklearn.metrics, math
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.utils import check_array
from keras import regularizers
import xgboost as xgb

class NeuralNetModel_DAFD3:
    model = None
    filepath = "resources/inputs/DAFD3_data.csv"

    def train_model(self, output_name, regime):
        # Initializing NN
        self.model = Sequential()
        # more layers result in high over-fitting, a simple 2 layer model is used here.
        # Adding the input layer and the first hidden layer
        self.model.add(Dense(units=512, activation='relu', input_dim=8, name='scratch_dense_1',
                        kernel_regularizer=regularizers.l2(0.001)))  # more options to avoid over-fitting

        self.model.add(Dense(units=16, activation='relu', name='scratch_dense_2',
                        kernel_regularizer=regularizers.l2(0.001)))  # more options to avoid over-fitting

        self.model.add(Dense(units=1, name='scratch_dense_6'))
                ### Optimizer
        adam = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, amsgrad=False)
                ### Compiling the NN
        self.model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error', rmse, r_square])




        ### Fitting the model to the train set
        X_train, Y_train, X_test, Y_test = self.prep_data()
        self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32,
                           epochs=5000)  # , callbacks=[earlystopping])

    def load_model(self, output_name):
        model_name = output_name

        # load json and create model
        json_file = open(os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name + ".json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.dirname(os.path.abspath(__file__)) + "/saved/" + model_name + ".h5")
        self.model = loaded_model

    def predict(self, features):
        return self.model.predict(np.asarray(features).reshape(1, -1))[0]

    def prep_data(self):
        df = pd.read_csv(self.filepath)
        X = df.loc[:, ['Orifice width (um)', 'Aspect ratio', 'Flow rate ratio', 'New_ca_number', 'Normalized oil inlet',
                       'Normalized water inlet', 'Expansion ratio', 'viscosity ratio']]

        Y = df.loc[:,
            'Norm hyd size']  # make sure to update Ori parameter to be the same parameter for normaliztion if Y is Norm Hyd size, Ori should be Hydraulic diameter; if Y is Norm size Ori should be orifice
        D = df.loc[:, 'Observed droplet diameter (um)']
        Z = df.loc[:, ['Observed generation rate (Hz)', 'Qin']]

        Ori = df.loc[:, 'Hyd_d']  # swap out to orifice when normalizing by orifice width
        Ori = np.array(Ori)

        X = np.array(X)
        Y = np.array(Y)  # Regime labels
        Z = np.array(Z)

        X1 = []  # Regime 1 data-set
        X2 = []  # Regime 2 data-set
        Y11 = []  # Regime 1 Output 1 (generation rate)
        Y12 = []  # Regime 1 Output 2 (size)
        Y21 = []  # Regime 2 Output 1 (generation rate)
        Y22 = []  # Regime 2 Output 2 (size)

        Y12 = Y
        X1 = X

        ###train-test split
        validation_size = 0.20
        X_train, X_test, Y_train, Y_test, Ori_train, Ori_test, D_train, D_test, Z_train, Z_test = model_selection.train_test_split(
            X1, Y12, Ori, D, Z, test_size=validation_size)  # Regime 1 Output 2

        ###data scaling
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        return X_train, Y_train, X_test, Y_test

class XGBoost_DAFD3:
    model = None
    filepath = "resources/inputs/DAFD3_data.csv"

    def train_model(self, output_name, regime):
        ### Fitting the model to the train set
        X_train, Y_train, X_test, Y_test = self.prep_data()
        self.model = xgb.XGBRegressor(tree_method="hist").fit(X_train, Y_train)

    def load_model(self, output_name):
        #TODO: figure out how/if I need to load weights for the XGboost model
        return None

    def predict(self, features):
        return self.model.predict(np.asarray(features).reshape(1, -1))[0]

    def prep_data(self):
        df = pd.read_csv(self.filepath)
        X = df.loc[:, ['Orifice width (um)', 'Aspect ratio', 'Flow rate ratio', 'New_ca_number', 'Normalized oil inlet',
                       'Normalized water inlet', 'Expansion ratio', 'viscosity ratio']]

        Y = df.loc[:,
            'Norm hyd size']  # make sure to update Ori parameter to be the same parameter for normaliztion if Y is Norm Hyd size, Ori should be Hydraulic diameter; if Y is Norm size Ori should be orifice
        D = df.loc[:, 'Observed droplet diameter (um)']
        Z = df.loc[:, ['Observed generation rate (Hz)', 'Qin']]

        Ori = df.loc[:, 'Hyd_d']  # swap out to orifice when normalizing by orifice width
        Ori = np.array(Ori)

        X = np.array(X)
        Y = np.array(Y)  # Regime labels
        Z = np.array(Z)

        X1 = []  # Regime 1 data-set
        X2 = []  # Regime 2 data-set
        Y11 = []  # Regime 1 Output 1 (generation rate)
        Y12 = []  # Regime 1 Output 2 (size)
        Y21 = []  # Regime 2 Output 1 (generation rate)
        Y22 = []  # Regime 2 Output 2 (size)

        Y12 = Y
        X1 = X

        ###train-test split
        validation_size = 0.20
        X_train, X_test, Y_train, Y_test, Ori_train, Ori_test, D_train, D_test, Z_train, Z_test = model_selection.train_test_split(
            X1, Y12, Ori, D, Z, test_size=validation_size)  # Regime 1 Output 2

        ###data scaling
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        return X_train, Y_train, X_test, Y_test

