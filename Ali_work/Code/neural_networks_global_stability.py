#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:17:27 2022

@author: alilashkaripour
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 20:07:01 2022

@author: alilashkaripour
"""




import numpy as np #array & sqrt stats
import pandas as pd #data frame excel for python
import xlrd  #loading excel files
import openpyxl #Ali added for excel not sure if it works
import matplotlib.pyplot as plt #plots
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from keras.callbacks import EarlyStopping
from keras import backend
from keras import optimizers
import h5py
import sklearn.metrics, math
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.utils import check_array
from keras import regularizers
import xgboost as xgb

#-----------------------------------------------------------------------------
#  Custom Loss Functions
#-----------------------------------------------------------------------------

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
    #y_obs, y_pred =check_array(y_obs, y_pred)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100


def mean_absolute_percentage_error2(y_obs, y_pred): #for when the MAPE doesnt need reshaping
    y_obs, y_pred = np.array(y_obs), np.array(y_pred)
    #y_obs=y_obs.reshape(-1,1)
    #y_obs, y_pred =check_array(y_obs, y_pred)
    return  np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100


#-----------------------------------------------------------------------------
#   Data Preparation
#-----------------------------------------------------------------------------
#normalization
    

#scaler=StandardScaler().fit(X_all)

maer_train = []
rmser_train = []
r2r_train = []
maper_train = []

maer_test = []
rmser_test = []
r2r_test = []
maper_test = []

maes_train = []
rmses_train = []
r2s_train = []
mapes_train = []

maes_test = []
rmses_test = []
r2s_test = []
mapes_test = []

mael = []
rmsel = []
r2l = []
mapel = []


for i in range(15):
#### add address of the dataset
    loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/NewAll_2in1_data_normalization.xlsx")

### Read data
    wb = pd.read_excel(loc ,engine='openpyxl')
#wb = xlrd.open_workbook(loc) 
#sheet = wb.sheet_by_index(0)

### Extract input and output
#X=wb.iloc[:,1:9] # Geometry + flow Features
    #X = wb.loc[:,['Orifice width (um)','Flow rate ratio','New_ca_number' ]]
    X = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

#Y = wb.loc[:, 'Norm size']#use for normalized size by orifice
    Y = wb.loc[:, 'Norm hyd size'] #make sure to update Ori parameter to be the same parameter for normaliztion if Y is Norm Hyd size, Ori should be Hydraulic diameter; if Y is Norm size Ori should be orifice
    D = wb.loc[:, 'Observed droplet diameter (um)']
    Z=wb.loc[:,['Observed generation rate (Hz)','Qin']]

        
    Ori= wb.loc[:, 'Hyd_d']#swap out to orifice when normalizing by orifice width

    Ori=np.array(Ori)

    X=np.array(X)
    Y=np.array(Y) # Regime labels
    Z=np.array(Z)


    X1=[] #Regime 1 data-set
    X2=[] #Regime 2 data-set
    Y11=[] # Regime 1 Output 1 (generation rate)
    Y12=[] # Regime 1 Output 2 (size)
    Y21=[] # Regime 2 Output 1 (generation rate)
    Y22=[] # Regime 2 Output 2 (size)


    Y12=Y
    X1=X

###train-test split
    validation_size = 0.20

    X_train, X_test, Y_train, Y_test, Ori_train, Ori_test, D_train, D_test, Z_train, Z_test = model_selection.train_test_split(X1, Y12, Ori, D, Z, test_size=validation_size) #Regime 1 Output 2

###data scaling
    scaler=StandardScaler().fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)

    X_train =np.array(X_train)
    Y_train=np.array(Y_train)
    X_test =np.array(X_test)
    Y_test =np.array(Y_test)
    Y12 = np.array(Y12)
#-----------------------------------------------------------------------------
#   Training Nueral Network Model
#-----------------------------------------------------------------------------

### Initializing NN
    model = Sequential()
### more layers result in high over-fitting, a simple 2 layer model is used here.
### Adding the input layer and the first hidden layer
    model.add(Dense(units = 512, activation = 'relu', input_dim = 8 , name='scratch_dense_1', kernel_regularizer=regularizers.l2(0.001))) #more options to avoid over-fitting
### Droput to avoid further over-fitting
    #model.add(Dropout(0.8))
    model.add(Dense(units = 16, activation = 'relu', name='scratch_dense_2', kernel_regularizer=regularizers.l2(0.001))) #more options to avoid over-fitting
    ### Droput to avoid further over-fitting
    # model.add(Dropout(0.5))
### Adding the output layer
    # model.add(Dense(units = 128, activation = 'relu', name='scratch_dense_3'))#, kernel_regularizer=eegularizers.l2(0.001))) #more options to avoid over-fitting
    #model.add(Dropout(0.5))


    #model.add(Dense(units = 128, activation = 'relu', name='scratch_dense_4'))#, kernel_regularizer=regularizers.l2(0.001))) #more options to avoid over-fitting
    #model.add(Dropout(0.5))


    # model.add(Dense(units = 8, activation = 'relu', name='scratch_dense_5'))#, kernel_regularizer=regularizers.l2(0.001))) #more options to avoid over-fitting
    # model.add(Dropout(0.2))


    model.add(Dense(units = 1, name='scratch_dense_6'))#, kernel_regularizer=regularizers.l2(0.001)))

### Optimizer 
    adam=optimizers.Adam(lr=0.0003,beta_1=0.9, beta_2=0.999, amsgrad=False)

### Compiling the NN
    model.compile(optimizer = adam, loss = 'mean_squared_error',metrics=['mean_squared_error', rmse, r_square] )

### Early stopping
    # earlystopping=EarlyStopping(monitor="mean_squared_error", patience=20, verbose=1, mode='auto')

### Fitting the model to the train set
    result = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size = 32, epochs = 5000)# , callbacks=[earlystopping])
    # X_train = PolynomialFeatures(degree=3).fit_transform(X_train)
    # X_test = PolynomialFeatures(degree=3).fit_transform(X_test)
    # lin = Ridge(alpha=10.0).fit(X_train, Y_train)
    #reg = xgb.XGBRegressor(tree_method="gpu_hist").fit(X_train, Y_train)

#-----------------------------------------------------------------------------
#   Predictions of the Trained Nueral Network Model
#-----------------------------------------------------------------------------

### Test-set prediction
    #y_pred = lin.predict(X_test)
    y_pred = model.predict(X_test)
### train-set prediction
    # y_pred_train = lin.predict(X_train)
    y_pred_train = model.predict(X_train)



##-----------------------------------------------------------------------------
##  Plot Predictions and Learning Curves 
##-----------------------------------------------------------------------------

### Test-set Prediction
    plt.plot(Y_test, color = 'blue', label = 'Real data')
    plt.plot(y_pred, color = 'red', label = 'Predicted data')
    plt.title('Prediction')
    plt.legend()
    # plt.show()
### Predicted VS Observed 
    plt.scatter(Y_test, y_pred, color='red', label= 'Predicted data')
    plt.plot(Y_test, Y_test, color='blue', linewidth=2,label = 'y=x')
    plt.xlabel('observed')
    plt.ylabel('predicted')
    # plt.show()
### Learning curve for RMSE  
    plt.plot(result.history['rmse'])
    plt.plot(result.history['val_rmse'])
    plt.title('rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
### Learning curve for MSE
    plt.plot(result.history['mean_squared_error'])
    plt.plot(result.history['val_mean_squared_error'])
    plt.title('loss function')
    plt.ylabel('mean squared error')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

#-----------------------------------------------------------------------------
#   Save the Trained Nueral Network Model
#-----------------------------------------------------------------------------

# Save model weights
#model.save_weights('Y12_weights.h5')


##-----------------------------------------------------------------------------
##  statistical Summary
##-----------------------------------------------------------------------------


# Diameter conversion




    y_pred=np.array(y_pred[:,0])
    y_pred_train=np.array(y_pred_train[:,0])

    # Ori_train = np.array(Ori_train[:,0])
    # Ori_test= np.array(Ori_test[:,0])

    d_pred_test=Ori_test*y_pred
    d_pred_train=Ori_train*y_pred_train


    maes_train.append(sklearn.metrics.mean_absolute_error(D_train,d_pred_train))
    rmses_train.append(math.sqrt(sklearn.metrics.mean_squared_error(D_train,d_pred_train)))
    r2s_train.append(sklearn.metrics.r2_score(D_train,d_pred_train))
    mapes_train.append(mean_absolute_percentage_error2(D_train,d_pred_train))

    maes_test.append(sklearn.metrics.mean_absolute_error(D_test,d_pred_test))
#print("Mean squared error (MSE) for test-set:       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))
    rmses_test.append(math.sqrt(sklearn.metrics.mean_squared_error(D_test,d_pred_test)))
    r2s_test.append(sklearn.metrics.r2_score(D_test,d_pred_test))
    mapes_test.append(mean_absolute_percentage_error2(D_test,d_pred_test))

    print("\n")
    print("Size Mean absolute error (MAE) for train-set:      %f" % sklearn.metrics.mean_absolute_error(D_train,d_pred_train))
#print("Mean squared error (MSE) for test-set:       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))
    print("Size Root mean squared error (RMSE) for train-set: %f" % math.sqrt(sklearn.metrics.mean_squared_error(D_train,d_pred_train)))
    print("Size R square (R^2) for test-set:                 %f" % sklearn.metrics.r2_score(D_train,d_pred_train))
    print("Size Mean absolute percentage error train-set:   %f" % mean_absolute_percentage_error2(D_train,d_pred_train))



    print("\n")
    print("Size Mean absolute error (MAE) for test-set:      %f" % sklearn.metrics.mean_absolute_error(D_test,d_pred_test))
#print("Mean squared error (MSE) for test-set:       %f" % sklearn.metrics.mean_squared_error(Y_test,y_pred))
    print("Size Root mean squared error (RMSE) for test-set: %f" % math.sqrt(sklearn.metrics.mean_squared_error(D_test,d_pred_test)))
    print("Size R square (R^2) for test-set:                 %f" % sklearn.metrics.r2_score(D_test,d_pred_test))
    print("Size Mean absolute percentage error test-set:   %f" % mean_absolute_percentage_error2(D_test,d_pred_test))



### Predicted VS Observed 
    plt.plot(D_train, D_train, color='black', linewidth=1,label = 'y=x')
    plt.scatter(D_train, d_pred_train, label= 'Predicted data train')
    plt.scatter(D_test, d_pred_test, label= 'Predicted data test')
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.show()





#rate prediciton and conservation of mass

    Z_train=np.array(Z_train)
    Z_test=np.array(Z_test)


    Rate = np.array(Z_train[:,0])
    Rate2 = np.array(Z_test[:,0])
     
    Rate=np.array(Rate)
    Rate2=np.array(Rate2)

    Q_in_train = np.array(Z_train[:,1])

    Q_in_train = np.array(Q_in_train)

    Q_in_train_metric = (Q_in_train/3600)*(10**-9)



    Q_in_test = np.array(Z_test[:,1])
    Q_in_test = np.array(Q_in_test)


    Q_in_test_metric = (Q_in_test/3600)*(10**-9)

    d_pred_train=np.array(d_pred_train)
    d_pred_test=np.array(d_pred_test)


    pred_drop_volume_train = ((d_pred_train**3)*3.143/6)*(10**-18)
    pred_drop_volume_test=((d_pred_test**3)*3.143/6)*(10**-18)

    pred_rate_train= np.divide(Q_in_train_metric,pred_drop_volume_train)





#pred_rate_train=Q_in_train_metric/pred_drop_volume_train


    pred_rate_test=Q_in_test_metric/pred_drop_volume_test



    plt.plot(Rate,pred_rate_train,marker="o",linestyle="")

    plt.plot(Rate2,pred_rate_test,marker="o",linestyle="")
    x_axis = np.array((10, 13000))
# plt.plot(x_axis, m*x_axis + b, linewidth=0.3, c='grey')  # line of best fit
    plt.plot(x_axis, x_axis, linewidth=0.5, c='black')  # one to one line
    plt.show()







    maer_train.append(sklearn.metrics.mean_absolute_error(Rate, pred_rate_train))
    rmser_train.append(math.sqrt(sklearn.metrics.mean_squared_error(Rate, pred_rate_train)))
    r2r_train.append(sklearn.metrics.r2_score(Rate, pred_rate_train))
    maper_train.append(mean_absolute_percentage_error2(Rate, pred_rate_train))

    print("\n")

    maer_test.append(sklearn.metrics.mean_absolute_error(Rate2, pred_rate_test))
    rmser_test.append(math.sqrt(sklearn.metrics.mean_squared_error(Rate2, pred_rate_test)))
    r2r_test.append(sklearn.metrics.r2_score(Rate2, pred_rate_test))
    maper_test.append(mean_absolute_percentage_error2(Rate2, pred_rate_test))


    print("rate train MAE:      %f" % sklearn.metrics.mean_absolute_error(Rate, pred_rate_train))
    print("rate train  RMSE:      %f" % math.sqrt(sklearn.metrics.mean_squared_error(Rate, pred_rate_train)))
    print("rate train R square (R^2):                 %f" % sklearn.metrics.r2_score(Rate, pred_rate_train))
    print("rate train MAPE:   %f" % mean_absolute_percentage_error2(Rate, pred_rate_train))

    print("\n")

    print("rate test MAE:      %f" % sklearn.metrics.mean_absolute_error(Rate2, pred_rate_test))
    print("rate test  RMSE:      %f" % math.sqrt(sklearn.metrics.mean_squared_error(Rate2, pred_rate_test)))
    print("rate test R square (R^2):                 %f" % sklearn.metrics.r2_score(Rate2, pred_rate_test))
    print("rate test MAPE:   %f" % mean_absolute_percentage_error2(Rate2, pred_rate_test))







#generalzibility to literature



    loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/FF1_lit.xlsx")

### Read data
    wb = pd.read_excel(loc ,engine='openpyxl')

    wb.loc[:,['Orifice width (um)','Flow rate ratio','New_ca_number' ]]
  #  X_lit =  wb.loc[:,['Orifice width (um)','Flow rate ratio','New_ca_number' ]]

    X_lit = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

    Y_lit = wb.loc[:, 'Observed droplet diameter (um)']

    Ori_lit=wb.loc[:, 'Hyd_d']

    X_lit =np.array(X_lit)
    Y_lit =np.array(Y_lit)
    Ori_lit =np.array(Ori_lit)

    X_lit=scaler.transform(X_lit)
    # X_lit = PolynomialFeatures(degree=3).fit_transform(X_lit)
    # y_lit_pred = lin.predict(X_lit)
    y_lit_pred = model.predict(X_lit)
    y_lit_pred=np.array(y_lit_pred[:,0])


    D_lit_pred=Ori_lit*y_lit_pred



    # Y_lit=np.array(Y_lit[:,0])






    print("\n")
    mael.append(sklearn.metrics.mean_absolute_error(Y_lit, D_lit_pred))
    rmsel.append(math.sqrt(sklearn.metrics.mean_squared_error(Y_lit, D_lit_pred)))
    r2l.append(sklearn.metrics.r2_score(Y_lit, D_lit_pred))
    mapel.append(mean_absolute_percentage_error2(Y_lit, D_lit_pred))
    print("Mean absolute error (MAE) for lit:      %f" % sklearn.metrics.mean_absolute_error(Y_lit, D_lit_pred))
#print("Mean squared error (MSE) for train-set:       %f" % sklearn.metrics.mean_squared_error(Y_train, y_pred_train))
    print("Root mean squared error (RMSE) for lit: %f" % math.sqrt(sklearn.metrics.mean_squared_error(Y_lit, D_lit_pred)))
    print("R square (R^2) for lit:                 %f" % sklearn.metrics.r2_score(Y_lit, D_lit_pred))
    print("Mean absolute percentage error lit:   %f" % mean_absolute_percentage_error2(Y_lit, D_lit_pred))


    plt.scatter(Y_lit, D_lit_pred, color='red', label= 'Predicted data')
    plt.plot(Y_lit, Y_lit, color='blue', linewidth=2,label = 'y=x')
    plt.show()




    loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/FF1_RPMI_Dropception.xlsx")

### Read data
    wb = pd.read_excel(loc ,engine='openpyxl')

    wb.loc[:,['Orifice width (um)','Flow rate ratio','New_ca_number' ]]
  #  X_lit =  wb.loc[:,['Orifice width (um)','Flow rate ratio','New_ca_number' ]]

    X_lit = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

    Y_lit = wb.loc[:, 'Observed droplet diameter (um)']

    Ori_lit=wb.loc[:, 'Hyd_d']

    X_lit =np.array(X_lit)
    Y_lit =np.array(Y_lit)
    Ori_lit =np.array(Ori_lit)

    X_lit=scaler.transform(X_lit)
    # X_lit = PolynomialFeatures(degree=3).fit_transform(X_lit)
    # y_lit_pred = lin.predict(X_lit)
    y_lit_pred = model.predict(X_lit)
    y_lit_pred=np.array(y_lit_pred[:,0])


    D_lit_pred=Ori_lit*y_lit_pred



    # Y_lit=np.array(Y_lit[:,0])






    print("\n")
    mael.append(sklearn.metrics.mean_absolute_error(Y_lit, D_lit_pred))
    rmsel.append(math.sqrt(sklearn.metrics.mean_squared_error(Y_lit, D_lit_pred)))
    r2l.append(sklearn.metrics.r2_score(Y_lit, D_lit_pred))
    mapel.append(mean_absolute_percentage_error2(Y_lit, D_lit_pred))
    print("Mean absolute error (MAE) for RPMI:      %f" % sklearn.metrics.mean_absolute_error(Y_lit, D_lit_pred))
#print("Mean squared error (MSE) for train-set:       %f" % sklearn.metrics.mean_squared_error(Y_train, y_pred_train))
    print("Root mean squared error (RMSE) for RPMI: %f" % math.sqrt(sklearn.metrics.mean_squared_error(Y_lit, D_lit_pred)))
    print("R square (R^2) for lit:                 %f" % sklearn.metrics.r2_score(Y_lit, D_lit_pred))
    print("Mean absolute percentage error RPMI:   %f" % mean_absolute_percentage_error2(Y_lit, D_lit_pred))


    plt.scatter(Y_lit, D_lit_pred, color='red', label= 'Predicted data')
    plt.plot(Y_lit, Y_lit, color='blue', linewidth=2,label = 'y=x')
    plt.show()



print(f'Size Mean absolute error (MAE) for train-set: {np.mean(maes_train):.2f} +- {1.96 * np.std(maes_train) / np.sqrt(len(maes_train)):.2f}')
print(f'Size Root mean squared error (RMSE) for train-set: {np.mean(rmses_train):.2f} +- {1.96 * np.std(rmses_train) / np.sqrt(len(rmses_train)):.2f}')
print(f'Size R square (R^2) for train-set: {np.mean(r2s_train):.2f} +- {1.96 * np.std(r2s_train) / np.sqrt(len(r2s_train)):.2f}')
print(f'Size Mean absolute percentage error for train-set: {np.mean(mapes_train):.2f} +- {1.96 * np.std(mapes_train) / np.sqrt(len(mapes_train)):.2f}')
print('\n')

print(f'Size Mean absolute error (MAE) for test-set: {np.mean(maes_test):.2f} +- {1.96 * np.std(maes_test) / np.sqrt(len(maes_test)):.2f}')
print(f'Size Root mean squared error (RMSE) for test-set: {np.mean(rmses_test):.2f} +- {1.96 * np.std(rmses_test) / np.sqrt(len(rmses_test)):.2f}')
print(f'Size R square (R^2) for test-set: {np.mean(r2s_test):.2f} +- {1.96 * np.std(r2s_test) / np.sqrt(len(r2s_test)):.2f}')
print(f'Size Mean absolute percentage error for test-set: {np.mean(mapes_test):.2f} +- {1.96 * np.std(mapes_test) / np.sqrt(len(mapes_test)):.2f}')
print('\n')

print(f'rate Mean absolute error (MAE) for train-set: {np.mean(maer_train):.2f} +- {1.96 * np.std(maer_train) / np.sqrt(len(maer_train)):.2f}')
print(f'rate Root mean squared error (RMSE) for train-set: {np.mean(rmser_train):.2f} +- {1.96 * np.std(rmser_train) / np.sqrt(len(rmser_train)):.2f}')
print(f'rate R square (R^2) for train-set: {np.mean(r2r_train):.2f} +- {1.96 * np.std(r2r_train) / np.sqrt(len(r2r_train)):.2f}')
print(f'rate Mean absolute percentage error for train-set: {np.mean(maper_train):.2f} +- {1.96 * np.std(maper_train) / np.sqrt(len(maper_train)):.2f}')
print('\n')

print(f'rate Mean absolute error (MAE) for test-set: {np.mean(maer_test):.2f} +- {1.96 * np.std(maer_test) / np.sqrt(len(maer_test)):.2f}')
print(f'rate Root mean squared error (RMSE) for test-set: {np.mean(rmser_test):.2f} +- {1.96 * np.std(rmser_test) / np.sqrt(len(rmser_test)):.2f}')
print(f'rate R square (R^2) for test-set: {np.mean(r2r_test):.2f} +- {1.96 * np.std(r2r_test) / np.sqrt(len(r2r_test)):.2f}')
print(f'rate Mean absolute percentage error for test-set: {np.mean(maper_test):.2f} +- {1.96 * np.std(maper_test) / np.sqrt(len(maper_test)):.2f}')
print('\n')

print(f'rate Mean absolute error (MAE) for lit: {np.mean(mael):.2f} +- {1.96 * np.std(mael) / np.sqrt(len(mael)):.2f}')
print(f'rate Root mean squared error (RMSE) for lit: {np.mean(rmsel):.2f} +- {1.96 * np.std(rmsel) / np.sqrt(len(rmsel)):.2f}')
print(f'rate R square (R^2) for lit: {np.mean(r2l):.2f} +- {1.96 * np.std(r2l) / np.sqrt(len(r2l)):.2f}')
print(f'rate Mean absolute percentage error for lit: {np.mean(mapel):.2f} +- {1.96 * np.std(mapel) / np.sqrt(len(mapel)):.2f}')






#Stability prediction 

#FF1

loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/NewFF1_Ali_dropception_capillary_reduced7_pruned15.xlsx")

### Read data
wb = pd.read_excel(loc ,engine='openpyxl')

X_inner =   wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

   # X_lit = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio']]

Y_inner = wb.loc[:, 'Observed droplet diameter (um)']

Ori_inner=wb.loc[:, 'Hyd_d']
X_inner =np.array(X_inner)
Y_inner =np.array(Y_inner)
Ori_inner =np.array(Ori_inner)
X_inner=scaler.transform(X_inner)
y_inner_pred = model.predict(X_inner) 
y_inner_pred = model.predict(X_inner) 
y_inner_pred=np.array(y_inner_pred[:,0])
D_inner_pred=np.multiply(Ori_inner, y_inner_pred)



plt.scatter(Y_inner, D_inner_pred, color='red', label= 'Predicted data')
plt.plot(Y_inner, Y_inner, color='blue', linewidth=2,label = 'y=x')
plt.show()




Rate_inner = wb.loc[:, 'Observed generation rate (Hz)']
Rate_inner=np.array(Rate_inner)
Q_in_inner = wb.loc[:, 'Qin']
Q_in_inner_metric = (Q_in_inner/3600)*(10**-9)
pred_drop_volume_inner = ((D_inner_pred**3)*3.143/6)*(10**-18)
pred_rate_inner= np.divide(Q_in_inner_metric,pred_drop_volume_inner)


plt.scatter(Rate_inner, pred_rate_inner, color='red', label= 'Predicted data')
plt.plot(Rate_inner, Rate_inner, color='blue', linewidth=2,label = 'y=x')
plt.show()

print("Mean absolute percentage error Rate FF1:   %f" % mean_absolute_percentage_error2(Rate_inner, pred_rate_inner))




#FF2

loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/NewFF2_Ali_dropception_capillary_reduced7_pruned15.xlsx")

### Read data
wb = pd.read_excel(loc ,engine='openpyxl')

X_outer =   wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

   # X_lit = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio']]

Y_outer = wb.loc[:, 'Observed droplet diameter (um)']
Rate_outer = wb.loc[:, 'Observed generation rate (Hz)']

Ori_outer=wb.loc[:, 'Hyd_d']
X_outer =np.array(X_outer)
Y_outer =np.array(Y_outer)
Ori_outer =np.array(Ori_outer)
X_outer=scaler.transform(X_outer)
y_outer_pred = model.predict(X_outer) 
y_outer_pred=np.array(y_outer_pred[:,0])

D_outer_pred=Ori_outer*y_outer_pred

plt.scatter(Y_outer, D_outer_pred, color='red', label= 'Predicted data')
plt.plot(Y_outer, Y_outer, color='blue', linewidth=2,label = 'y=x')
plt.show()




Rate_outer = wb.loc[:, 'Observed generation rate (Hz)']
Rate_outer=np.array(Rate_outer)
Q_in_outer = wb.loc[:, 'Qin']
Q_in_outer_metric = (Q_in_outer/3600)*(10**-9)
pred_drop_volume_outer = ((D_outer_pred**3)*3.143/6)*(10**-18)
pred_rate_outer= np.divide(Q_in_outer_metric,pred_drop_volume_outer)

plt.plot(Rate_outer, Rate_outer, color='blue', linewidth=2,label = 'y=x')
plt.scatter(Rate_outer, pred_rate_outer, color='red', label= 'Predicted data')
plt.show()

print("Mean absolute percentage error Rate FF2:   %f" % mean_absolute_percentage_error2(Rate_outer, pred_rate_outer))


Truth_rate_dif_perc=100*(Rate_inner - Rate_outer)/(Rate_inner)

Pred_rate_dif_perc=100*(pred_rate_inner - pred_rate_outer)/(pred_rate_inner)

plt.scatter(Truth_rate_dif_perc, Pred_rate_dif_perc, color='red', label= 'Predicted data')

plt.show()










#instability prediction



#FF1
loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/NewFF1_instability.xlsx")

### Read data
wb = pd.read_excel(loc ,engine='openpyxl')

X_ins1 =   wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

   # X_lit = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio']]

Y_ins1 = wb.loc[:, 'Observed droplet diameter (um)']

Ori_ins1=wb.loc[:, 'Hyd_d']
X_ins1 =np.array(X_ins1)
Y_ins1 =np.array(Y_ins1)
Ori_ins1 =np.array(Ori_ins1)
X_ins1=scaler.transform(X_ins1)
y_ins1_pred = model.predict(X_ins1)
y_ins1_pred=np.array(y_ins1_pred[:,0])
 
D_ins1_pred=Ori_ins1*y_ins1_pred


#plt.scatter(Y_ins1, D_ins1_pred, color='red', label= 'Predicted data')
#plt.plot(Y_ins1, Y_ins1, color='blue', linewidth=2,label = 'y=x')
#plt.show()


Rate_ins1 = wb.loc[:, 'Observed generation rate (Hz)']
Rate_ins1=np.array(Rate_ins1)
Q_in_ins1 = wb.loc[:, 'Qin']
Q_in_ins1_metric = (Q_in_ins1/3600)*(10**-9)
pred_drop_volume_ins1 = ((D_ins1_pred**3)*3.143/6)*(10**-18)
pred_rate_ins1= np.divide(Q_in_ins1_metric,pred_drop_volume_ins1)



#FF2
loc = ("/Users/alilashkaripour/Desktop/Fordyce lab/Dropception modeling/Data/New corrected data pruned 15 per/NewFF2_instability.xlsx")

### Read data
wb = pd.read_excel(loc ,engine='openpyxl')


X_ins2 =   wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio','viscosity ratio']]

   # X_lit = wb.loc[:,['Orifice width (um)','Aspect ratio','Flow rate ratio','New_ca_number','Normalized oil inlet','Normalized water inlet','Expansion ratio']]

Y_ins2 = wb.loc[:, 'Observed droplet diameter (um)']

Ori_ins2=wb.loc[:, 'Hyd_d']
X_ins2 =np.array(X_ins2)
Y_ins2 =np.array(Y_ins2)
Ori_ins2 =np.array(Ori_ins2)
X_ins2=scaler.transform(X_ins2)
y_ins2_pred = model.predict(X_ins2) 
y_ins2_pred=np.array(y_ins2_pred[:,0])

D_ins2_pred=Ori_ins2*y_ins2_pred


#plt.scatter(Y_ins2, D_ins2_pred, color='red', label= 'Predicted data')
#plt.plot(Y_ins2, Y_ins2, color='blue', linewidth=2,label = 'y=x')
#plt.show()


Rate_ins2 = wb.loc[:, 'Observed generation rate (Hz)']
Rate_ins2=np.array(Rate_ins2)
Q_in_ins2 = wb.loc[:, 'Qin']
Q_in_ins2_metric = (Q_in_ins2/3600)*(10**-9)
pred_drop_volume_ins2 = ((D_ins2_pred**3)*3.143/6)*(10**-18)
pred_rate_ins2= np.divide(Q_in_ins2_metric,pred_drop_volume_ins2)




case_ins = wb.loc[:, 'Case']

rate_diff_truth=100 * (Rate_ins1 - Rate_ins2) / (Rate_ins1)
plt.scatter(rate_diff_truth, case_ins, color='red', label= 'Predicted data')
plt.show()

rate_diff_pred = 100 * (pred_rate_ins1 - pred_rate_ins2)/ (pred_rate_ins1)
plt.scatter(rate_diff_pred, case_ins, color='red', label= 'Predicted data')

