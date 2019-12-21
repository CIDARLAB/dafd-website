from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

from keras import backend as K

from config import no_tuning_params
'''
TUNABLE_MODELS = [
			('Classifier', KerasClassifier(build_fn=createClassifier,
                                        loss_func='binary_crossentropy', opt_func='adam', 
                                        act_hidden='relu',
                                        act_output='sigmoid')),
			('Regressor', KerasRegressor(build_fn=createRegressor, 
                                        loss_func='mean_squared_error', opt_func='adam', 
                                        act_hidden='relu',
                                        act_output='linear'))
		]

NO_TUNABLE_MODELS = [
			('Classifier', KerasClassifier(build_fn=createClassifier,
                                        loss_func='binary_crossentropy', opt_func='adam', 
                                        batch_size=no_tuning_params['model_00']['batch_size'],
                                        epochs=no_tuning_params['model_00']['epochs'],
                                        num_hidden=no_tuning_params['model_00']['num_hidden'],
                                        node_hidden=no_tuning_params['model_00']['node_hidden'],
                                        act_hidden='relu', act_output='sigmoid')),
			('Regressor', KerasRegressor(build_fn=createRegressor, 
                                        loss_func='mean_squared_error', opt_func='adam', 
                                        batch_size=no_tuning_params['model_11']['batch_size'],
                                        epochs=no_tuning_params['model_11']['epochs'],
                                        num_hidden=no_tuning_params['model_11']['num_hidden'],
                                        node_hidden=no_tuning_params['model_11']['node_hidden'],
                                        act_hidden='relu', act_output='linear'))
		]
'''
def createClassifier(loss_func, opt_func, num_hidden, node_hidden, act_hidden, act_output):

    model = Sequential()

    for dummy_index in range(num_hidden):
        model.add(Dense(node_hidden, activation=act_hidden))

    model.add(Dense(1, activation=act_output))
    model.compile(loss=loss_func, optimizer=opt_func, metrics=['accuracy', f1])

    return model

def createRegressor(loss_func, opt_func, num_hidden, node_hidden, act_hidden, act_output):

    model = Sequential()

    for dummy_index in range(num_hidden):
        model.add(Dense(node_hidden, activation=act_hidden))

    model.add(Dense(1))
    model.compile(loss=loss_func, optimizer=opt_func, metrics=['mean_squared_error', r_square])

    return model

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	prec = true_positives / (predicted_positives + K.epsilon())
	return prec

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	rec = true_positives / (possible_positives + K.epsilon())
	return rec

def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec )/(prec + rec + K.epsilon()))

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))
