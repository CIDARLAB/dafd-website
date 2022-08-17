import pandas as pd
import numpy as np
import os
import json

from datetime import datetime, timedelta

from config import CONFIG, HYPERPARAMS
#from app.mod_NN.models import TUNABLE_MODELS, NO_TUNABLE_MODELS

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from app.mod_NN.models import createClassifier, createRegressor

from keras import backend as K
from keras.callbacks import EarlyStopping

from app.mod_dafd.DAFD_CMD import runDAFD, runDAFD_2

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
RESOURCES = os.path.join(APP_ROOT, '../resources/inputs/')
METRICS_MAP = {
	'accuracy': 'Accuracy',
	'precision': 'Precision',
	'recall': 'Recall',
	'f1': 'F1 Score',
	'roc_auc': 'ROC-AUC',
	'neg_mean_squared_error': 'MSE',
	'neg_mean_absolute_error': 'MAE',
	'r2': 'R-squared'
}

def validFile(filename):

	return '.' in filename and filename.rsplit('.', 1)[1].lower() in set(['xlsx', 'xls', 'csv'])

def readFile(filename):

	df = pd.read_csv(filename)
	df.rename(columns={'index': 'date'}, inplace=True)

	return df

def getDataType(filename):

	df = readFile(filename)

	val = []
	flag = []
	for col in df.columns:
		if df[col].dtype == np.float64 or df[col].dtype == np.int64:
			val.append('numeric')
			flag.append(0)
		else:
			try:
				df[col] = pd.to_datetime(df[col])
				val.append('datetime')
				flag.append(1)
			except:
				val.append('text')
				flag.append(2)

	#df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val, 'flag': flag})
	df1 = pd.DataFrame({'column': df.columns.tolist(), 'type': val})
	df2 = df.describe().round(2).transpose().reset_index().rename(columns={'index': 'column'})
	unique = df.nunique().reset_index().rename(columns={'index': 'column', 0: 'unique'})
	#nan = df.isnull().sum().reset_index().rename(columns={'index': 'column', 0: 'missing'})
	
	df3 = pd.merge(df1, unique, how='left')
	#df3 = pd.merge(df3, nan, how='left')
	df3 = pd.merge(df3, df2, how='left')
	
	df3.fillna('', inplace=True)

	return df3

def getScore(mode, name, metrics, test, pred, prob, xval, tuned):

	if metrics == 'accuracy':
		value = accuracy_score(test, pred)
	elif metrics == 'precision':
		value = precision_score(test, pred)
	elif metrics == 'recall':
		value = recall_score(test, pred)
	elif metrics == 'f1':
		value = f1_score(test, pred)
	elif prob != None and metrics== 'roc_auc':
		value = roc_auc_score(test, prob)

	elif metrics == 'neg_mean_squared_error':
		value = mean_squared_error(test, pred)
	elif metrics == 'neg_mean_absolute_error':
		value = mean_absolute_error(test, pred)
	elif metrics == 'r2':
		value = roc_auc_score(test, pred)
	
	return ({
		'Mode': [mode.title()],
		'Model Name': [name],
		METRICS_MAP[metrics]: [value],
		'Xval?': [xval],
		'Tuned?': [tuned]
	})

def getScoreCV(mode, name, metrics, cv):

	return ({
		'Mode': [mode.title()],
		'Model Name': [name],
		METRICS_MAP[metrics]: [cv['test_' + metrics].mean()],
		'Xval?': ['Yes'],
		'Tuned?': ['No']
	})

'''
def getClassificationScore(name, score, test, pred, prob):

	acc, prec, rec, f1, roc = None, None, None, None, None

	#for score in scores:
	if score == 'accuracy':
		acc = accuracy_score(test, pred)
	elif score == 'precision':
		prec = precision_score(test, pred)
	elif score == 'recall':
		rec = recall_score(test, pred)
	elif score == 'f1':
		f1 = f1_score(test, pred)
	elif score == 'roc_auc':
		roc = roc_auc_score(test, prob)

	score_dict = {
		'Mode': 'Classification',
		'Model Name': name,
		'Accuracy': acc,
		'Precision': prec,
		'Recall': rec,
		'F-Score': f1,
		'ROC-AUC': roc
	}

	return {k:[v] for k,v in score_dict.items() if v is not None}

def getRegressionScore(name, score, pred, test):

	mae, mse, rmse, r2 = None, None, None, None

	#for score in scores:
	if score == 'mae':
		mae = mean_absolute_error(test, pred)
	elif score == 'mse':
		mse = mean_squared_error(test, pred)
	elif score == 'rmse':
		rmse = np.sqrt(mean_squared_error(test, pred))
	elif score == 'r2':
		r2 = r2_score(test, pred)

	score_dict = {
		'Mode': 'Regression',
		'Model Name': name,
		'Mean Absolute Error': mae,
		'Mean Squared Error': mse,
		'RMSE': rmse,
		'R-squared': r2
	} 
	return {k:[v] for k,v in score_dict.items() if v is not None}
'''

def generateParams(payload):

	params = []
	if payload['normalization'] != 'none':
		if payload['normalization'] == 'minmax':
			scaler = MinMaxScaler()
		elif payload['normalization'] == 'standard':
			scaler = StandardScaler()
		elif payload['normalization'] == 'robust':
			scaler = RobustScaler()
		params.append(('scaler', scaler))

	return params

def makeDataset(df, target):

	X = df.drop(target, axis=1)
	y = df[target]

	return X.values, y.values.ravel()

def runNN(payload, tuning_params):

	if K.backend() == 'tensorflow':
		K.clear_session()

	model = None
	best_config = None

	'''Preparing dataset and pipeline'''
	complete_filename = os.path.join(RESOURCES, payload['filename'])
	df = readFile(complete_filename)

	if payload['target'] in payload['drops']:
		payload['drops'].remove(payload['target'])
	df.drop(payload['drops'], axis=1, inplace=True)

	params = generateParams(payload)
	X, y = makeDataset(df, payload['target'])
	
	if payload['mode'] == 'classification':
		y = y - 1

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=float(payload['holdout']))

	'''Selecting between hyperparameter tuning, cross-validation, or holdout methods'''
	if payload['tuning'] != 'none':
		if payload['mode']=='classification':
			model = KerasClassifier(build_fn=createClassifier, loss_func='binary_crossentropy', opt_func='adam', act_hidden='relu', act_output='sigmoid')
		elif payload['mode']=='regression':
			model = KerasRegressor(build_fn=createRegressor, loss_func='mean_squared_error', opt_func='adam', act_hidden='relu', act_output='linear')
		
		params.append(('mod', model))
		pipeline = Pipeline(params)
		
		if payload['tuning'] == 'grid':
			exe = GridSearchCV(pipeline, tuning_params, cv=int(payload['fold']), n_jobs=-1, verbose=2)
		elif payload['tuning'] == 'random':
			exe = RandomizedSearchCV(pipeline, tuning_params, cv=int(payload['fold']), n_jobs=-1, verbose=2)
		
		start = datetime.now()
		exe.fit(X_train, y_train)
		end = datetime.now()
		print('Total execution time:', str(end-start))

		'''
		top3 = pd.DataFrame(exe.cv_results_)
		top3.sort_values(by='rank_test_score', inplace=True)
		print('******Each fold result******')
		print(top3)
		'''

		best_params = exe.best_params_
		print('Best config:', best_params)

		y_pred = exe.best_estimator_.predict(X_test)
		y_prob = None
		if payload['mode']=='classification':
			y_prob = exe.best_estimator_.predict_proba(X_test)[:, 1]
		
		##subject to change
		'''
		if (payload['mode']=='classification'):
			results = getClassificationScore(payload['model-name'], payload['metrics'], y_test, y_pred, y_prob)
		elif (payload['mode']=='regression'):
			results = getRegressionScore(payload['model-name'], payload['metrics'], y_test, y_pred)
		'''
		results = getScore(payload['mode'], payload['model-name'], payload['metrics'], y_test, y_pred, y_prob, 'Yes', 'Yes')
		
		model_saver = exe.best_estimator_['mod'].model
		best_config = {
            'Batch-size': best_params['mod__batch_size'],
            'Epoch': best_params['mod__epochs'],
            'Number of hidden layers': best_params['mod__num_hidden'],
            'Number of nodes per layer': best_params['mod__node_hidden']
        }
		
		if payload['save-best-config']:
			json_str = json.dumps(exe.best_params_)
			best_config_json = os.path.join(RESOURCES, payload['best-config-file'])
			with open(best_config_json, 'w') as json_file:
				json_file.write(json_str)


	elif payload['tuning'] == 'none' and payload['validation'] == 'holdout':

		es = EarlyStopping(monitor=('loss'), mode='min', verbose=1)
		if payload['mode']=='classification':
			model = KerasClassifier(build_fn=createClassifier,
						loss_func='binary_crossentropy', opt_func='adam', 
						batch_size=tuning_params['batch_size'],
						epochs=tuning_params['epochs'],
						num_hidden=tuning_params['num_hidden'],
						node_hidden=tuning_params['node_hidden'],
						act_hidden='relu', act_output='sigmoid', callbacks=[es])
		elif payload['mode']=='regression':
			model = KerasRegressor(build_fn=createRegressor, 
						loss_func='mean_squared_error', opt_func='adam', 
						batch_size=tuning_params['batch_size'],
						epochs=tuning_params['epochs'],
						num_hidden=tuning_params['num_hidden'],
						node_hidden=tuning_params['node_hidden'],
						act_hidden='relu', act_output='linear', callbacks=[es])

		params.append(('mod', model))
		pipeline = Pipeline(params)

		start = datetime.now()
		pipeline.fit(X_train, y_train)
		end = datetime.now()
		print('Total execution time:', str(end-start))

		y_pred = pipeline.predict(X_test)
		y_prob = None
		if payload['mode']=='classification':
			y_prob = pipeline.predict_proba(X_test)[:, 1]
		
		##subject to change
		'''
		if (payload['mode']=='classification'):
			results = getClassificationScore(payload['model-name'], payload['metrics'], y_test, y_pred, y_prob)
		elif (payload['mode']=='regression'):
			results = getRegressionScore(payload['model-name'], payload['metrics'], y_test, y_pred)
		'''
		results = getScore(payload['mode'], payload['model-name'], payload['metrics'], y_test, y_pred, y_prob, 'No', 'No')

		model_saver = pipeline.named_steps['mod'].model
		best_config = {
            'Batch-size': tuning_params['batch_size'],
            'Epoch': tuning_params['epochs'],
            'Number of hidden layers': tuning_params['num_hidden'],
            'Number of nodes per layer': tuning_params['node_hidden']
        }

	elif payload['tuning'] == 'none' and payload['validation'] == 'crossval':

		if payload['mode']=='classification':
			model = KerasClassifier(build_fn=createClassifier,
						loss_func='binary_crossentropy', opt_func='adam', 
						batch_size=tuning_params['batch_size'],
						epochs=tuning_params['epochs'],
						num_hidden=tuning_params['num_hidden'],
						node_hidden=tuning_params['node_hidden'],
						act_hidden='relu', act_output='sigmoid')
		elif payload['mode']=='regression':
			model = KerasRegressor(build_fn=createRegressor, 
						loss_func='mean_squared_error', opt_func='adam', 
						batch_size=tuning_params['batch_size'],
						epochs=tuning_params['epochs'],
						num_hidden=tuning_params['num_hidden'],
						node_hidden=tuning_params['node_hidden'],
						act_hidden='relu', act_output='linear')

		params.append(('mod', model))
		pipeline = Pipeline(params)

		'''
		if payload['metrics'] == 'mse':
			scoring = ['neg_mean_squared_error']
		elif payload['metrics'] == 'mae':
			scoring = ['neg_mean_absolute_error']
		else:
			scoring = [payload['metrics']]
		'''

		start = datetime.now()
		res = cross_validate(estimator=pipeline, X=X_train, y=y_train, cv=int(payload['fold']), scoring=[payload['metrics']])
		end = datetime.now()
		print('Total execution time:', str(end-start))

		'''
		res_dict = {}
		res_dict['Model Name'] = payload['model-name']
		#for score in [payload['metrics']]:
		key = 'test_' + payload['metrics']
		res_dict[key] = res[key].mean()

		print(res_dict)
		'''
		results = getScoreCV(payload['mode'], payload['model-name'], payload['metrics'], res)

		start = datetime.now()
		pipeline.fit(X_train, y_train)
		end = datetime.now()
		print('Total execution time:', str(end-start))

		model_saver = pipeline.named_steps['mod'].model
		best_config = {
            'Batch-size': tuning_params['batch_size'],
            'Epoch': tuning_params['epochs'],
            'Number of hidden layers': tuning_params['num_hidden'],
            'Number of nodes per layer': tuning_params['node_hidden']
        }
	
	if (payload['save-architecture']):
		architecture = os.path.join(RESOURCES, payload['architecture-file'])
		with open(architecture, 'w') as json_file:
			json_file.write(model_saver.to_json())
		print("Saved architecture to disk")
		
	if (payload['save-weights']):
		weights = os.path.join(RESOURCES, payload['weights-file'])
		model_saver.save_weights(weights)
		print("Saved model to disk")
	
	return results, best_config

def runForward(forward):

	with open("app/mod_dafd/cmd_inputs.txt", "w") as f: 
		f.write("FORWARD\n")
		for key in forward:
			if forward[key] != None:
				f.write(key + '=' + str(forward[key]) + '\n')

	return runDAFD()

def runReverse(constraints, desired_vals):

	with open("app/mod_dafd/cmd_inputs.txt", "w") as f:

		if constraints:
			f.write("CONSTRAINTS\n")
			for key in constraints:
				if constraints[key] != None:
					f.write(key + '=' + str(constraints[key]) + ':' + str(constraints[key]) + '\n')
		if desired_vals:
			f.write("DESIRED_VALS\n")
			for key in desired_vals:
				if desired_vals[key] != None:
					f.write(key + '=' + str(desired_vals[key]) + '\n')

	return runDAFD()


def runReverse_2(constraints, desired_vals, metrics):

	with open("app/mod_dafd/cmd_inputs.txt", "w") as f:
		if metrics:
			f.write("VERSATILITY\n")
			f.write("FLOW_STABILITY\n")
			for key in metrics:
				if metrics[key] is not None:
					f.write(key + '=' + str(metrics[key]) + '\n')

		if constraints:
			f.write("CONSTRAINTS\n")
			for key in constraints:
				if constraints[key] != None:
					f.write(key + '=' + str(constraints[key]) + ':' + str(constraints[key]) + '\n')
		if desired_vals:
			f.write("DESIRED_VALS\n")
			for key in desired_vals:
				if desired_vals[key] != None:
					f.write(key + '=' + str(desired_vals[key]) + '\n')

	return runDAFD_2()