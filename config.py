REDIS = {
	'broker': 'redis://localhost:6379',
	'backend': 'redis://localhost:6379'
}

CONFIG = {
	'secret_key': '1234567890',
	#'domain': 'http://localhost:5000',
	'domain': 'http://ml.dafdcad.org',
	'path': '../resources/',
	'model_index': 0	#0:regime classification, 1:regime 1-rate regression, 2:regime 1-size regression, 3:regime 2-rate regression, 4:regime 2-size regression
}
						
payload = {
	'filename': 'DAFD_classification.csv',
	#'filename': 'DAFD_regression.csv',
	'mode': 'Classification',
	#'mode': 'Regression',
	'encoding': 'none',
	'missing': 'drop',
	'normalization': 'standard',
	'crossval': None,		#If hyper-param=True, this will be automatically overwritten with True, None for no crossval
	'test-size': 0.2,
	'cv_method': 'stratkfold',
	'dim_red': None,
	'num_of_dim': None,
	'hyper-param': None,	#True for optimizing, None for not optimizing
	'tuning': 'grids',		#At the moment, the only other option that works is 'randoms'
	'cls_metrics': ['accuracy', 'f1'],	#Other metrics include 'precision', 'recall'
	'reg_metrics': ['r2', 'rmse'],		#Other metrics include 'mae', 'mse'
	'drops': ['Experiment', 'Rate', 'Size', 'Monodispersity'],	#for regime classification
	#'drops': ['Experiment', 'regime', 'Size', 'Monodispersity'],	#for rate regression
	#'drops': ['Experiment', 'regime', 'Rate', 'Monodispersity'],	#for size regression
	'targets': ['regime'],	#regime classification
	#'targets': ['Rate'],	#rate regression
	#'targets': ['Size'],	#size regression
	'filter': 'regime',		#this value only matter for regression
	'selected_condition': 2,	#Or 2, this value will not matter for regime classification
	'save-best-config': True,
	'best-config-file': 'best-config-classification.json',
	'save-architecture': True,
	'architecture-file': 'architecture-classification.json',
	'save-weights': True,
	'weights-file': 'weights-classification.h5'
}

no_tuning_params = {
	'model_00': {
		'batch_size': 32,
		'epochs': 100,
		'node_hidden': 16,
		'num_hidden': 8
	},
	'model_11': {
		'batch_size': 32,
		'epochs': 100,
		'node_hidden': 16,
		'num_hidden': 8
	},
	'model_12': {
		'batch_size': 32,
		'epochs': 100,
		'node_hidden': 16,
		'num_hidden': 8
	},
	'model_21': {
		'batch_size': 32,
		'epochs': 100,
		'node_hidden': 16,
		'num_hidden': 8
	},
	'model_22': {
		'batch_size': 32,
		'epochs': 100,
		'node_hidden': 16,
		'num_hidden': 8
	},
}

##### PARAMETER SPACE #####
batch_sizes_00 = [20, 40]
epochs_00 = [50, 100]
node_hidden_00 = [8, 16]
num_hidden_00 = [3, 4]

batch_sizes_11 = [20, 40]
epochs_11 = [50, 100]
node_hidden_11 = [8, 16]
num_hidden_11 = [3, 4]

batch_sizes_12 = [20, 40]
epochs_12 = [50, 100]
node_hidden_12 = [8, 16]
num_hidden_12 = [3, 4]

batch_sizes_21 = [20, 40]
epochs_21 = [50, 100]
node_hidden_21 = [8, 16]
num_hidden_21 = [3, 4]

batch_sizes_22 = [20, 40]
epochs_22 = [50, 10]
node_hidden_22 = [8, 16]
num_hidden_22 = [3, 4]
#############################

grid_00 = {'mod__batch_size': batch_sizes_00,
				'mod__epochs': epochs_00,
				'mod__node_hidden': node_hidden_00,
				'mod__num_hidden': num_hidden_00}
grid_11 = {'mod__batch_size': batch_sizes_11,
				'mod__epochs': epochs_11,
				'mod__node_hidden': node_hidden_11,
				'mod__num_hidden': num_hidden_11}
grid_12 = {'mod__batch_size': batch_sizes_12,
				'mod__epochs': epochs_12,
				'mod__node_hidden': node_hidden_12,
				'mod__num_hidden': num_hidden_12}
grid_21 = {'mod__batch_size': batch_sizes_21,
				'mod__epochs': epochs_21,
				'mod__node_hidden': node_hidden_21,
				'mod__num_hidden': num_hidden_21}
grid_22 = {'mod__batch_size': batch_sizes_22,
				'mod__epochs': epochs_22,
				'mod__node_hidden': node_hidden_22,
				'mod__num_hidden': num_hidden_22}

HYPERPARAMS = [
			('Model-00 (Regime Classification)', grid_00),
			('Model-11 (Regime#1-Rate Regression)', grid_11),
			('Model-12 (Regime#1-Size Regression)', grid_12),
			('Model-21 (Regime#2-Rate Regression)', grid_21),
			('Model-22 (Regime#2-Size Regression)', grid_22)	
		]
