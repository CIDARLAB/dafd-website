from app.mod_dafd.models.forward_models.SVRModel import SVRModel
from app.mod_dafd.models.forward_models.NearestDataPointModel import NearestDataPointModel
from app.mod_dafd.models.forward_models.RidgeRegressor import RidgeRegressor
from app.mod_dafd.models.forward_models.LassoRegressor import LassoRegressor
from app.mod_dafd.models.forward_models.RandomForestModel import RandomForestModel
from app.mod_dafd.models.forward_models.LinearModel import LinearModel
from app.mod_dafd.models.forward_models.NeuralNetModel import NeuralNetModel
from app.mod_dafd.models.forward_models.NeuralNetModel_keras import NeuralNetModel_keras
from app.mod_dafd.models.forward_models.NeuralNetModel_rate1 import NeuralNetModel_rate1
from app.mod_dafd.models.forward_models.NeuralNetModel_rate2 import NeuralNetModel_rate2
from app.mod_dafd.models.forward_models.NeuralNetModel_size1 import NeuralNetModel_size1
from app.mod_dafd.models.forward_models.NeuralNetModel_size2 import NeuralNetModel_size2
from app.mod_dafd.helper_scripts.ModelHelper import ModelHelper
import numpy as np
import sklearn.metrics

load_model = True	# Load the file from disk

class Regressor:
	"""
	Small adapter class that handles training and usage of the underlying models
	"""

	regression_model = None

	def __init__(self, output_name, regime):
		self.MH = ModelHelper.get_instance() # type: ModelHelper

		regime_indices = self.MH.regime_indices[regime]
		regime_feature_data = [self.MH.train_features_dat[x] for x in regime_indices]
		regime_label_data = [self.MH.train_labels_dat[output_name][x] for x in regime_indices]

		print("Regression model " + output_name + str(regime))
		if output_name == "generation_rate":
			if regime == 1:
				self.regression_model = NeuralNetModel_rate1()
			elif regime == 2:
				self.regression_model = NeuralNetModel_rate2()
		elif output_name == "droplet_size":
			if regime == 1:
				self.regression_model = NeuralNetModel_size1()
			elif regime == 2:
				self.regression_model = NeuralNetModel_size2()

		if load_model:
			print("Loading Regressor")
			self.regression_model.load_model(output_name, regime)
		else:
			print("Training Regressor")
			print("All data points: " + str(len(self.MH.train_features_dat)))
			print("Train points: " + str(len(regime_indices)))
			self.regression_model.train_model(output_name, regime, regime_feature_data, regime_label_data)

		train_features = np.stack(regime_feature_data)
		train_labels = np.stack(regime_label_data)
		print("R square (R^2) for Train:                 %f" % sklearn.metrics.r2_score(train_labels, self.regression_model.regression_model.predict(train_features)))
		print()


	def predict(self,features):
		return self.regression_model.predict(features)
