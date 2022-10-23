from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
from app.mod_dafd.models.forward_models.DAFD3_models import NeuralNetModel_DAFD3, XGBoost_DAFD3
import numpy as np

class ForwardModel3:
	"""
	Bundles both DAFD3 models together.
	This is meant to be a kind of end interface.
	Plug in features, get predicted outputs. Simple!

	Works by predicting droplet size with both XG boost and neural networks, and then averaging them.
	Generation rate is predicted with conservation of mass.
	"""

	regressor_nn = None
	regressor_xgb = None

	def __init__(self):
		self.MH = ModelHelper3.get_instance() # type: ModelHelper
		self.regressor_nn = NeuralNetModel_DAFD3()
		self.regressor_xgb = XGBoost_DAFD3()

	def predict(self, features, normalized = False):
		# regime is an optional parameter that tells the prediction to override the regime prediction
		ret_dict = {}
		if normalized:
			norm_features = features
		else:
			norm_features = self.MH.normalize(features)
		return np.mean([self.predict_nn(norm_features), self.predict_xgb(norm_features)])*features[0]

	def predict_nn(self, features):
		return self.regressor_nn.predict(features)

	def predict_xgb(self, features):
		return self.regressor_xgb.predict(features)
