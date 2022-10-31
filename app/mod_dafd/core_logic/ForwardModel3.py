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
			norm_features = self.MH.normalize_set(features)
		return np.mean([self.predict_nn(norm_features), self.predict_xgb(norm_features)])*features[0]

	def predict_nn(self, features):
		return self.regressor_nn.predict(features)

	def predict_xgb(self, features):
		return self.regressor_xgb.predict(features)

	def predict_size_rate(self, features, fluid_properties, normalized = False, as_dict = True):
		input_dict = {self.MH.input_headers[i]: features[i] for i in range(len(features))}
		input_dict["normalized_diameter"] = self.predict(features, normalized=normalized)
		input_dict.update(fluid_properties)
		_,_,droplet_size, generation_rate = self.MH.calculate_formulaic_relations(input_dict)
		if as_dict:
			return {"droplet_size": droplet_size, "generation_rate": generation_rate}
		else:
			return droplet_size, generation_rate