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
		if normalized:
			norm_features = features
		else:
			norm_features = self.MH.normalize_set(features)
		return np.mean([self.predict_nn(norm_features, normalized=True), self.predict_xgb(norm_features, normalized=True)])

	def predict_nn(self, features, normalized = False):
		if normalized:
			norm_features = features
		else:
			norm_features = self.MH.normalize_set(features)
		return self.regressor_nn.predict(norm_features[:-1]) #Taking out viscosity ratio here as an input

	def predict_xgb(self, features, normalized = False):
		if normalized:
			norm_features = features
		else:
			norm_features = self.MH.normalize_set(features)
		return self.regressor_xgb.predict(norm_features)


	def predict_size_rate(self, features, fluid_properties, inner=True, normalized = False, as_dict = True, prediction = ""):
		input_dict = {self.MH.input_headers[i]: features[i] for i in range(len(features))}
		if prediction == "xgb":
			input_dict["normalized_diameter"] = self.predict_xgb(features, normalized=normalized) #TODO: figure out why xgb isn't changing with flow rate ratio....
		elif prediction == "nn":
			input_dict["normalized_diameter"] = self.predict_nn(features, normalized=normalized)
		else:
			input_dict["normalized_diameter"] = self.predict(features, normalized=normalized)
		input_dict.update(fluid_properties)
		_,_,droplet_size, generation_rate = self.MH.calculate_formulaic_relations(input_dict, inner=inner)
		if as_dict:
			return {"droplet_size": droplet_size, "generation_rate": generation_rate}
		else:
			return droplet_size, generation_rate