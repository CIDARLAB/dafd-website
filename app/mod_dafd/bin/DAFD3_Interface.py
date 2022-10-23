""" Interface class for DAFD"""
from app.mod_dafd.core_logic.ForwardModel3 import ForwardModel3
from app.mod_dafd.core_logic.InterModel3 import InterModel3

from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
import numpy as np

class DAFD3_Interface:
	"""A class that provides an interface for DAFD"""

	def __init__(self):
		self.it = InterModel3()
		self.fw = ForwardModel3()

		self.MH = ModelHelper3.get_instance() # type: ModelHelper

		self.input_headers = self.MH.input_headers
		self.output_headers = self.MH.output_headers

	def runInterp(self, desired_vals, constraints):
		results = self.it.interpolate(desired_vals,constraints)
		return results

	def runForward(self, features):
		# features is a dictionary containing the name of each feature as the key and the feature value as the value
		raw_features = np.array([features[x] for x in self.input_headers])
		results = {}
		results["droplet_size"] = self.fw.predict(raw_features)
		design_params = {}
		for feature in features:
			design_params[feature] = features[feature]
		design_params["droplet_size"] = results["droplet_size"]
		results["oil_rate"], results["water_rate"], results["generation_rate"] = self.MH.calculate_formulaic_relations(design_params)
		return results

