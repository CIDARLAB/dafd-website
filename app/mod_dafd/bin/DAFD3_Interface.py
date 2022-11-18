""" Interface class for DAFD"""
from app.mod_dafd.core_logic.ForwardModel3 import ForwardModel3
from app.mod_dafd.core_logic.InterModel3 import InterModel3
from app.mod_dafd.core_logic.InterModel3_DE import InterModel3_DE

from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
import numpy as np

class DAFD3_Interface:
	"""A class that provides an interface for DAFD"""

	def __init__(self):
		self.it_se = InterModel3()
		self.it_de = InterModel3_DE()
		self.fw = ForwardModel3()

		self.MH = ModelHelper3.get_instance() # type: ModelHelper

		self.input_headers = self.MH.input_headers
		self.output_headers = self.MH.output_headers

	def runInterpSE(self, desired_vals, constraints, fluid_properties):
		results = self.it_se.interpolate(desired_vals,constraints, fluid_properties)
		return results

	def runInterpDE(self, inner_features, outer_features, desired_vals, fluid_properties):
		inner_results, outer_results = self.it_de.interpolate(inner_features, outer_features, desired_vals, fluid_properties)
		return inner_results, outer_results

	def runForward(self, features, fluid_properties, model="xgb"):
		# features is a dictionary containing the name of each feature as the key and the feature value as the value
		raw_features = np.array([features[x] for x in self.input_headers])
		results = self.fw.predict_size_rate(raw_features, fluid_properties, prediction=model)

		return results

