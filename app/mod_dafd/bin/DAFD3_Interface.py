""" Interface class for DAFD"""
from app.mod_dafd.core_logic.ForwardModel3 import ForwardModel3
from app.mod_dafd.core_logic.InterModel3 import InterModel3_se

from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
import numpy as np

class DAFD3_Interface:
	"""A class that provides an interface for DAFD"""

	def __init__(self):
		self.it_se = InterModel3_se()
		self.fw = ForwardModel3()

		self.MH = ModelHelper3.get_instance() # type: ModelHelper

		self.input_headers = self.MH.input_headers
		self.output_headers = self.MH.output_headers

	def runInterpSE(self, desired_vals, constraints, fluid_properties):
		results = self.it_se.interpolate(desired_vals,constraints, fluid_properties)
		return results

	def runInterpDE(self, desired_vals, constraints):
		results = self.it.interpolate(desired_vals, constraints)
		return results

	def runForward(self, features, fluid_properties):
		# features is a dictionary containing the name of each feature as the key and the feature value as the value
		raw_features = np.array([features[x] for x in self.input_headers])
		results = self.fw.predict_size_rate(raw_features, fluid_properties)

		return results

