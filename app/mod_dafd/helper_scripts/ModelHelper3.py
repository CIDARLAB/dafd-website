import os
import csv
import sklearn
import numpy as np
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler


class ModelHelper3:
	"""
	This class handles data retrieval, partitioning, and normalization.
	Singleton
	"""

	RESOURCE_PATH = "experimental_data/DAFD3_data.csv"		# Experimental data location
	NUM_OUTPUTS = 2													# Droplet Generation Rate + Droplet Size

	instance = None				# Singleton
	input_headers = ['orifice_width', 'aspect_ratio', 'flow_rate_ratio', 'capillary_number', 'normalized_oil_inlet',
					'normalized_water_inlet', 'expansion_ratio', 'viscosity_ratio']		# Feature names (orifice size, aspect ratio, etc...)
	output_headers = []			# Output names (droplet size, generation rate)
	all_dat = []				# Raw experimental data
	train_data_size = 0			# Number of data points in the training set
	train_features_dat = []		# Normalized and reduced features from all_dat
	train_labels_dat = {}		# Normalized and reduced labels from all_dat

	ranges_dict = {} 			# Dictionary of ranges for each ranges variable
	ranges_dict_normalized = {} 			# Dictionary of ranges for each ranges variable
	transform_dict = {}			# Dictionary of sklearn transform objects for normalization
	scaler = None

	def __init__(self):
		if ModelHelper3.instance is None:
			self.get_data()
			ModelHelper3.instance = self

	@staticmethod
	def get_instance():
		if ModelHelper3.instance is None:
			ModelHelper3()
		return ModelHelper3.instance

	def resource_path(self, relative_path):
		""" Get absolute path to resource, works for dev and for PyInstaller """
		#try:
		#	# PyInstaller creates a temp folder and stores path in _MEIPASS
		#	base_path = sys._MEIPASS
		#except Exception:
		#	base_path = os.path.abspath(".")
		#
		#return os.path.join(base_path, relative_path)
		return os.path.dirname(os.path.abspath(__file__)) + "/" + relative_path


	def get_data(self):
		self.all_dat = pd.read_csv(self.resource_path(self.RESOURCE_PATH))
		self.features = np.array(self.all_dat.loc[:, self.input_headers])

		self.labels = np.array(self.all_dat.loc[:, 'normalized_hyd_size'])  # make sure to update Ori parameter to be the same parameter for normaliztion if Y is Norm Hyd size, Ori should be Hydraulic diameter; if Y is Norm size Ori should be orifice
		self.observed_diameter = np.array(self.all_dat.loc[:, 'observed_size'])
		self.observed_rate = np.array(self.all_dat.loc[:, ['observed_rate', 'q_in']])

		self.Ori = np.array(self.all_dat.loc[:, 'hyd_d'])  # swap out to orifice when normalizing by orifice width


	def train_test_split(self):
		###train-test split
		validation_size = 0.20
		X_train, X_test, Y_train, Y_test, Ori_train, Ori_test, D_train, D_test, Z_train, Z_test = model_selection.train_test_split(
			self.features, self.labels, self.Ori, self.observed_diameter, self.observed_rate, test_size=validation_size)  # Regime 1 Output 2

		###data scaling
		self.scaler = StandardScaler().fit(X_train)
		self.X_train = self.scaler.transform(X_train)
		self.X_test = self.scaler.transform(X_test)

		self.X_train = np.array(X_train)
		self.Y_train = np.array(Y_train)
		self.X_test = np.array(X_test)
		self.Y_test = np.array(Y_test)

		self.Ori_train = np.array(Ori_train)
		self.Ori_test = np.array(Ori_test)
		self.D_train = np.array(D_train)
		self.D_test = np.array(D_test)
		self.Z_train = np.array(Z_train)
		self.Z_test = np.array(Z_test)


	def normalize(self, X):
		"""Return min max normalization of a variable
		Args:
			X: Value to be normalized

		Returns 0-1 normalization of value with 0 being the min and 1 being the max
		"""
		if self.scaler is None:
			self.train_test_split()
		return self.scaler.transform(X.reshape(1,-1))


	def denormalize(self, X, return_dict = False):
		"""Return actual of a value of a normalized variable
		Args:
			X: Value to be corrected

		Returns actual value of given 0-1 normalized value
		"""
		if self.scaler is None:
			self.train_test_split()
		X_denormalized = self.scaler.inverse_transform(X)
		if return_dict:
			X_denormalized = self.arr_to_dict(X_denormalized)
		return X_denormalized

	def arr_to_dict(self, X):
		return {key: X[i] for i,key in enumerate(self.input_headers)}

	def calculate_formulaic_relations(self, design_inputs):

		"""
			Calculate water flow rate, oil flow rate, and inferred droplet size off the design inputs from DAFD forward model
		"""
		# Get relevant design params
		orifice_size = design_inputs["orifice_width"]
		aspect_ratio = design_inputs["aspect_ratio"]
		normalized_water_inlet = design_inputs["normalized_water_inlet"]
		normalized_oil_inlet = design_inputs["normalized_oil_inlet"]
		flow_rate_ratio = design_inputs["flow_rate_ratio"]
		capillary_number = design_inputs["capillary_number"]
		normalized_diameter = design_inputs["normalized_diameter"]
		surface_tension = design_inputs["surface_tension"]
		oil_viscosity = design_inputs["oil_viscosity"]

		# Calculate oil flow rate
		channel_height = orifice_size * aspect_ratio
		water_inlet_width = orifice_size * normalized_water_inlet
		oil_inlet = orifice_size * normalized_oil_inlet
		#TODO: NEED TO GET NEW FLOW RATE FROM CAP NUMBER, its giving me the wrong values
		oil_flow_rate = capillary_number * orifice_size**2 * aspect_ratio * surface_tension / oil_viscosity * (10**6/60**2)

		# Calculate water flow rate
		water_flow_rate = oil_flow_rate / flow_rate_ratio

		# Calculate droplet diameter
		hydraulic_diameter = (2*aspect_ratio*orifice_size)/(1+aspect_ratio)
		droplet_diameter = normalized_diameter*hydraulic_diameter

		# Calculate generation rate
		droplet_volume_uL = 4/3 * np.pi * (droplet_diameter/2)**3 * 10E-9
		generation_rate = droplet_volume_uL/water_flow_rate/3600
		return oil_flow_rate, water_flow_rate, droplet_diameter, generation_rate
