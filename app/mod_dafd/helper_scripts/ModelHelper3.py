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
	output_headers = ["droplet_size", "generation_rate"]			# Output names (droplet size, generation rate)
	all_dat_df = []				# Raw experimental data
	train_data_size = 0			# Number of data points in the training set
	train_features_dat = []		# Normalized and reduced features from all_dat_df
	train_labels_dat = {}		# Normalized and reduced labels from all_dat_df
	data_size = 0

	ranges_dict = {} 			# Dictionary of ranges for each ranges variable
	ranges_dict_normalized = {} 			# Dictionary of ranges for each ranges variable
	transform_dict = {}			# Dictionary of sklearn transform objects for normalization
	scaler = None

	def __init__(self):
		self.Ori = None
		self.observed_rate = None
		self.observed_diameter = None
		self.labels = None
		self.features = None
		self.outputs = None
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
		self.all_dat_df = pd.read_csv(self.resource_path(self.RESOURCE_PATH))
		self.all_dat = self.all_dat_df.to_dict(orient="records")
		self.data_size = len(self.all_dat)
		self.features = np.array(self.all_dat_df.loc[:, self.input_headers])

		self.labels = np.array(self.all_dat_df.loc[:, 'normalized_hyd_size'])  # make sure to update Ori parameter to be the same parameter for normaliztion if Y is Norm Hyd size, Ori should be Hydraulic diameter; if Y is Norm size Ori should be orifice
		self.observed_diameter = np.array(self.all_dat_df.loc[:, 'droplet_size'])
		self.observed_rate = np.array(self.all_dat_df.loc[:, ['generation_rate', 'q_in']])
		self.outputs = self.all_dat_df.loc[:, ['droplet_size', 'generation_rate']].to_dict(orient="records")

		self.Ori = np.array(self.all_dat_df.loc[:, 'hyd_d'])  # swap out to orifice when normalizing by orifice width
		self.scaler = StandardScaler()
		self.scaler.fit(self.features)

		for i, head in enumerate(self.all_dat_df.columns):
			values = np.array(self.all_dat_df.loc[:,head])
			self.transform_dict[head] = StandardScaler()
			self.transform_dict[head].fit(values.reshape(-1,1))
			self.ranges_dict[head] = (min(values), max(values))
			self.ranges_dict_normalized[head] = (self.transform_dict[head].transform([[min(values)]])[0][0],
									  			 self.transform_dict[head].transform([[max(values)]])[0][0])
		self.normalized_features = self.scaler.transform(self.features)

	def train_test_split(self, validation_size=0.20):
		###train-test split
		X_train, X_test, Y_train, Y_test, Ori_train, Ori_test, D_train, D_test, Z_train, Z_test = model_selection.train_test_split(
			self.features, self.labels, self.Ori, self.observed_diameter, self.observed_rate, test_size=validation_size)  # Regime 1 Output 2

		###data scaling
		self.X_train = self.scaler.transform(X_train)
		self.X_test = self.scaler.transform(X_test)
		self.train_data_size

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


	def normalize_set(self, values, as_dict = False):
		""" Normalizes a set of features
		Args:
			values: list of features to be normalized (same order as input_headers)

		Returns list of normalized features in the same order as input_headers
		"""
		if as_dict:
			ret = {}
			for k in values.keys():
				ret[k] = self.MH.normalize(values[k], k)

		else:
			ret = []
			for i, header in enumerate(self.input_headers):
				ret.append(self.normalize(values[i], header))
		return ret

	def denormalize_set(self, values, as_dict = False):
		""" Denormalizes a set of features
		Args:
			values: list of features to be denormalized (same order as input_headers)

		Returns list of denormalized features in the same order as input_headers
		"""
		if as_dict:
			ret = {}
			for k in values.keys():
				ret[k] = self.MH.denormalize(values[k], k)

		else:
			ret = []
			for i, header in enumerate(self.input_headers):
				ret.append(self.denormalize(values[i], header))
		return ret

	def normalize(self, value, inType):
		"""Return min max normalization of a variable
		Args:
			value: Value to be normalized
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized

		Returns 0-1 normalization of value with 0 being the min and 1 being the max
		"""
		return self.transform_dict[inType].transform([[value]])[0][0]


	def denormalize(self, value, inType):
		"""Return actual of a value of a normalized variable
		Args:
			value: Value to be corrected
			inType: The type of the value (orifice size, aspect ratio, etc) to be normalized

		Returns actual value of given 0-1 normalized value
		"""
		return self.transform_dict[inType].inverse_transform([[value]])[0][0]

	def arr_to_dict(self, X):
		return {key: X[i] for i,key in enumerate(self.input_headers)}

	def calculate_formulaic_relations(self, design_inputs, inner=True, flow_only=False):

		"""
			Calculate water flow rate, oil flow rate, and inferred droplet size off the design inputs from DAFD forward model
		"""
		# Get relevant design params
		orifice_size = design_inputs["orifice_width"]
		aspect_ratio = design_inputs["aspect_ratio"]
		flow_rate_ratio = design_inputs["flow_rate_ratio"]
		capillary_number = design_inputs["capillary_number"]
		surface_tension = design_inputs["surface_tension"]
		if inner:
			oil_viscosity = design_inputs["oil_viscosity"]
		else:
			oil_viscosity = design_inputs["outer_aq_viscosity"]
		# Calculate oil flow rate
		oil_flow_rate = capillary_number * orifice_size**2 * aspect_ratio * surface_tension / oil_viscosity * 60**2 * 10**-3

		# Calculate water flow rate
		water_flow_rate = oil_flow_rate / flow_rate_ratio
		if flow_only:
			return oil_flow_rate, water_flow_rate
		else:
			normalized_diameter = design_inputs["normalized_diameter"]
			# Calculate droplet diameter
			hydraulic_diameter = (2*aspect_ratio*orifice_size)/(1+aspect_ratio)
			droplet_diameter = normalized_diameter*hydraulic_diameter

			# Calculate generation rate
			droplet_volume_uL = 1/6 * np.pi * (droplet_diameter)**3 * 10**-9
			generation_rate = water_flow_rate / droplet_volume_uL / 60**2
			return oil_flow_rate, water_flow_rate, droplet_diameter, generation_rate
