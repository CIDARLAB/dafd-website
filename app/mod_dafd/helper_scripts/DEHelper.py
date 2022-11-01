import os
import csv
import sklearn
import numpy as np
from sklearn import model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler
from app.mod_dafd.metrics_study.metric_utils import make_sweep_range
import itertools

class DEHelper:
	"""
	This class handles data retrieval, partitioning, and normalization.
	Singleton
	"""
	instance = None				# Singleton
	MH = None
	fluid_properties = None
	input_headers = ['orifice_width', 'aspect_ratio', 'flow_rate_ratio', 'capillary_number', 'normalized_oil_inlet',
					'normalized_water_inlet', 'expansion_ratio', 'viscosity_ratio']	# Feature names (orifice size, aspect ratio, etc...)
	output_headers = ["droplet_size", "generation_rate"] # Output names (droplet size, generation rate)
	inner_flow_range = (50,650)
	oil_flow_range = (200,1200)
	outer_flow_range = (2500,7000)

	def __init__(self, MH, fluid_properties):
		if DEHelper.instance is None:
			self.get_data()
			DEHelper.instance = self
		self.MH = MH
		self.fluid_properties = fluid_properties

	@staticmethod
	def get_instance():
		if DEHelper.instance is None:
			DEHelper()
		return DEHelper.instance

	def generate_grid(self, features, disp_flow_range, cont_flow_range, grid_size=25):
		# generate list of flow_dicts
		disp_flows = make_sweep_range(disp_flow_range, grid_size)
		cont_flows = make_sweep_range(cont_flow_range, grid_size)
		flows = itertools.product(disp_flows, cont_flows)
		flows = [{"dispersed_flow_rate": f[0], "continuous_flow_rate":f[1]} for f in flows]
		return flows

	def generate_inner_grid(self, features):
		flows = self.generate_grid(features, self.inner_flow_range, self.oil_flow_range)
		feature_list = []
		for f in flows:
			feature_copy = features.copy()
			feature_copy["flow_rate_ratio"], feature_copy["capillary_number"] = self.normalize_flow(features, f, outer=False)
			feature_list.append(feature_copy)
		return feature_list

	def generate_outer_grid(self, features):
		flows = self.generate_grid(features, self.oil_flow_range, self.outer_flow_range)
		feature_list = []
		for f in flows:
			feature_copy = features.copy()
			feature_copy["flow_rate_ratio"], feature_copy["capillary_number"] = self.normalize_flow(features, f, outer=True)
			feature_list.append(feature_copy)
		return feature_list

	def normalize_flow(self, design, flow_rates, outer = True):
		frr = flow_rates["continuous_flow_rate"] / flow_rates["dispersed_flow_rate"]
		if outer:
			ca_num = self.fluid_properties["outer_aq_viscosity"] * flow_rates["continuous_flow_rate"] / (
						design["orifice_width"] ** 2 * design["aspect_ratio"]) * (1 / 3.6)
			ca_num = ca_num / self.fluid_properties["outer_surface_tension"]
		else:
			ca_num = self.fluid_properties["oil_viscosity"] * flow_rates["continuous_flow_rate"] / (
						design["orifice_width"] ** 2 * design["aspect_ratio"]) * (1 / 3.6)
			ca_num = ca_num / self.fluid_properties["inner_surface_tension"]
		return frr, ca_num