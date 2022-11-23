"""The interpolation model that DAFD runs on"""

from tqdm import tqdm
from scipy.interpolate import Rbf
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.optimize import minimize
import random
import numpy
import itertools
import csv
import sys
import os
from app.mod_dafd.core_logic.ForwardModel3 import ForwardModel3
from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
from app.mod_dafd.helper_scripts.DEHelper import DEHelper
import pandas as pd

import tensorflow as tf

from matplotlib import pyplot as plt

def resource_path(relative_path):
	""" Get absolute path to resource, works for dev and for PyInstaller """
	try:
		# PyInstaller creates a temp folder and stores path in _MEIPASS
		base_path = sys._MEIPASS
	except Exception:
		base_path = os.path.abspath(".")

	return os.path.join(base_path, relative_path)

class InterModel3_DE:
	"""
	This class handles interpolation over our forward models to make the reverse predictions for double emulsions
	"""

	def __init__(self):
		"""Make and save the interpolation models"""
		self.MH = ModelHelper3.get_instance() # type: ModelHelper
		self.fwd_model = ForwardModel3()
		self.fluid_properties = None
		self.DH = None

	def get_closest_point(self, desired_vals, constraints={}, max_drop_exp_error=-1, skip_list = []):
		"""Return closest real data point to our desired values that is within the given constraints
		Used to find a good starting point for our solution
		THIS IS FUNDAMENTALLY DIFFERENT FROM THE NEAREST DATA POINT FORWARD MODEL!
			Nearest data point forward model - Find the outputs of the closest data point to the prediction
			This method - Find the data point closest to the desired outputs

		We will try to find the point closest to the center of our constraints that is close to the target answer

		ALL INPUTS ARE NORMALIZED!

		By itself, this class really isn't all that bad at performing DAFD's main functionality: reverse model prediction
			Therefore, this class should be the baseline level of accuracy for DAFD.
		"""

		closest_point = {}
		min_val = float("inf")
		match_index = -1
		for i in range(self.MH.data_size):
			if i in skip_list:
				continue

			if max_drop_exp_error != -1 and "droplet_size" in desired_vals:
				exp_error = abs(self.MH.denormalize(desired_vals["droplet_size"],"droplet_size") - self.MH.all_dat[i]["droplet_size"])
				if exp_error > max_drop_exp_error:
					continue

			feat_point = self.MH.features[i]
			output_point = self.MH.outputs[i]

			nval = sum([abs(self.MH.normalize( self.MH.all_dat[i][x], x) - desired_vals[x]) for x in desired_vals])
			if "droplet_size" in desired_vals:
				nval += abs(self.MH.normalize(output_point["droplet_size"],"droplet_size") - desired_vals["droplet_size"])

				denorm_feat_list = self.MH.denormalize_set(feat_point)
				denorm_feat = {x:denorm_feat_list[i] for i,x in enumerate(self.MH.input_headers)}
				denorm_feat["generation_rate"] = output_point["generation_rate"]

			if "generation_rate" in desired_vals:
				nval += abs(self.MH.normalize(output_point["generation_rate"],"generation_rate") - desired_vals["generation_rate"])

			for j in range(len(self.MH.input_headers)):
				if self.MH.input_headers[j] in 	constraints:
					cname = self.MH.input_headers[j]
					if feat_point[j] < constraints[cname][0] or feat_point[j] > constraints[cname][0]:
						nval += 1000
						nval += abs(feat_point[j] - (constraints[cname][0] + constraints[cname][1])/2.0)

			if nval < min_val:
				closest_point = feat_point
				min_val = nval
				match_index = i

		return closest_point, match_index


	def callback_func(self, x):
		"""Returns how far each solution mapped on the model deviates from the desired value
		Used in our minimization function
		"""
		prediction = self.fwd_model.predict_size_rate(x, self.fluid_properties, normalized=True, as_dict=True)
		#merrors = [abs(self.MH.normalize(prediction[head], head) - self.norm_desired_vals_global_adjusted[head]) for head in self.norm_desired_vals_global_adjusted]
		val_dict = {self.MH.input_headers[i]:self.MH.denormalize(val,self.MH.input_headers[i]) for i,val in enumerate(x)}
		val_dict["generation_rate"] = prediction["generation_rate"]
		print(prediction["droplet_size"])
		print(prediction["generation_rate"])
		merrors = [abs(self.MH.normalize(prediction[head], head) - self.norm_desired_vals_global[head]) for head in self.norm_desired_vals_global]
		all_errors = sum(merrors)
		print(all_errors)
		print()

		with open("InterResults.csv","a") as f:
			f.write(",".join(map(str,self.MH.denormalize_set(x))) + "," + str(prediction['generation_rate']) +
					"," + str(prediction['droplet_size']) + "," + str(all_errors) + "\n")

	def correct_by_constraints(self,values,constraints):
		"""Sets values to be within constraints (can be normalized or not, as long as values match constraints)"""
		for i,head in enumerate(self.MH.input_headers):
			if head in constraints:
				if values[i] < constraints[head][0]:
					values[i] = constraints[head][0]
				elif values[i] > constraints[head][1]:
					values[i] = constraints[head][1]


	def predict_sweep(self, feature_sweep, inner = False):
		results = []
		fluid_properties = self.fluid_properties.copy()
		if inner:
			fluid_properties["surface_tension"] = fluid_properties["inner_surface_tension"]
		else:
			try:
				fluid_properties["surface_tension"] = fluid_properties["inner_surface_tension"]
			except:
				fluid_properties["surface_tension"] = fluid_properties["_surface_tension"]
		for feature_set in feature_sweep:
			fwd_results = self.fwd_model.predict_size_rate([feature_set[x] for x in self.MH.input_headers],
														   fluid_properties, as_dict=True)
			feature_set.update(fwd_results)
			results.append(feature_set)
		return results

	def optimize(self, inner, outer, k=5):
		pairs = []
		for i, pt in inner.iterrows():
			# ID all outer points within 15% difference in generation rate
			FF1_flow = pt.continuous_flow_rate + pt.dispersed_flow_rate
			adjacent = outer.loc[outer.dispersed_flow_rate == FF1_flow, :] # TODO: need to change this to be the sum of cont + dispersed
			adjacent = adjacent.loc[self.DH.pct_difference(pt["generation_rate"], adjacent.generation_rate) < 15,:] #TODO: add in if there isn't anything less than 15, increase to 30 and repeat
			# Pick outer point with minimum size difference from outer point
			adjacent = adjacent.sort_values("size_err")
			if len(adjacent > 0):
				pairs.append((pt, adjacent.iloc[0,:]))
			else:
				pairs.append(None)
		# Finally, choose top-k solutions for TOTAL size err (inner_err% + outer_err%)
		total_errs = []
		for i, pair in enumerate(pairs):
			if pair is None:
				total_errs.append(np.inf)
			else:
				total_errs.append(pair[0].size_err + pair[1].size_err)
		idx = np.argpartition(total_errs, k)[:k]  # Indices not sorted
		idx = idx[np.argsort(np.array(total_errs)[idx])]
		return np.array(pairs)[idx], np.array(total_errs)[idx]

	def interpolate(self,inner_features, outer_features, desired_values, fluid_properties):
		"""Return an input set within the given constraints that produces the output set
		The core part of DAFD
		Args:
				inner_features:
				outer_features:
				desired_values:
				fluid_properties:
		"""
		#TODO: need to add flow rates in before normalizing set
		self.fluid_properties = fluid_properties
		self.DH = DEHelper(self.MH, fluid_properties)


		#TODO: Make this a loop if have a list for orifice width
		if type(inner_features["orifice_width"]) is list:
			inner_results = []
			outer_results = []
			errors = []
			for i in range(len(inner_features["orifice_width"])):
				in_feat = inner_features.copy()
				out_feat = outer_features.copy()
				in_feat["orifice_width"] = in_feat["orifice_width"][i]
				out_feat["orifice_width"] = out_feat["orifice_width"][i]
				inner_generator_sweep = self.DH.generate_inner_grid(in_feat)
				inner_generator_results = pd.DataFrame(self.predict_sweep(inner_generator_sweep, inner = True))

				outer_generator_sweep = self.DH.generate_outer_grid(out_feat)
				outer_generator_results = pd.DataFrame(self.predict_sweep(outer_generator_sweep, inner=False))

				### ALGORITHM ###
				# INNER: find and rank by distance between sweep sizes and desired sizes
				inner_generator_results.loc[:, "size_err"] = self.DH.pct_difference(desired_values["inner_droplet_size"], inner_generator_results.loc[:,"droplet_size"])
				# OUTER: find and rank by distance between sweep sizes and desired sizes
				outer_generator_results.loc[:, "size_err"] = self.DH.pct_difference(desired_values["outer_droplet_size"], outer_generator_results.loc[:,"droplet_size"])
				# Find optimal pairs for DE design automation
				results, err = self.optimize(inner_generator_results, outer_generator_results)
				#TODO: make downloadable csv for all of the different results (top-k)
				errors.append(err[0])
				inner_results.append(results[0][0])
				outer_results.append(results[0][1])
			top_idx = np.argmin(err)
			inner_result = inner_results[top_idx]
			outer_result = outer_results[top_idx]

		else:
			inner_generator_sweep = self.DH.generate_inner_grid(inner_features)
			inner_generator_results = pd.DataFrame(self.predict_sweep(inner_generator_sweep, inner = True))

			outer_generator_sweep = self.DH.generate_outer_grid(outer_features)
			outer_generator_results = pd.DataFrame(self.predict_sweep(outer_generator_sweep, inner=False))

			### ALGORITHM ###
			# INNER: find and rank by distance between sweep sizes and desired sizes
			inner_generator_results.loc[:, "size_err"] = self.DH.pct_difference(desired_values["inner_droplet_size"], inner_generator_results.loc[:,"droplet_size"])
			# OUTER: find and rank by distance between sweep sizes and desired sizes
			outer_generator_results.loc[:, "size_err"] = self.DH.pct_difference(desired_values["outer_droplet_size"], outer_generator_results.loc[:,"droplet_size"])
			# Find optimal pairs for DE design automation
			results, err = self.optimize(inner_generator_results, outer_generator_results)
			#TODO: make downloadable csv for all of the different results (top-k)
			inner_result = results[0][0]
			outer_result = results[0][1]
		return inner_result, outer_result

