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

class InterModel3:
	"""
	This class handles interpolation over our forward models to make the reverse predictions
	"""

	def __init__(self):
		"""Make and save the interpolation models"""
		self.MH = ModelHelper3.get_instance() # type: ModelHelper
		self.fwd_model = ForwardModel3()
		self.fluid_properties = None
		self.param_bounds = {"orifice_width":(15.0, 175.0), "aspect_ratio": (1.0,3.0), "flow_rate_ratio": (0.69,22.0),
							 "capillary_number":(0.014, 0.5), "normalized_oil_inlet":(1.0,4.0),
							 "normalized_water_inlet":(1.0,4.0), "expansion_ratio":(1.0,6.0)}
		self.param_bounds_list = [self.param_bounds[k] for k in self.param_bounds.keys()]

	def xgb_nn_error(self, features):
		xgb_size, xgb_rate = self.fwd_model.predict_size_rate(features, self.fluid_properties, as_dict=False, prediction="xgb")
#		nn_size, nn_rate = self.fwd_model.predict_size_rate(features, self.fluid_properties, as_dict=False, prediction="nn")
		return 0

		# return np.abs(self.MH.normalize(xgb_size, "droplet_size") - self.MH.normalize(nn_size, "droplet_size"))


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
			feat_dict = {x:feat_point[i] for i,x in enumerate(self.MH.input_headers)}
			output_point = self.MH.outputs[i]

			nval = sum([abs(self.MH.normalize( self.MH.all_dat[i][x], x) - desired_vals[x]) for x in desired_vals])
			if "droplet_size" in desired_vals:
				nval += abs(self.MH.normalize(output_point["droplet_size"],"droplet_size") - desired_vals["droplet_size"])

				denorm_feat_list = self.MH.denormalize_set(feat_point)
				denorm_feat = {x:denorm_feat_list[i] for i,x in enumerate(self.MH.input_headers)}
				denorm_feat["generation_rate"] = output_point["generation_rate"]

				# nval+= self.xgb_nn_error(feat_point)



			if "generation_rate" in desired_vals:
				nval += abs(self.MH.normalize(output_point["generation_rate"],"generation_rate") - desired_vals["generation_rate"])

			if "oil_flow_rate" in self.flow_constraints.keys():
				ca_num = self.fluid_properties["oil_viscosity"] * self.flow_constraints["oil_flow_rate"] / (
							feat_dict["orifice_width"] ** 2 * feat_dict["aspect_ratio"]) * (1 / 3.6)
				ca_num = ca_num/self.fluid_properties["surface_tension"]
				constraints['capillary_number'] = [ca_num, ca_num]

			if "water_flow_rate" in self.flow_constraints.keys():
				oil_flow = (feat_dict["capillary_number"] * feat_dict["orifice_width"]**2 * feat_dict["aspect_ratio"]*3.6  \
							* self.fluid_properties["surface_tension"]) /  (self.fluid_properties["oil_viscosity"])
				q_ratio = oil_flow / self.flow_constraints["water_flow_rate"]
				constraints["flow_rate_ratio"] = [q_ratio, q_ratio]

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

	def model_error(self, x):
		"""Returns how far each solution mapped on the model deviates from the desired value
		Used in our minimization function
		"""
		prediction = self.fwd_model.predict_size_rate(x, self.fluid_properties, normalized=False, as_dict=True)
		val_dict = {self.MH.input_headers[i]:self.MH.denormalize(val,self.MH.input_headers[i]) for i,val in enumerate(x)}
		val_dict["generation_rate"] = prediction["generation_rate"]
		#TODO: Add in area where xgb and nn disagree on the droplet size
		# model_error = self.xgb_nn_error(x)
		merrors = [abs(self.MH.normalize(prediction[head], head) - self.norm_desired_vals_global[head]) for head in self.norm_desired_vals_global]
		return sum(merrors) #+ model_error

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
		val_dict = {}
		for i,head in enumerate(self.MH.input_headers):
			val_dict[head] = values[i]
			if head in constraints:
				if values[i] < constraints[head][0]:
					values[i] = constraints[head][0]
				elif values[i] > constraints[head][1]:
					values[i] = constraints[head][1]
		if "oil_flow_rate" in self.flow_constraints.keys():
			ca_num = self.fluid_properties["oil_viscosity"] * self.flow_constraints["oil_flow_rate"] / (
						val_dict["orifice_width"] ** 2 * val_dict["aspect_ratio"]) * (1 / 3.6)
			ca_num = ca_num/self.fluid_properties["surface_tension"]
			val_dict["capillary_number"] = ca_num
			values[3] = ca_num

		if "water_flow_rate" in self.flow_constraints.keys():
			if "oil_flow_rate" in self.flow_constraints.keys():
				oil_flow = self.flow_constraints["oil_flow_rate"]
			else:
				oil_flow = (val_dict["capillary_number"] * val_dict["orifice_width"]**2 * val_dict["aspect_ratio"]*3.6  \
							* self.fluid_properties["surface_tension"]) /  (self.fluid_properties["oil_viscosity"])
			q_ratio = oil_flow / self.flow_constraints["water_flow_rate"]
			val_dict["flow_rate_ratio"] = q_ratio
			values[2] = q_ratio

	def interpolate(self,desired_val_dict,constraints, fluid_properties):
		"""Return an input set within the given constraints that produces the output set
		The core part of DAFD
		Args:
			desired_val_dict: Dict with output type as the key and desired value as the value
				Just don't include other output type if you just want to interpolate on one

			constraints: Dict with input type as key and acceptable range as the value
				The acceptable range should be a tuple with the min as the first val and the max as the second val
				Again, just leave input types you don't care about blank
		"""
		norm_constraints = {}
		denorm_constraints = {}
		self.flow_constraints = {}

		for cname in constraints:
			if cname in ["oil_flow_rate", "water_flow_rate"]:
				self.flow_constraints[cname] = constraints[cname][0]
			else:
				cons_low = self.MH.normalize(constraints[cname][0], cname)
				cons_high = self.MH.normalize(constraints[cname][1],cname)
				norm_constraints[cname] = (cons_low, cons_high)
				denorm_constraints[cname] = (constraints[cname][0], constraints[cname][1])
		constraints = {k:constraints[k] for k in constraints.keys() if k not in ["oil_flow_rate", "water_flow_rate"]}


		norm_desired_vals = {}
		for lname in desired_val_dict:
			norm_desired_vals[lname] = self.MH.normalize(desired_val_dict[lname], lname)

		self.norm_desired_vals_global = norm_desired_vals
		self.desired_vals_global = desired_val_dict
		self.fluid_properties = fluid_properties


		skip_list = []
		j = 0
		while(True):
			j+=1
			start_pos, closest_index = self.get_closest_point(norm_desired_vals, constraints=denorm_constraints, max_drop_exp_error=5, skip_list=skip_list)
			if closest_index == -1:
				start_pos, closest_index = self.get_closest_point(norm_desired_vals, constraints=denorm_constraints)
				break
			skip_list.append(closest_index)

			closest_point = self.MH.all_dat[closest_index]



			all_dat_labels = self.MH.input_headers +  self.MH.output_headers
			print(",".join(all_dat_labels))
			print("Starting point")
			print(closest_point)
			print([closest_point[x] for x in all_dat_labels])

			should_skip_optim_rate = True
			should_skip_optim_size = True
			should_skip_optim_constraints = True

			for constraint in constraints:
				cons_range = constraints[constraint]
				this_val = self.MH.all_dat[closest_index][constraint]
				if this_val < cons_range[0] or this_val > cons_range[1]:
					should_skip_optim_constraints = False

			if "generation_rate" in desired_val_dict:
				if desired_val_dict["generation_rate"] > 100:
					pred_rate_error = abs(desired_val_dict["generation_rate"] - closest_point["generation_rate"]) / desired_val_dict["generation_rate"]
					exp_rate_error = abs(desired_val_dict["generation_rate"] - self.MH.all_dat[closest_index]["generation_rate"]) / self.MH.all_dat[closest_index]["generation_rate"]
					if pred_rate_error > 0.15 or exp_rate_error > 0.15:
						should_skip_optim_rate = False
				else:
					pred_rate_error = abs(desired_val_dict["generation_rate"] - closest_point["generation_rate"])
					exp_rate_error = abs(desired_val_dict["generation_rate"] - self.MH.all_dat[closest_index]["generation_rate"])
					if pred_rate_error > 15 or exp_rate_error > 15:
						should_skip_optim_rate = False

			if "droplet_size" in desired_val_dict:
				pred_size_error = abs(desired_val_dict["droplet_size"] - closest_point["droplet_size"])
				exp_size_error = abs(desired_val_dict["droplet_size"] - self.MH.all_dat[closest_index]["droplet_size"])
				print(self.MH.all_dat[closest_index])
				pred_point = {x:self.MH.all_dat[closest_index][x] for x in self.MH.all_dat[closest_index]}
				pred_point["generation_rate"] = closest_point["generation_rate"]
				print(self.MH.all_dat[closest_index])
				#TODO: if needed add model error into skip
				if pred_size_error > 10 or exp_size_error > 5:
					should_skip_optim_size = False

			if should_skip_optim_rate and should_skip_optim_size and should_skip_optim_constraints:
				results = {x: self.MH.all_dat[closest_index][x] for x in self.MH.input_headers}
				results["point_source"] = "Experimental"
				print(results)
				return results

		with open("InterResults.csv","w") as f:
			f.write("Experimental outputs:"+str(self.MH.all_dat[closest_index]["generation_rate"])+","+str(self.MH.all_dat[closest_index]["droplet_size"])+"\n")

			if "generation_rate" not in desired_val_dict:
				des_rate = "-1"
			else:
				des_rate = str(desired_val_dict["generation_rate"])

			if "droplet_size" not in desired_val_dict:
				des_size = "-1"
			else:
				des_size = str(desired_val_dict["droplet_size"])

			f.write("Desired outputs:"+des_rate+","+des_size+"\n")
			f.write(",".join(self.MH.input_headers) + ",generation_rate,droplet_size,cost_function\n")

		pos = start_pos

		self.correct_by_constraints(pos,denorm_constraints)

		loss = self.model_error(pos)

		stepsize = [2.5, 0.25, 2, 0.025, 0.25, 0.25, 0.25, 0]
		ftol = 1e-9



		for i in range(5000):
			new_pos = pos
			new_loss = loss
			for index, val in enumerate(pos[:-1]): #do not do viscosity ratio
				copy = [x for x in pos]
				if val + stepsize[index] > self.param_bounds_list[index][1]:
					copy[index] = self.param_bounds_list[index][1]
				else:
					copy[index] = val+stepsize[index]
				self.correct_by_constraints(copy,denorm_constraints)
				error = self.model_error(copy)
				if error < new_loss:
					new_pos = copy
					new_loss = error

				copy = [x for x in pos]
				if val - stepsize[index] < self.param_bounds_list[index][0]:
					copy[index] = self.param_bounds_list[index][0]
				else:
					copy[index] = val-stepsize[index]
				self.correct_by_constraints(copy,denorm_constraints)
				error = self.model_error(copy)
				if error < new_loss:
					new_pos = copy
					new_loss = error

			if loss - new_loss < ftol:
				print(loss)
				print(new_loss)
				break

			pos = new_pos
			loss = new_loss

			#self.callback_func(pos)

		self.last_point = pos

		#Denormalize results
		results = {x: pos[i] for i, x in enumerate(self.MH.input_headers)}
		prediction = self.fwd_model.predict_size_rate([results[x] for x in self.MH.input_headers], self.fluid_properties, as_dict=True)
		print("Final Suggestions")
		print(",".join(self.MH.input_headers) + "," + "desired_size" + "," + "predicted_generation_rate" + "," + "predicted_droplet_size")
		output_string = ",".join([str(results[x]) for x in self.MH.input_headers])
		output_string += "," + str(desired_val_dict["droplet_size"])
		output_string += "," + str(prediction["generation_rate"])
		output_string += "," + str(prediction["droplet_size"])
		print(output_string)
		print("Final Prediction")
		print(prediction)
		results["point_source"] = "Predicted"
		return results

