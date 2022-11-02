#!/usr/bin/python3

import sys
import os
from app.mod_dafd.bin.DAFD_Interface import DAFD_Interface
from app.mod_dafd.bin.DAFD3_Interface import DAFD3_Interface
from app.mod_dafd.helper_scripts.MetricHelper import MetricHelper
import pandas as pd
from keras import backend as K

def runDAFD():

	if K.backend() == 'tensorflow':
		K.clear_session()

	di = DAFD_Interface()

	constraints = {}
	desired_vals = {}
	features = {}

	stage = 0
	with open(os.path.dirname(os.path.abspath(__file__)) + "/" + "cmd_inputs.txt","r") as f:
		for line in f:
			line = line.strip()
			if line == "CONSTRAINTS":
				stage=0
				continue
			elif line == "DESIRED_VALS":
				stage=1
				continue
			elif line == "FORWARD":
				stage=2
				continue

			if stage == 0:
				param_name = line.split("=")[0]
				param_pair = line.split("=")[1].split(":")
				if param_name=="regime":
					wanted_constraint=float(param_pair[0])
					constraints[param_name] = wanted_constraint
				else:
					wanted_constraint = (float(param_pair[0]), float(param_pair[1]))
					constraints[param_name] = wanted_constraint

			if stage == 1:
				param_name = line.split("=")[0]
				param_val = float(line.split("=")[1])
				desired_vals[param_name] = param_val

			if stage == 2:
				param_name = line.split("=")[0]
				param_val = float(line.split("=")[1])
				features[param_name] = param_val

	if stage == 2:
		fwd_results = di.runForward(features)

		result_str = "BEGIN:"

		for x in di.MH.get_instance().output_headers:
			result_str += str(fwd_results[x]) + "|"
		result_str += str(fwd_results["regime"]) + "|"
		result_str += str(fwd_results["oil_rate"]) + "|"
		result_str += str(fwd_results["water_rate"]) + "|"
		result_str += str(fwd_results["inferred_droplet_size"]) + "|"
		print(result_str)

	else:
		rev_results = di.runInterp(desired_vals, constraints)
		fwd_results = di.runForward(rev_results)

		print(rev_results)
		print(fwd_results)


		result_str = "BEGIN:"
		for x in di.MH.get_instance().input_headers:
			result_str += str(rev_results[x]) + "|"

		result_str += str(rev_results["point_source"]) + "|"

		for x in di.MH.get_instance().output_headers:
			result_str += str(fwd_results[x]) + "|"
		result_str += str(fwd_results["regime"]) + "|"
		result_str += str(fwd_results["oil_rate"]) + "|"
		result_str += str(fwd_results["water_rate"]) + "|"
		result_str += str(fwd_results["inferred_droplet_size"]) + "|"

		print(result_str)
		
	return result_str

def runDAFD_2():
	if K.backend() == 'tensorflow':
		K.clear_session()


	di = DAFD_Interface()

	constraints = {}
	desired_vals = {}
	features = {}

	stage = -1
	tolerance_test = False
	sort_by = None
	flow_stability = False
	versatility = False
	top_k=3

	with open(os.path.dirname(os.path.abspath(__file__)) + "/" + "cmd_inputs.txt","r") as f:
		for line in f:
			line = line.strip()
			if line == "CONSTRAINTS":
				stage=0
				continue
			elif line == "DESIRED_VALS":
				stage=1
				continue
			elif line == "FORWARD":
				stage=2
				continue
			elif line == "FLOW_STABILITY":
				flow_stability=True
				continue
			elif line == "VERSATILITY":
				versatility=True
				continue
			elif "sort_by" in line:
				sort_by = line.split("=")[1]
				continue
			elif "top_k" in line:
				top_k = int(line.split("=")[1])
				continue

			if stage == 0:
				param_name = line.split("=")[0]
				param_pair = line.split("=")[1].split(":")
				if param_name=="regime":
					wanted_constraint=float(param_pair[0])
					constraints[param_name] = wanted_constraint
				else:
					wanted_constraint = (float(param_pair[0]), float(param_pair[1]))
					constraints[param_name] = wanted_constraint

			if stage == 1:
				param_name = line.split("=")[0]
				param_val = float(line.split("=")[1])
				desired_vals[param_name] = param_val

			if stage == 2:
				param_name = line.split("=")[0]
				param_val = float(line.split("=")[1])
				features[param_name] = param_val

	if flow_stability or versatility:
		if sort_by is None:
			if flow_stability:
				sort_by = "flow_stability"
			else:
				sort_by = "overall_versatility"
		reg_str = ""
		if "versatility" in sort_by:
			try:
				if constraints["regime"] == 1:
					reg_str = "dripping"
				else:
					reg_str = "jetting"
			except:
				reg_str = "all"
			sort_by = reg_str + "_" + sort_by.split("_")[0] + "_" + "score"



	if stage == 2:
		fwd_results = di.runForward(features)
		result_str = "BEGIN:"

		for x in di.MH.get_instance().output_headers:
			result_str += str(fwd_results[x]) + "|"
		result_str += str(fwd_results["regime"]) + "|"
		result_str += str(fwd_results["oil_rate"]) + "|"
		result_str += str(fwd_results["water_rate"]) + "|"
		result_str += str(fwd_results["inferred_droplet_size"]) + "|"
		print(result_str)
		# if flow_stability or versatility:
		# 	results = features.copy()
		# 	results.update(fwd_results)
		# 	if fwd_results["regime"] == 1:
		# 		reg_str = "Dripping"
		# 	else:
		# 		reg_str = "Jetting"
		# 	MetHelper = MetricHelper(results, di=di)
		# 	MetHelper.run_all_flow_stability()
		# 	MetHelper.run_all_versatility()
		# 	results.update(MetHelper.versatility_results)
		# 	results.update({"flow_stability":MetHelper.point_flow_stability})
		# 	report_info = {
		# 		"regime": reg_str,
		# 		"results_df": pd.DataFrame([results]),
		# 		"sort_by": sort_by
		# 	}
		# 	report_info["feature_denormalized"] = MetHelper.features_denormalized
		# 	MetHelper.generate_report(report_info)

	else:
		if flow_stability or versatility:
			results = di.runInterpQM(desired_vals, constraints.copy(), top_k=top_k)
			for i, result in enumerate(results):
				MetHelper = MetricHelper(result, di=di)
				MetHelper.run_all_flow_stability()
				results[i]["flow_stability"] = MetHelper.point_flow_stability
				MetHelper.run_all_versatility()
				results[i].update(MetHelper.versatility_results)
				results[i].update(di.runForward(result))
			results_df = pd.DataFrame(results)

			results_df.sort_values(by=sort_by, ascending=False, inplace=True)
			# report_info = {
			# 	"regime": reg_str,
			# 	"results_df": results_df,
			# 	"sort_by": sort_by
			# }
			# MetHelper = MetricHelper(results_df.to_dict(orient="records")[0], di=di)
			# MetHelper.run_all_flow_stability()
			# MetHelper.run_all_versatility()
			# report_info["feature_denormalized"] = MetHelper.features_denormalized

			rev_results = results_df.to_dict(orient="records")[0]
			fwd_results = di.runForward(rev_results)

			import datetime
			date = datetime.datetime.today().isoformat()[:16]
			size = int(fwd_results["droplet_size"])
			rate = int(rev_results["generation_rate"])
			folder_path = os.path.join(os.getcwd(), "app", "resources")
			for filename in os.listdir(folder_path):
				if "Hz" in filename:
					os.remove(os.path.join(folder_path, filename))
			filepath = f"app/resources/{date}_{size}um_{rate}Hz.csv"
			filepath = filepath.replace(":", "_")
			results_df.to_csv(filepath)
			file_name = f"{date}_{size}um_{rate}Hz.csv"
			file_name = file_name.replace(":", "_")


		else:
			rev_results = di.runInterp(desired_vals, constraints)
			fwd_results = di.runForward(rev_results)

		result_str = "BEGIN:"
		for x in di.MH.get_instance().input_headers:
			result_str += str(rev_results[x]) + "|"

		result_str += str(rev_results["point_source"]) + "|"

		for x in di.MH.get_instance().output_headers:
			result_str += str(fwd_results[x]) + "|"
		result_str += str(fwd_results["regime"]) + "|"
		result_str += str(fwd_results["oil_rate"]) + "|"
		result_str += str(fwd_results["water_rate"]) + "|"
		result_str += str(fwd_results["inferred_droplet_size"]) + "|"

	return result_str, file_name


def runDAFD_3():
	if K.backend() == 'tensorflow':
		K.clear_session()

	di = DAFD3_Interface()

	constraints = {}
	desired_vals = {}
	features = {}
	fluid_properties = {}

	stage = 0
	with open(os.path.dirname(os.path.abspath(__file__)) + "/" + "cmd_inputs.txt", "r") as f:
		for line in f:
			line = line.strip()
			if line == "FLUID_PROPERTIES":
				stage = -1
				continue
			elif line == "CONSTRAINTS":
				stage = 0
				continue
			elif line == "DESIRED_VALS":
				stage = 1
				continue
			elif line == "FORWARD":
				stage = 2
				continue

			if stage == -1:
				param_name = line.split("=")[0]
				param_val = line.split("=")[1]
				fluid_properties[param_name] = float(param_val)
			if stage == 0:
				param_name = line.split("=")[0]
				param_pair = line.split("=")[1].split(":")
				if param_name == "regime":
					wanted_constraint = float(param_pair[0])
					constraints[param_name] = wanted_constraint
				else:
					wanted_constraint = (float(param_pair[0]), float(param_pair[1]))
					constraints[param_name] = wanted_constraint

			if stage == 1:
				param_name = line.split("=")[0]
				param_val = float(line.split("=")[1])
				desired_vals[param_name] = param_val

			if stage == 2:
				param_name = line.split("=")[0]
				param_val = float(line.split("=")[1])
				features[param_name] = param_val

	if stage == 2:
		fwd_results = di.runForward(features, fluid_properties)

		result_str = "BEGIN:"

		for x in fwd_results.keys():
			result_str += str(fwd_results[x]) + "|"
		print(result_str)
		all_params = fwd_results.copy()
	else:
		rev_results = di.runInterpSE(desired_vals, constraints, fluid_properties)
		fwd_results = di.runForward(rev_results, fluid_properties)

		print(rev_results)
		print(fwd_results)

		result_str = "BEGIN:"
		for x in di.MH.get_instance().input_headers:
			result_str += str(rev_results[x]) + "|"

		result_str += str(rev_results["point_source"]) + "|"

		for x in di.MH.get_instance().output_headers:
			result_str += str(fwd_results[x]) + "|"
		all_params = rev_results.copy()
		all_params.update(fwd_results)
		all_params.update(fluid_properties)
		oil_flow_rate, water_flow_rate = di.MH.calculate_formulaic_relations(all_params, flow_only=True)
		all_params["oil_flow_rate"] = oil_flow_rate
		all_params["water_flow_rate"] = water_flow_rate
		result_str += str(oil_flow_rate) + "|"
		result_str += str(water_flow_rate)
		print(result_str)

	return result_str, all_params

def runDAFD_3_DE(inner_features, outer_features, desired_vals, fluid_properties):
	if K.backend() == 'tensorflow':
		K.clear_session()

	di = DAFD3_Interface()
	inner_results, outer_results = di.runInterpDE(inner_features, outer_features, desired_vals, fluid_properties)

	return inner_results, outer_results