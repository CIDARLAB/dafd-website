import os
import csv
import sklearn
import numpy as np
from sklearn import model_selection
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from app.mod_dafd.metrics_study.metric_utils import make_sweep_range
from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
import itertools
from app.mod_dafd.core_logic.ForwardModel3 import ForwardModel3
#import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

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
	# inner_flow_range = (50,650)
	# oil_flow_range = (200,1200)
	# outer_flow_range = (2500,7000)


	#Full solution
	inner_flow_range = (50, 650)
	oil_flow_range = (200, 1200)
	outer_flow_range = (1500, 10000)

	#For solution specific, do +/- 50%


	def __init__(self, fluid_properties):
		if DEHelper.instance is None:
			DEHelper.instance = self
		self.MH = ModelHelper3.get_instance()
		self.fluid_properties = fluid_properties
		self.fwd_model = ForwardModel3()
	@staticmethod
	def get_instance():
		if DEHelper.instance is None:
			DEHelper()
		return DEHelper.instance

	def generate_grid(self, disp_flow_range, cont_flow_range, inner_grid_size=25, outer_grid_size=18):
		# generate list of flow_dicts
		if len(disp_flow_range) == 2:
			disp_flows = make_sweep_range(disp_flow_range, inner_grid_size)
		else:
			disp_flows = disp_flow_range
		cont_flows = make_sweep_range(cont_flow_range, outer_grid_size)
		flows = itertools.product(disp_flows, cont_flows)
		flows = [{"dispersed_flow_rate": f[0], "continuous_flow_rate":f[1]} for f in flows]
		return flows

	def predict_sweep(self,feature_sweep, inner=False):
		results = []
		fprop = self.fluid_properties.copy()
		if inner:
			fprop["surface_tension"] = self.fluid_properties["inner_surface_tension"]
		else:
			try:
				fprop["surface_tension"] = self.fluid_properties["inner_surface_tension"]
			except:
				fprop["surface_tension"] = self.fluid_properties["outer_surface_tension"]
		for feature_set in feature_sweep:
			fwd_results = self.fwd_model.predict_size_rate([feature_set[x] for x in self.MH.input_headers],
													  		fprop, as_dict=True)
			if type(feature_set) is not dict:
				feature_set = feature_set.to_dict()
			feature_set.update(fwd_results)
			results.append(feature_set)
		return results


	def generate_inner_grid(self, features):
		flows = self.generate_grid(self.inner_flow_range, self.oil_flow_range, inner_grid_size=25, outer_grid_size=41)
		self.total_inner_flows = np.unique([np.sum(list(item.values())) for item in flows])
		feature_list = []
		for f in flows:
			feature_copy = features.copy()
			feature_copy["flow_rate_ratio"], feature_copy["capillary_number"] = self.normalize_flow(features, f, outer=False)
			feature_copy["dispersed_flow_rate"] = f["dispersed_flow_rate"]
			feature_copy["continuous_flow_rate"] = f["continuous_flow_rate"]
			feature_list.append(feature_copy)
		return feature_list

	def pct_difference(self, ref, pt):
		return np.abs((ref - pt)/ref)*100

	def generate_outer_grid(self, features):
		flows = self.generate_grid(self.total_inner_flows, self.outer_flow_range, inner_grid_size=25, outer_grid_size=18)
		feature_list = []
		for f in flows:
			feature_copy = features.copy()
			feature_copy["flow_rate_ratio"], feature_copy["capillary_number"] = self.normalize_flow(features, f, outer=True)
			feature_copy["dispersed_flow_rate"] = f["dispersed_flow_rate"]
			feature_copy["continuous_flow_rate"] = f["continuous_flow_rate"]

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

	def plot_stability(self, in_hm, out_hm, stab_mask, outer_flow_rate, rate=False):
		dx = 0.15
		dy = 1
		figsize = plt.figaspect(float(dx * 2) / float(dy * 1))
		fig, axs = plt.subplots(1, 2, facecolor="w", figsize=figsize)
		fig.suptitle('Device stability with outer flow rate of ' + str(outer_flow_rate) + " (\u03BCL/hr)", fontsize=16)
		plt.subplots_adjust(wspace=0.3, bottom=0.2)
		axs[0].set_facecolor("#bebebe")
		axs[1].set_facecolor("#bebebe")

		#TODO: take out if rate no longer needed
		if rate:
			cbar_kw = "Generation Rate (Hz)"
			vmin = in_hm.min().min()
			if out_hm.min().min() < vmin:
				vmin = out_hm.min().min()

			vmax = in_hm.max().max()
			if out_hm.max().max() > vmax:
				vmax = out_hm.max().max()

			sns.heatmap(in_hm, vmin=vmin, vmax=vmax, cmap="plasma",
						mask=stab_mask, ax=axs[0], cbar_kws={'label': 'Inner ' + cbar_kw})
			sns.heatmap(out_hm, vmin=vmin, vmax=vmax, cmap="plasma",
						mask=stab_mask, ax=axs[1], cbar_kws={'label': 'Outer ' + cbar_kw})

		else:
			cbar_kw = "Droplet Size (\u03BCm)"
			sns.heatmap(in_hm, vmin=in_hm.min().min(), vmax=in_hm.max().max(), cmap="viridis",
						mask=stab_mask, ax=axs[0], cbar_kws={'label': 'Inner ' + cbar_kw})
			sns.heatmap(out_hm, vmin=out_hm.min().min(), vmax=out_hm.max().max(), cmap="viridis",
						mask=stab_mask, ax=axs[1], cbar_kws={'label': 'Outer ' + cbar_kw})

		axs[0].set_xlabel('Dispersed Phase Flow Rate (\u03BCL/hr)')
		axs[0].set_ylabel('Continuous Phase Flow Rate (\u03BCL/hr)')
		axs[1].set_xlabel('Dispersed Phase Flow Rate (\u03BCL/hr)')
		axs[1].set_ylabel('Continuous Phase Flow Rate (\u03BCL/hr)')
		return fig


	def run_stability(self, inner_features, outer_features):
		inner_generator_sweep = self.generate_inner_grid(inner_features)
		inner_generator_results = pd.DataFrame(self.predict_sweep(inner_generator_sweep, inner=True))

		outer_generator_sweep = self.generate_outer_grid(outer_features)
		outer_generator_results = pd.DataFrame(self.predict_sweep(outer_generator_sweep, inner=False))
		in_fprop = {
			"surface_tension": self.fluid_properties["inner_surface_tension"],
			"water_viscosity": self.fluid_properties["inner_aq_viscosity"],
			"oil_viscosity": self.fluid_properties["oil_viscosity"],
			"viscosity_ratio": inner_features["viscosity_ratio"]
		}

		out_fprop = {
			"surface_tension": self.fluid_properties["outer_surface_tension"],
			"water_viscosity": self.fluid_properties["outer_aq_viscosity"],
			"oil_viscosity": self.fluid_properties["outer_aq_viscosity"],
			"viscosity_ratio": outer_features["viscosity_ratio"]

		}
		fnames = []
		folder_path = os.path.join(os.getcwd(), "app", "static", "img")

		for filename in os.listdir(folder_path):
			if filename.startswith('stability'):
				os.remove(os.path.join(folder_path, filename))

		for outer in outer_generator_results.continuous_flow_rate.unique():
			in_results = inner_generator_results.copy()
			for i, row in in_results.iterrows():
				total_flow = row.dispersed_flow_rate + row.continuous_flow_rate
				outer_point = outer_generator_results.loc[outer_generator_results.continuous_flow_rate == outer, :]
				outer_point = outer_point.loc[outer_point.dispersed_flow_rate == total_flow, :]
				in_results.loc[i, "stability"] = float(
					self.pct_difference(row["generation_rate"], outer_point.generation_rate) <= 5.0)
				in_results.loc[i, "outer_diameter"] = float(outer_point.droplet_size)
				in_results.loc[i, "outer_rate"] = float(outer_point.generation_rate) #TODO: take out if needed
			# depending on the stability value, either have it be (1) colored or (2) greyed out
			## Need to make a mask showing where things are stable or not
			in_results.continuous_flow_rate = np.round(in_results.continuous_flow_rate, 1)
			stab_mask = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
										 values="stability")
			stab_mask = stab_mask[::-1].astype(bool)
			stab_mask = ~stab_mask
			inner_size_hm = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
											 values="droplet_size")[::-1]
			outer_size_hm = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
											 values="outer_diameter")[::-1]
			fig = self.plot_stability(inner_size_hm, outer_size_hm, stab_mask, outer)
			fname = "stability_plot_" + str(int(outer)) + "_" + str(time.time()) + "_flow.png"
			fnames.append(fname)
			fig.savefig(os.path.join(folder_path, fname))
			plt.close(fig)

			# TODO: TEMP, JUST FOR ALI, TAKE THIS OUT WHEN NOT NEEDED
			inner_rate_hm = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
											 values="generation_rate")[::-1]
			outer_rate_hm = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
											 values="outer_rate")[::-1]
			fig = self.plot_stability(inner_rate_hm, outer_rate_hm, stab_mask, outer, rate=True)
			fname = "stability_plot_" + str(int(outer)) + "_" + str(time.time()) + "_rate_flow.png"
			fnames.append(fname)
			fig.savefig(os.path.join(folder_path, fname))
			plt.close(fig)

		return fnames


