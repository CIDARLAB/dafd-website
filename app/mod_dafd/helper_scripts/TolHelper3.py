from app.mod_dafd.bin.DAFD3_Interface import DAFD3_Interface
from app.mod_dafd.tolerance_study.plot_utils import *
from app.mod_dafd.tolerance_study.tol_utils import *
from SALib.sample import saltelli
from SALib.analyze import sobol
import os
import numpy as np
import pandas as pd
import seaborn as sns


class TolHelper3:
    """This class contains the main functions needed for JUST plotting ."""
    features_normalized = {}
    features_denormalized = {}
    warnings = []
    tolerance = None
    di = None
    tol_df = None

    flow_heatmap_size = None
    flow_heatmap_gen = None
    flow_grid_size = None


    def __init__(self, feature_inputs, fluid_properties, di=None, tolerance=25, flow_grid_size = 11, pf_samples=100):
        self.features_normalized = feature_inputs
        self.fluid_properties = fluid_properties

        self.features_denormalized = self.denormalize_features(self.features_normalized)
        self.tolerance = tolerance/100
        if di == None:
            self.di = DAFD3_Interface()
        else:
            self.di = di
        self.tol_df = self.make_tol_df(self.features_denormalized, self.tolerance)
        self.feature_names = list(self.tol_df.columns)
        self.flow_grid_size = flow_grid_size

    def run_all(self):
        self.flow_heatmaps()

    def plot_all(self, base="toltest"):
        self.file_base = base
        if self.flow_heatmap_size is None or self.flow_heatmap_gen is None:
            self.run_all()
        folder_path = os.path.join(os.getcwd(),"app","static","img")
        for filename in os.listdir(folder_path):
            if filename.startswith('flow_hm'):
                os.remove(os.path.join(folder_path, filename))

        fname = "flow_hm_" + str(time.time()) + ".png"
        self.plot_flow_heatmaps(self.flow_heatmap_size, self.flow_heatmap_gen)
        plt.savefig(os.path.join(folder_path, fname))
        sns.set_style("ticks")

        return fname

    def flow_heatmaps(self, range_mult=None):
        if range_mult is None:
            range_mult = self.tolerance
        oil_range = [self.features_denormalized["oil_flow_rate"]*(1-range_mult),
                     self.features_denormalized["oil_flow_rate"]*(1+range_mult)]
        water_range = [self.features_denormalized["water_flow_rate"]*(1-range_mult),
                       self.features_denormalized["water_flow_rate"]*(1+range_mult)]

        flow_heatmap_size, flow_heatmap_gen = self.make_flow_heatmaps(oil_range, water_range)
        self.flow_heatmap_size = flow_heatmap_size
        self.flow_heatmap_gen  = flow_heatmap_gen
        return flow_heatmap_size, flow_heatmap_gen

    def make_tol_df(self, features, tol):
        max_feat = {key: (features[key] + tol * features[key]) for key in features.keys()}
        min_feat = {key: (features[key] - tol * features[key]) for key in features.keys()}
        return pd.DataFrame([min_feat, features, max_feat])


    def make_flow_heatmaps(self, oil_range, water_range):
        oil_rounding = int(np.abs(np.floor(np.log10((oil_range[1] - oil_range[0])/self.flow_grid_size))))
        water_rounding = int(np.abs(np.floor(np.log10((water_range[1] - water_range[0])/self.flow_grid_size))))
        oil = np.around(make_grid_range(pd.Series(oil_range), self.flow_grid_size), oil_rounding)
        water = np.around(make_grid_range(pd.Series(water_range), self.flow_grid_size), water_rounding)

        grid_dict = {"oil_flow_rate": oil, "water_flow_rate": water}
        flow_heatmap_size = self.generate_heatmap_data(grid_dict, "droplet_size", percent=False)
        flow_heatmap_gen = self.generate_heatmap_data(grid_dict, "generation_rate", percent=False)
        return flow_heatmap_size, flow_heatmap_gen


    def _heatmap_loop(self, pc, tol_df_shuff, output):
        pc_range = make_grid_range(tol_df_shuff.loc[:, pc], self.feature_grid_size)
        features = [feat for feat in tol_df_shuff.columns if feat != pc]
        heatmap_data = []
        for feat in features:
            feat_range = make_grid_range(tol_df_shuff.loc[:, feat], self.feature_grid_size)
            grid_dict = {pc: pc_range, feat: feat_range}
            heatmap_data.append(self.generate_heatmap_data(grid_dict, output))
        return heatmap_data


    def generate_heatmap_data(self, grid_dict, output, percent=True):
        key_names = list(grid_dict.keys())
        pts, grid = make_sample_grid(self.features_denormalized, grid_dict)
        grid_measure = [self.di.runForward(self.renormalize_features(pt), self.fluid_properties, model="both") for pt in grid]
        outputs = [out[output] for out in grid_measure]
        for i, pt in enumerate(pts):
            pt.append(outputs[i])
        heat_df = pd.DataFrame(pts, columns=[key_names[0], key_names[1], output])
        if percent:
            heat_df.loc[:, key_names[0]] = pct_change(heat_df.loc[:, key_names[0]],
                                                      self.features_denormalized[key_names[0]]).astype(float)
            heat_df.loc[:, key_names[1]] = pct_change(heat_df.loc[:, key_names[1]],
                                                      self.features_denormalized[key_names[1]]).astype(float)
            base_out = self.di.runForward(self.features_normalized, self.fluid_properties, model="both")[output]
            heat_df.loc[:, output] = pct_change(heat_df.loc[:, output], base_out)
        heat_pivot = heat_df.pivot(index=key_names[1], columns=key_names[0], values=output)
        return heat_pivot[::-1]



    def denormalize_features(self, features):
        Or = features["orifice_width"]
        As = features["aspect_ratio"]
        Exp = features["expansion_ratio"]
        norm_Wi = features["normalized_water_inlet"]
        norm_Oi = features["normalized_oil_inlet"]
        Q_ratio = features["flow_rate_ratio"]
        capillary_number = features["capillary_number"]

        surface_tension = self.fluid_properties["surface_tension"]
        oil_viscosity = self.fluid_properties["oil_viscosity"]

        channel_height = Or * As
        outlet_channel_width = Or * Exp
        water_inlet_width = Or * norm_Wi
        oil_inlet = Or * norm_Oi
        oil_flow_rate = capillary_number * Or ** 2 * As * surface_tension / oil_viscosity * 60 ** 2 * 10 ** -3
        water_flow_rate = oil_flow_rate / Q_ratio

        ret_dict = {
            "orifice_width": Or,
            "depth": channel_height,
            "outlet_width": outlet_channel_width,
            "water_inlet_width": water_inlet_width,
            "oil_inlet_width": oil_inlet,
            "oil_flow_rate": oil_flow_rate,
            "water_flow_rate": water_flow_rate
        }
        return ret_dict


    def renormalize_features(self, features):
        channel_height = features["depth"]
        outlet_channel_width = features["outlet_width"]
        water_inlet_width = features["water_inlet_width"]
        oil_inlet = features["oil_inlet_width"]
        oil_flow_rate = features["oil_flow_rate"]
        water_flow_rate = features["water_flow_rate"]

        Or = features["orifice_width"]
        As = channel_height/Or
        Exp = outlet_channel_width/Or
        norm_Wi = water_inlet_width/Or
        norm_Oi = oil_inlet/Or

        Q_ratio = oil_flow_rate / water_flow_rate

        ca_num = self.fluid_properties["oil_viscosity"]*oil_flow_rate/(Or**2*As*self.fluid_properties["surface_tension"]) * (1/3.6)

        ret_dict = {
            "orifice_width": Or,
            "aspect_ratio": As,
            "expansion_ratio": Exp,
            "normalized_water_inlet": norm_Wi,
            "normalized_oil_inlet": norm_Oi,
            "flow_rate_ratio": Q_ratio,
            "capillary_number":  round(ca_num, 5),
            "viscosity_ratio": self.fluid_properties["oil_viscosity"]/self.fluid_properties["water_viscosity"]
            }
        return ret_dict

    def plot_flow_heatmaps(self, size_df, rate_df):
        SMALL_SIZE = 12
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

        tick_spacing = int(np.floor(len(size_df.columns) / 10))
        dx = 0.15
        dy = 1
        figsize = plt.figaspect(float(dx * 2) / float(dy * 1))
        fig, axs = plt.subplots(1, 2, facecolor="w", figsize=figsize)
        plt.subplots_adjust(wspace=0.3, bottom=0.2)
        sns.set_style("white")
        sns.set_context("notebook")
        sns.set(font_scale=1.25)
        sns.heatmap(size_df, cmap="viridis", ax=axs[0], xticklabels=tick_spacing,
                    yticklabels=tick_spacing, cbar_kws={'label': 'Droplet Size (\u03BCm)'})
        axs[0].tick_params(axis='x', labelrotation=30)
        axs[0].scatter(len(size_df.columns) / 2, len(size_df.columns) / 2, marker="*", color="w", s=200)
        plt.setp(axs[0], xlabel="Oil Flow Rate (\u03BCL/hr)", ylabel="Water Flow Rate (\u03BCL/hr)")

        sns.heatmap(rate_df, cmap="plasma", ax=axs[1], xticklabels=tick_spacing,
                    yticklabels=tick_spacing, cbar_kws={'label': 'Generation Rate (Hz)'})
        axs[1].tick_params(axis='x', labelrotation=30)
        plt.setp(axs[1], xlabel="Oil Flow Rate (\u03BCL/hr)", ylabel="Water Flow Rate (\u03BCL/hr)")
        axs[1].scatter(len(size_df.columns) / 2, len(size_df.columns) / 2, marker="*", color="w", s=200)
        return fig