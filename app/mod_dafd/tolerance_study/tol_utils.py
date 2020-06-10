import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import time
import itertools

def make_grid_range(vals, size):
    return np.linspace(vals.min(), vals.max(), size)

def make_sample_grid(base_features, perturbations):
    base_copy = base_features.copy()
    pert_vals = list(perturbations.values())
    options = itertools.product(pert_vals[0], pert_vals[1])
    pts = []
    grid = []
    for option in options:
        pts.append(list(option))
        base_copy.update({key: option[i] for i, key in enumerate(perturbations.keys())})
        grid.append(base_copy.copy())
    return pts, grid



def get_principal_feature(si, feature_names):
    ST = list(si["ST"])
    return feature_names[ST.index(max(ST))]


def min_dist_idx(pt, array):
    distances = [np.linalg.norm(pt - arraypt) for arraypt in array]
    return distances.index(min(distances))


def main_effect_analysis(data, inputs_df):
    size_vars = []
    gen_vars = []
    for col in inputs_df.columns:
        size_means = data.groupby(col)["droplet_size"].mean()
        gen_means = data.groupby(col)["generation_rate"].mean()
        size_vars.append(np.var(size_means))
        gen_vars.append(np.var(gen_means))

    size_var = np.var(data.loc[:, "droplet_size"])
    gen_var = np.var(data.loc[:, "generation_rate"])
    summary = pd.DataFrame([size_vars / size_var, gen_vars / gen_var], index=["size var", "gen var"],
                           columns=inputs_df.columns)
    summary = summary.T
    return summary

def to_list_of_dicts(samples, keys):
    sample_dict_list = []
    for sample in samples:
        sample_dict_list.append({key: sample[i] for i, key in enumerate(keys)})
    return sample_dict_list


def pct_change(array, base):
    return np.around((array - base)/base * 100, 3)

def denormalize_features(features):
    Or = features["orifice_size"]
    As = features["aspect_ratio"]
    Exp = features["expansion_ratio"]
    norm_Ol = features["normalized_orifice_length"]
    norm_Wi = features["normalized_water_inlet"]
    norm_Oi = features["normalized_oil_inlet"]
    Q_ratio = features["flow_rate_ratio"]
    Ca_num = features["capillary_number"]

    channel_height = Or * As
    outlet_channel_width = Or * Exp
    orifice_length = Or * norm_Ol
    water_inlet_width = Or * norm_Wi
    oil_inlet = Or * norm_Oi
    oil_flow_rate = (Ca_num * 0.005 * channel_height * oil_inlet * 1e-12) / \
                    (0.0572 * ((water_inlet_width * 1e-6)) * (
                            (1 / (Or * 1e-6)) - (1 / (2 * oil_inlet * 1e-6))))
    oil_flow_rate_ml_per_hour = oil_flow_rate * 3600 * 1e6
    water_flow_rate = oil_flow_rate_ml_per_hour / Q_ratio
    water_flow_rate_ul_per_min = water_flow_rate * 1000 / 60

    ret_dict = {
        "orifice_size": Or,
        "depth": channel_height,
        "outlet_width": outlet_channel_width,
        "orifice_length": orifice_length,
        "water_inlet": water_inlet_width,
        "oil_inlet": oil_inlet,
        "oil_flow": oil_flow_rate_ml_per_hour,
        "water_flow": water_flow_rate_ul_per_min
    }
    return ret_dict
