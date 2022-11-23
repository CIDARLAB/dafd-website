import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.mod_dafd.helper_scripts.ModelHelper3 import ModelHelper3
from app.mod_dafd.helper_scripts.DEHelper import DEHelper
from app.mod_dafd.core_logic.ForwardModel3 import ForwardModel3
import seaborn as sns
from app.mod_dafd.helper_scripts.TolHelper3 import TolHelper3

global MH
global fwd_model
global fluid_properties

def predict_sweep(feature_sweep, inner=False):
    results = []
    if inner:
        fluid_properties["surface_tension"] = fluid_properties["inner_surface_tension"]
    else:
        try:
            fluid_properties["surface_tension"] = fluid_properties["inner_surface_tension"]
        except:
            fluid_properties["surface_tension"] = fluid_properties["_surface_tension"]
    for feature_set in feature_sweep:
        fwd_results = fwd_model.predict_size_rate([feature_set[x] for x in MH.input_headers],
                                                       fluid_properties, as_dict=True)
        feature_set.update(fwd_results)
        results.append(feature_set)
    return results




in_feat = {'orifice_width': 22.5, 'aspect_ratio': 1, 'normalized_oil_inlet': 1, 'normalized_water_inlet': 1, 'expansion_ratio': 1, 'viscosity_ratio': 1.6339240506329114}
out_feat = {'orifice_width': 45, 'aspect_ratio': 1, 'normalized_oil_inlet': 1, 'normalized_water_inlet': 1, 'expansion_ratio': 1, 'viscosity_ratio': 0.807}
fluid_properties = {'inner_aq_viscosity': 0.9875, 'oil_viscosity': 1.6135, 'outer_aq_viscosity': 1.3034, 'inner_surface_tension': 0.319, 'outer_surface_tension': 0.318}

fwd_model = ForwardModel3()
MH = ModelHelper3.get_instance()
DH = DEHelper(MH, fluid_properties)

inner_generator_sweep = DH.generate_inner_grid(in_feat)
inner_generator_results = pd.DataFrame(predict_sweep(inner_generator_sweep, inner=True))

outer_generator_sweep = DH.generate_outer_grid(out_feat)
outer_generator_results = pd.DataFrame(predict_sweep(outer_generator_sweep, inner=False))
in_fprop = {
        "surface_tension": fluid_properties["inner_surface_tension"],
        "water_viscosity": fluid_properties["inner_aq_viscosity"],
        "oil_viscosity": fluid_properties["oil_viscosity"],
        "viscosity_ratio": in_feat["viscosity_ratio"]
        }

out_fprop = {
    "surface_tension": fluid_properties["outer_surface_tension"],
    "water_viscosity": fluid_properties["outer_aq_viscosity"],
    "oil_viscosity": fluid_properties["outer_aq_viscosity"],
    "viscosity_ratio": out_feat["viscosity_ratio"]

}


for outer in outer_generator_results.continuous_flow_rate.unique():
    in_results = inner_generator_results.copy()
    for i, row in in_results.iterrows():
        total_flow = row.dispersed_flow_rate + row.continuous_flow_rate
        outer_point = outer_generator_results.loc[outer_generator_results.continuous_flow_rate == outer,:]
        outer_point = outer_point.loc[outer_point.dispersed_flow_rate == total_flow,:]
        in_results.loc[i, "stability"] = float(DH.pct_difference(row["generation_rate"], outer_point.generation_rate) <= 15.0)
        in_results.loc[i, "outer_diameter"] = float(outer_point.droplet_size)
    # plt.scatter(stable.dispersed_flow_rate, stable.continuous_flow_rate)
    # plt.scatter(unstable.dispersed_flow_rate, unstable.continuous_flow_rate)
    # plt.title("Outer aqueous flow rate: " + str(outer))
    # plt.xlabel("Water flow rate")
    # plt.ylabel("Oil flow rate")
    # plt.show()
    # depending on the stability value, either have it be (1) colored or (2) greyed out
    ## Need to make a mask showing where things are stable or not
    in_results.continuous_flow_rate = np.round(in_results.continuous_flow_rate,1)
    stab_mask = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
                                              values="stability")
    stab_mask = stab_mask[::-1].astype(bool)
    stab_mask = ~stab_mask
    inner_size_hm = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
                                                  values="droplet_size")[::-1]
    outer_size_hm = in_results.pivot(index="continuous_flow_rate", columns="dispersed_flow_rate",
                                                  values="outer_diameter")[::-1]


    dx = 0.15
    dy = 1
    figsize = plt.figaspect(float(dx *  2) / float(dy * 1))
    fig, axs = plt.subplots(1, 2, facecolor="w", figsize=figsize)
    fig.suptitle('Device stability with outer flow rate of ' + str(outer), fontsize=16)
    plt.subplots_adjust(wspace=0.3, bottom=0.2)
    axs[0].set_facecolor("#bebebe")
    axs[1].set_facecolor("#bebebe")

    sns.heatmap(inner_size_hm, vmin=inner_size_hm.min().min(), vmax=inner_size_hm.max().max(), cmap="viridis", mask=stab_mask, ax=axs[0], cbar_kws={'label': 'Inner Droplet Size (\u03BCm)'})
    sns.heatmap(outer_size_hm, vmin=outer_size_hm.min().min(), vmax=outer_size_hm.max().max(), cmap="plasma", mask=stab_mask, ax=axs[1], cbar_kws={'label': 'Outer Droplet Size (\u03BCm)'})
    fig.show()

a = 2