from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import time

nn_blueprint = Blueprint('nn', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
from app.mod_NN.controllers import validFile, getDataType, runNN, runForward, runForward_3, runReverse, runReverse_2, runReverse_3
#
# @nn_blueprint.route('/')
# @nn_blueprint.route('/index')
# @nn_blueprint.route('/index.html')
# def index():
#
#     return redirect(url_for('index'))

@nn_blueprint.route("/index_1.html")
@nn_blueprint.route("/index_1")
def index_1():
	return render_template('index_1.html')

@nn_blueprint.route("/index_2.html")
@nn_blueprint.route("/index_2")
def index_2():
    return render_template('index_2.html')

@nn_blueprint.route("/index_3.html")
@nn_blueprint.route("/index_3")
def index_3():
    return render_template('index_3.html')

@nn_blueprint.route('/forward_1', methods=['GET', 'POST'])
def forward_1():

    if request.method == 'POST':

        forward = {}
        forward['orifice_size'] = request.form.get('oriWid2')
        forward['aspect_ratio'] = request.form.get('aspRatio2')
        forward['expansion_ratio'] = request.form.get('expRatio2')
        forward['normalized_orifice_length'] = request.form.get('normOri2')
        forward['normalized_water_inlet'] = request.form.get('normInlet2')
        forward['normalized_oil_inlet'] = request.form.get('normOil2')
        forward['flow_rate_ratio'] = request.form.get('flowRatio2')
        forward['capillary_number'] = request.form.get('capNum2')

        strOutput = runForward(forward)
        parsed = strOutput.split(':')[1].split('|')[:-1]

        perform = {}
        perform['Generation Rate (Hz)'] = round(float(parsed[0]), 1)
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[1]), 1)
        perform['Regime'] = 'Dripping' if parsed[2]=='1' else 'Jetting'

        values = {}
        values['Oil Flow Rate (ml/hr)'] = round(float(parsed[3]), 3)
        values['Water Flow Rate (\u03BCl/min)'] = round(float(parsed[4]), 3)
        values['Droplet Inferred Size (\u03BCm)'] = round(float(parsed[5]), 1)

        forward2 = {}
        forward2['Orifice Width'] = forward['orifice_size']
        forward2['Aspect Ratio'] = forward['aspect_ratio']
        forward2['Expansion Ratio'] = forward['expansion_ratio']
        forward2['Normalized Orifice Length'] = forward['normalized_orifice_length']
        forward2['Normalized Water Inlet'] = forward['normalized_water_inlet']
        forward2['Normalized Oil Inlet'] = forward['normalized_oil_inlet']
        forward2['Flow Rate Ratio'] = forward['flow_rate_ratio']
        forward2['Capillary Number'] = forward['capillary_number']

        tolerance = request.form.get('tolerance2')
        if tolerance is not None:
            features_denormalized, fig_names = run_tolerance(forward, tolerance)
        else:
            features_denormalized = None
            fig_names = None

        return render_template('forward_1.html', perform=perform, values=values, forward2=forward2,
                               tolTest = (tolerance is not None), features=features_denormalized,
                               fig_names=fig_names, tolerance=tolerance)

    return redirect(url_for('index_1'))

@nn_blueprint.route('/forward_2', methods=['GET', 'POST'])
def forward_2():

    if request.method == 'POST':

        forward = {}
        forward['orifice_size'] = request.form.get('oriWid2')
        forward['aspect_ratio'] = request.form.get('aspRatio2')
        forward['expansion_ratio'] = request.form.get('expRatio2')
        forward['normalized_orifice_length'] = request.form.get('normOri2')
        forward['normalized_water_inlet'] = request.form.get('normInlet2')
        forward['normalized_oil_inlet'] = request.form.get('normOil2')
        forward['flow_rate_ratio'] = request.form.get('flowRatio2')
        forward['capillary_number'] = request.form.get('capNum2')

        strOutput = runForward(forward)
        parsed = strOutput.split(':')[1].split('|')[:-1]

        perform = {}
        perform['Generation Rate (Hz)'] = round(float(parsed[0]), 1)
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[1]), 1)
        perform['Regime'] = 'Dripping' if parsed[2]=='1' else 'Jetting'

        values = {}
        values['Oil Flow Rate (ml/hr)'] = round(float(parsed[3]), 3)
        values['Water Flow Rate (\u03BCl/min)'] = round(float(parsed[4]), 3)
        values['Droplet Inferred Size (\u03BCm)'] = round(float(parsed[5]), 1)

        forward2 = {}
        forward2['Orifice Width'] = forward['orifice_size']
        forward2['Aspect Ratio'] = forward['aspect_ratio']
        forward2['Expansion Ratio'] = forward['expansion_ratio']
        forward2['Normalized Orifice Length'] = forward['normalized_orifice_length']
        forward2['Normalized Water Inlet'] = forward['normalized_water_inlet']
        forward2['Normalized Oil Inlet'] = forward['normalized_oil_inlet']
        forward2['Flow Rate Ratio'] = forward['flow_rate_ratio']
        forward2['Capillary Number'] = forward['capillary_number']

        tolerance = request.form.get('tolerance2')
        if tolerance is not None:
            features_denormalized, fig_names = run_tolerance(forward, tolerance)
        else:
            features_denormalized = None
            fig_names = None

        if request.form.get('sort_by2') is not None or request.form.get('sort_by2') != "None":
            sort_by = "flow_stability"
            metrics_results, metrics_fig_name = run_metrics(forward, sort_by)

            if "dripping" in metrics_results["sort_by"]:
                metrics_results["metric_keys"] = ["dripping_overall_score", "dripping_size_score", "dripping_rate_score"]
                metrics_results["verse_group"] = "Versatility (in dripping regime)"
            elif "jetting" in metrics_results["sort_by"]:
                metrics_results["metric_keys"] = ["jetting_overall_score", "jetting_size_score", "jetting_rate_score"]
                metrics_results["verse_group"] = "Versatility (in jetting regime)"
            else:
                metrics_results["metric_keys"] = ["all_overall_score", "all_size_score", "all_rate_score"]
                metrics_results["verse_group"] = "Versatility (in both regimes)"

        else:
            metrics_results = None
            metrics_fig_name = None


        return render_template('forward_2.html', perform=perform, values=values, forward2=forward2,
                               tolTest = (tolerance is not None), features=metrics_results["feature_denormalized"],
                               fig_names=fig_names, tolerance=tolerance, metrics_results=metrics_results,
                               metrics_fig_name = metrics_fig_name, metricTest = (metrics_results is not None))

    return redirect(url_for('index_2'))

@nn_blueprint.route('/forward_3', methods=['GET', 'POST'])
def forward_3():

    if request.method == 'POST':
        # Get data
        forward = {}
        forward['orifice_width'] = request.form.get('oriWid2')
        forward['aspect_ratio'] = request.form.get('aspRatio2')
        forward['expansion_ratio'] = request.form.get('expRatio2')
        forward['normalized_water_inlet'] = request.form.get('normInlet2')
        forward['normalized_oil_inlet'] = request.form.get('normOil2')
        forward['oil_flow_rate'] = request.form.get('contFlow2')
        forward['water_flow_rate'] = request.form.get('dispFlow2')
        forward['oil_viscosity'] = request.form.get('contVisc2')
        forward['water_viscosity'] = request.form.get('dispVisc2')
        forward['surface_tension'] = request.form.get('surfTension2')

        #normalize to correct values
        forward = {key:float(forward[key]) for key in forward.keys()}
        forward['flow_rate_ratio'] = forward["oil_flow_rate"]/forward["water_flow_rate"]
        forward['viscosity_ratio'] = forward["oil_viscosity"]/forward["water_viscosity"]
        ca_num = forward["oil_viscosity"]*forward["oil_flow_rate"]/(forward["orifice_width"]**2*forward["aspect_ratio"]) * (1/3.6)
        forward['capillary_number'] = ca_num/forward["surface_tension"] #(Ca = mu * (Qc/(OriW*depth)))/surf

        strOutput = runForward_3(forward)
        parsed = strOutput.split(':')[1].split('|')[:-1]

        perform = {}
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[0]), 1)
        perform['Generation Rate (Hz)'] = round(float(parsed[3]), 1)

        values = {}
        values['Oil Flow Rate (\u03BCL/hr)'] = round(float(parsed[1]), 3)
        values['Water Flow Rate (\u03BCL/hr)'] = round(float(parsed[2]), 3)

        forward3 = {}
        forward3['Orifice Width'] = forward['orifice_width']
        forward3['Aspect Ratio'] = forward['aspect_ratio']
        forward3['Expansion Ratio'] = forward['expansion_ratio']
        forward3['Normalized Water Inlet'] = forward['normalized_water_inlet']
        forward3['Normalized Oil Inlet'] = forward['normalized_oil_inlet']
        forward3['Flow Rate Ratio'] = forward['flow_rate_ratio']
        forward3['Capillary Number'] = forward['capillary_number']
        forward3['Viscosity Ratio'] = forward['viscosity_ratio']

        return render_template('forward_3.html', perform=perform, values=values, forward3=forward3)

    return redirect(url_for('index_3'))




@nn_blueprint.route('/backward_1', methods=['GET', 'POST'])
def backward_1():

    if request.method == 'POST':
    
        constraints = {}
        constraints['orifice_size'] = request.form.get('oriWid')
        constraints['aspect_ratio'] = request.form.get('aspRatio')
        constraints['expansion_ratio'] = request.form.get('expRatio')
        constraints['normalized_orifice_length'] = request.form.get('normOri')
        constraints['normalized_water_inlet'] = request.form.get('normInlet')
        constraints['normalized_oil_inlet'] = request.form.get('normOil')
        constraints['flow_rate_ratio'] = request.form.get('flowRatio')
        constraints['capillary_number'] = request.form.get('capNum')
        constraints['regime'] = request.form.get('regime')

        desired_vals = {}
        desired_vals['generation_rate'] = request.form.get('genRate')
        desired_vals['droplet_size'] = request.form.get('dropSize')

        strOutput = runReverse(constraints, desired_vals)
        parsed = strOutput.split(':')[1].split('|')[:-1]
        geo = {}
        geo['Orifice Width (\u03BCm)'] = round(float(parsed[0]), 3)
        geo['Channel Depth (\u03BCm)'] = round(float(parsed[1]) * float(parsed[0]), 3)
        geo['Outlet Channel Width (\u03BCm)'] = round(float(parsed[2]) * float(parsed[0]), 3)
        geo['Orifice Length (\u03BCm)'] = round(float(parsed[3]) * float(parsed[0]), 3)
        geo['Water Inlet Width (\u03BCm)'] = round(float(parsed[4]) * float(parsed[0]), 3)
        geo['Oil Inlet Width (\u03BCm)'] = round(float(parsed[5]) * float(parsed[0]), 3)

        flow = {}
        flow['Flow Rate Ratio (Oil Flow Rate/Water Flow Rate)'] = round(float(parsed[6]), 3)
        flow['Capillary Number'] = round(float(parsed[7]), 3)

        opt = {}
        opt['Point Source'] = parsed[8]

        perform = {}
        perform['Generation Rate (Hz)'] = round(float(parsed[9]), 1)
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[10]), 1)
        perform['Inferred Droplet Diameter (\u03BCm)'] = round(float(parsed[14]), 1)
        perform['Regime'] = 'Dripping' if parsed[11]=='1' else 'Jetting'

        flowrate = {}
        flowrate['Oil Flow Rate (ml/hr)'] = round(float(parsed[12]), 3)
        flowrate['Water Flow Rate (\u03BCl/min)'] = round(float(parsed[13]), 3)

        gen_rate = float(parsed[9])
        flow_rate = float(parsed[13])

        tolerance = request.form.get('tolerance')
        if tolerance is not None:
            features = {key: round(float(parsed[i]),3) for i, key in enumerate(list(constraints.keys())[:-1])}
            flowrate['Droplet Inferred Size (\u03BCm)'] = perform['Inferred Droplet Diameter (\u03BCm)']
            features_denormalized, fig_names = run_tolerance(features, tolerance)
        else:
            features_denormalized = None
            fig_names = None
        return render_template('backward_1.html', geo=geo, flow=flow, opt=opt, perform=perform, flowrate=flowrate,
                               gen_rate=gen_rate, flow_rate=flow_rate, values=flowrate, features=features_denormalized,
                               fig_names=fig_names, tolTest = (tolerance is not None), tolerance=tolerance)

    return redirect(url_for('index_1'))


@nn_blueprint.route('/backward_2', methods=['GET', 'POST'])
def backward_2():
    if request.method == 'POST':

        constraints = {}
        constraints['orifice_size'] = request.form.get('oriWid')
        constraints['aspect_ratio'] = request.form.get('aspRatio')
        constraints['expansion_ratio'] = request.form.get('expRatio')
        constraints['normalized_orifice_length'] = request.form.get('normOri')
        constraints['normalized_water_inlet'] = request.form.get('normInlet')
        constraints['normalized_oil_inlet'] = request.form.get('normOil')
        constraints['flow_rate_ratio'] = request.form.get('flowRatio')
        constraints['capillary_number'] = request.form.get('capNum')
        constraints['regime'] = request.form.get('regime')

        desired_vals = {}
        desired_vals['generation_rate'] = request.form.get('genRate')
        desired_vals['droplet_size'] = request.form.get('dropSize')

        metrics = {}
        sort_by = request.form.get("sort_by")
        if sort_by is not None:
            metric_options = [
                "flow_stability",
                "overall_versatility_score",
                "size_versatility_score",
                "rate_versatility_score",
            ]
            sort_by = metric_options[int(sort_by) - 1]
        metrics["sort_by"] = sort_by
        metrics["top_k"] = request.form.get("top_k")
        if metrics["top_k"] is None:
            metrics["top_k"] = 3
        if metrics["sort_by"] is not None or metrics["top_k"] is not None:
            strOutput, filepath = runReverse_2(constraints, desired_vals, metrics)
        else:
            strOutput = runReverse(constraints, desired_vals)


        parsed = strOutput.split(':')[1].split('|')[:-1]
        geo = {}
        geo['Orifice Width (\u03BCm)'] = round(float(parsed[0]), 3)
        geo['Channel Depth (\u03BCm)'] = round(float(parsed[1]) * float(parsed[0]), 3)
        geo['Outlet Channel Width (\u03BCm)'] = round(float(parsed[2]) * float(parsed[0]), 3)
        geo['Orifice Length (\u03BCm)'] = round(float(parsed[3]) * float(parsed[0]), 3)
        geo['Water Inlet Width (\u03BCm)'] = round(float(parsed[4]) * float(parsed[0]), 3)
        geo['Oil Inlet Width (\u03BCm)'] = round(float(parsed[5]) * float(parsed[0]), 3)

        flow = {}
        flow['Flow Rate Ratio (Oil Flow Rate/Water Flow Rate)'] = round(float(parsed[6]), 3)
        flow['Capillary Number'] = round(float(parsed[7]), 3)

        opt = {}
        opt['Point Source'] = parsed[8]

        perform = {}
        perform['Generation Rate (Hz)'] = round(float(parsed[9]), 1)
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[10]), 1)
        perform['Inferred Droplet Diameter (\u03BCm)'] = round(float(parsed[14]), 1)
        perform['Regime'] = 'Dripping' if parsed[11] == '1' else 'Jetting'

        flowrate = {}
        flowrate['Oil Flow Rate (ml/hr)'] = round(float(parsed[12]), 3)
        flowrate['Water Flow Rate (\u03BCl/min)'] = round(float(parsed[13]), 3)

        features = {key: round(float(parsed[i]), 3) for i, key in enumerate(list(constraints.keys())[:-1])}

        gen_rate = float(parsed[9])
        flow_rate = float(parsed[13])

        tolerance = request.form.get('tolerance')
        if tolerance is not None:
            flowrate['Droplet Inferred Size (\u03BCm)'] = perform['Inferred Droplet Diameter (\u03BCm)']
            features_denormalized, fig_names = run_tolerance(features, tolerance)
        else:
            features_denormalized = None
            fig_names = None

        if metrics["sort_by"] is not None or metrics["top_k"] is not None:
            metrics_results, metrics_fig_name = run_metrics(features, sort_by)
            metrics_results["file_name"] = filepath
            if "dripping" in metrics_results["sort_by"]:
                metrics_results["metric_keys"] = ["dripping_overall_score", "dripping_size_score", "dripping_rate_score"]
                metrics_results["verse_group"] = "Versatility (in dripping regime)"
            elif "jetting" in metrics_results["sort_by"]:
                metrics_results["metric_keys"] = ["jetting_overall_score", "jetting_size_score", "jetting_rate_score"]
                metrics_results["verse_group"] = "Versatility (in jetting regime)"
            else:
                metrics_results["metric_keys"] = ["all_overall_score", "all_size_score", "all_rate_score"]
                metrics_results["verse_group"] = "Versatility (in both regimes)"
            metrics_results["sort_by"] = str.replace(str.capitalize(metrics_results["sort_by"]),"_"," ")
            metrics_results["top_k"] = metrics["top_k"]
        else:
            metrics_results = None
            metrics_fig_name = None

        return render_template('backward_2.html', geo=geo, flow=flow, opt=opt, perform=perform, flowrate=flowrate,
                               gen_rate=gen_rate, flow_rate=flow_rate, values=flowrate, features=features_denormalized,
                               fig_names=fig_names, tolTest=(tolerance is not None), tolerance=tolerance,
                               metrics_results=metrics_results, metrics_fig_name=metrics_fig_name, metricTest=(metrics_results is not None))

    return redirect(url_for('index_2'))


@nn_blueprint.route('/backward_3', methods=['GET', 'POST'])
def backward_3():
    if request.method == 'POST':

        constraints = {}
        constraints['orifice_width'] = request.form.get('oriWid')
        constraints['aspect_ratio'] = request.form.get('aspRatio')
        constraints['expansion_ratio'] = request.form.get('expRatio')
        constraints['normalized_water_inlet'] = request.form.get('normInlet')
        constraints['normalized_oil_inlet'] = request.form.get('normOil')
        constraints['oil_flow_rate'] = request.form.get('contFlow')
        constraints['water_flow_rate'] = request.form.get('dispFlow')


        fluid_properties = {}
        fluid_properties['oil_viscosity'] = request.form.get('contVisc')
        fluid_properties['water_viscosity'] = request.form.get('dispVisc')
        fluid_properties['surface_tension'] = request.form.get('surfTension')
        fluid_properties = {key: float(fluid_properties[key]) for key in fluid_properties.keys()}
        # normalize to correct values
        #TODO: issue here if there aren't any constraints on the flow rates,
        constraints = {key:float(constraints[key]) for key in constraints.keys() if constraints[key] is not None}
        constraints['viscosity_ratio'] = fluid_properties["oil_viscosity"]/fluid_properties["water_viscosity"]

        if "oil_flow_rate" in constraints.keys():
            ca_num = constraints["oil_viscosity"]*constraints["oil_flow_rate"]/(constraints["orifice_width"]**2*constraints["aspect_ratio"]) * (1/3.6)
            constraints['capillary_number'] = ca_num/constraints["surface_tension"] #TODO: handle weird capillary numbers without surface tension
            if "water_flow_rate" in constraints.keys():
                #TODO: need to handle case where water flow is a constraint but oil flow is not
                constraints['flow_rate_ratio'] = constraints["oil_flow_rate"] / constraints["water_flow_rate"]

        desired_vals = {}
        desired_vals['generation_rate'] = request.form.get('genRate')
        desired_vals['droplet_size'] = request.form.get('dropSize')

        strOutput, reverse_results = runReverse_3(constraints, desired_vals, fluid_properties)
        parsed = strOutput.split(':')[1].split('|')[:-1]
        geo = {}
        geo['Orifice Width (\u03BCm)'] = np.round(reverse_results["orifice_width"], 3)
        geo['Channel Depth (\u03BCm)'] = np.round(reverse_results["aspect_ratio"]*reverse_results["orifice_width"], 3)
        geo['Outlet Channel Width (\u03BCm)'] = np.round(reverse_results["expansion_ratio"]*reverse_results["orifice_width"], 3)
        geo['Dispersed Inlet Width (\u03BCm)'] = np.round(reverse_results["normalized_water_inlet"]*reverse_results["orifice_width"], 3)
        geo['Continuous Inlet Width (\u03BCm)'] = np.round(reverse_results["normalized_oil_inlet"]*reverse_results["orifice_width"], 3)

        flow = {}
        flow['Flow Rate Ratio '] = np.round(reverse_results["flow_rate_ratio"], 3)
        flow['Capillary Number'] = np.round(reverse_results["capillary_number"], 3)
        flow['Continuous Phase Dynamic Viscosity'] = np.round(fluid_properties['oil_viscosity'], 3)
        flow['Dispersed Phase Dynamic Viscosity'] = np.round(fluid_properties['water_viscosity'], 3)
        flow['Interfacial Surface Tension'] = np.round(fluid_properties['surface_tension'], 3)

        opt = {}
        opt['Point Source'] = fluid_properties['point_source']

        perform = {}
        perform['Generation Rate (Hz)'] = np.round(reverse_results["generation_rate"], 3)
        perform['Droplet Diameter (\u03BCm)'] = np.round(reverse_results["droplet_size"], 3)

        flowrate = {}
        flowrate['Continuous Phase Flow  Rate (\u03BCl/hr)'] = np.round(reverse_results["oil_flow_rate"], 3)
        flowrate['Dispersed Phase Flow Rate (\u03BCl/hr)'] = np.round(reverse_results["water_flow_rate"], 3)

        gen_rate = np.round(reverse_results["generation_rate"], 3)
        flow_rate = np.round(reverse_results["droplet_size"], 3)
        return render_template('backward_3.html', geo=geo, flow=flow, opt=opt, perform=perform, flowrate=flowrate,
                               gen_rate=gen_rate, flow_rate=flow_rate, values=flowrate)
    return redirect(url_for('index_3'))


@nn_blueprint.route('/dummy', methods=['GET', 'POST'])
def dummy():

    target = os.path.join(APP_ROOT, '../resources/inputs/')
    filename = 'dafd.csv'
    complete_filename = os.path.join(target, filename)

    df = getDataType(complete_filename)
    df = df.round(3)
    columns = df.columns.tolist()

    model_name = 'model-NN-' + str(int(round(time.time() * 1000)))
    
    return render_template('analysis.html', columns=columns, data=df.values, filename=filename, model_name=model_name)

@nn_blueprint.route('/analysis', methods=['GET', 'POST'])
def analysis():

    if request.method == 'POST':
        
        file = request.files['file']

        if not file:
            return "ERROR"
        
        if validFile(file.filename):

            target = os.path.join(APP_ROOT, '../resources/inputs/')
            filename = file.filename
            complete_filename = os.path.join(target, secure_filename(filename))

            file.save(complete_filename)

        df = getDataType(complete_filename)
        df = df.round(3)
        columns = df.columns.tolist()

        model_name = 'model-NN-' + str(int(round(time.time() * 1000)))
        
        return render_template('analysis.html', columns=columns, data=df.values, filename=filename, model_name=model_name)

    return redirect(url_for('index'))


@nn_blueprint.route("/metrics_results/<file_name>")
def metrics_results(file_name):
    directory = os.path.join(os.getcwd(),'app', 'resources')
    return send_from_directory(directory=directory, filename=file_name, as_attachment=True)#, cache_timeout=10)




@nn_blueprint.route('/run', methods=['GET', 'POST'])
def run():

    if request.method == 'POST':
        
        payload = {}
        payload['filename'] = request.form.get('filename')
        payload['model-name'] = request.form.get('model-name')
        payload['target'] = request.form.get('target_single')
        payload['mode'] = request.form.get('mode')

        payload['drops'] = request.form.getlist('drop')

        payload['metrics'] = request.form.get('metrics')
        payload['normalization'] = request.form.get('normalization')
        payload['holdout'] = float(int(request.form.get('holdout'))/100)
        payload['validation'] = request.form.get('validation')
        payload['fold'] = request.form.get('fold')
        payload['tuning'] = request.form.get('tuning')

        '''
        payload['encoding'] = request.form.get('encoding')
        payload['missing'] = request.form.get('missing')
        payload['targets'] = request.form.getlist('target')
        payload['crossval'] = request.form.get('crossval')
        payload['cv_method'] = request.form.get('cv_method')
        payload['dim_red'] = request.form.get('dim_red')
        payload['num_of_dim'] = request.form.get('dimension')
        payload['hyper-param'] = request.form.get('hyper-param')
        payload['grids'] = request.form.get('grids')
        payload['model-name'] = request.form.get('model-name')
        '''

        payload['filter'] = 'regime'	#this value only matter for regression
        payload['selected_condition'] = 1	#Or 2, this value will not matter for regime classification

        payload['save-best-config'] = True
        payload['best-config-file'] = 'best-config-classification.json'
        payload['save-architecture'] = True
        payload['architecture-file'] = 'architecture-classification.json'
        payload['save-weights'] = True
        payload['weights-file'] = 'weights-classification.h5'

        payload['epoch'] = request.form.get('epoch')
        payload['batch'] = request.form.get('batch')
        payload['num_layers'] = request.form.get('num_layers')
        payload['num_nodes'] = request.form.get('num_nodes')

        ### this actually handled by javascript
        if payload['epoch'] != "" and payload['epoch'] is not None:
            epochs = list(map(int, payload['epoch'].split(',')))
        else:
            epochs = [32]
        if payload['batch'] != "" and payload['batch'] is not None:
            batch_size = list(map(int, payload['batch'].split(',')))
        else:
            batch_size = [100]
        if payload['num_layers'] != "" and payload['num_layers'] is not None:
            num_hidden = list(map(int, payload['num_layers'].split(',')))
        else:
            num_hidden = [8]
        if payload['num_nodes'] != "" and payload['num_nodes'] is not None:
            node_hidden = list(map(int, payload['num_nodes'].split(',')))
        else:
            node_hidden = [8]
        ###
        
        if payload['tuning'] == 'none':
            tuning_params = {
                'batch_size': batch_size[0],
                'epochs': epochs[0],
                'node_hidden': node_hidden[0],
                'num_hidden': num_hidden[0]
            }
        else:
            tuning_params = {'mod__batch_size': batch_size,
				'mod__epochs': epochs,
				'mod__node_hidden': node_hidden,
				'mod__num_hidden': num_hidden
            }

        results, config = runNN(payload, tuning_params)

        #cv = 'Yes' if payload['validation']=='crossval' or payload['tuning']!='none' else 'No'
        #hy = 'Yes' if payload['tuning']!='none' else 'No'

        df = pd.DataFrame(results).round(3)
        cols = df.columns
        vals = df.values
        
        return render_template('result.html', columns=cols, data=vals, architecture=config)
        
    return redirect(url_for('index'))

@nn_blueprint.route('/download_weights', methods=['GET', 'POST'])
def download_weights():

    if request.method == 'POST':

        directory = os.path.join(APP_ROOT, '../resources/inputs/')
        return send_from_directory(directory=directory, filename='weights-classification.h5', as_attachment=True)
    
    return redirect(url_for('index'))

@nn_blueprint.route('/tolerance', methods=['GET', 'POST'])
def run_tolerance(features, tolerance):
    from app.mod_dafd.helper_scripts.TolHelper import TolHelper
    from app.mod_dafd.bin.DAFD_Interface import DAFD_Interface
    features = features.copy()
    features = {key: float(features[key]) for key in features.keys()}
    tolerance = float(tolerance)
    TH = TolHelper(features, di=DAFD_Interface(), tolerance=tolerance)
    TH.run_all()
    fig_names = TH.plot_all()
    TH.generate_report()

    return TH.features_denormalized, fig_names

@nn_blueprint.route('/metrics', methods=['GET', 'POST'])
def run_metrics(features, sort_by):
    from app.mod_dafd.helper_scripts.MetricHelper import MetricHelper
    from app.mod_dafd.bin.DAFD_Interface import DAFD_Interface
    di = DAFD_Interface()
    results = {key: float(features[key]) for key in features.keys()}
    results.update(di.runForward(results))
    if results["regime"] == 1:
        reg_str = "Dripping"
    else:
        reg_str = "Jetting"
    if "versatility" in sort_by:
        sort_by = str.lower(reg_str) + "_" + sort_by.split("_")[0] + "_" + "score"

    MetHelper = MetricHelper(results, di=di)
    MetHelper.run_all_flow_stability()
    MetHelper.run_all_versatility()
    results.update(MetHelper.versatility_results)
    results.update({"flow_stability": MetHelper.point_flow_stability})
    for k in results.keys():
        results[k] = np.round(results[k],3)

    report_info = {
        "regime": reg_str,
        "results": results,
        "sort_by": sort_by
    }
    report_info["feature_denormalized"] = MetHelper.features_denormalized
    fig_name = MetHelper.plot_metrics()
    return report_info, fig_name
