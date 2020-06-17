from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory
import pandas as pd
import os
from werkzeug.utils import secure_filename
import time

nn_blueprint = Blueprint('nn', __name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

from app.mod_NN.controllers import validFile, getDataType, runNN, runForward, runReverse

@nn_blueprint.route('/')
@nn_blueprint.route('/index')
@nn_blueprint.route('/index.html')
def index():

    return redirect(url_for('index'))

@nn_blueprint.route('/forward', methods=['GET', 'POST'])
def forward():

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
        perform['Generation Rate (Hz)'] = round(float(parsed[0]), 3)
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[1]), 3)
        perform['Regime'] = 'Dripping' if parsed[2]=='1' else 'Jetting'

        values = {}
        values['Oil Flow Rate (ml/hr)'] = round(float(parsed[3]), 3)
        values['Water Flow Rate (\u03BCl/min)'] = round(float(parsed[4]), 3)
        values['Droplet Inferred Size (\u03BCm)'] = round(float(parsed[5]), 3)

        forward2 = {}
        forward2['Orifice Size'] = forward['orifice_size']
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

        return render_template('forward.html', perform=perform, values=values, forward2=forward2,
                               tolTest = (tolerance is not None), features=features_denormalized,
                               fig_names=fig_names)

    return redirect(url_for('index'))

@nn_blueprint.route('/backward', methods=['GET', 'POST'])
def backward():

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
        perform['Generation Rate (Hz)'] = round(float(parsed[9]), 3)
        perform['Droplet Diameter (\u03BCm)'] = round(float(parsed[10]), 3)
        perform['Inferred Droplet Diameter (\u03BCm)'] = round(float(parsed[14]), 3)
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
        return render_template('backward.html', geo=geo, flow=flow, opt=opt, perform=perform, flowrate=flowrate,
                                gen_rate=gen_rate, flow_rate=flow_rate, values=flowrate, features=features_denormalized,
                                fig_names=fig_names, tolTest = (tolerance is not None))

    return redirect(url_for('index'))


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

@nn_blueprint.route('/download', methods=['GET', 'POST'])
def download():

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
