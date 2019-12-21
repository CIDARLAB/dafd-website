from flask import Flask, render_template, request, redirect, url_for, session
import os
from config import CONFIG, REDIS
from werkzeug.utils import secure_filename

from make_celery import make_celery

app = Flask(__name__)
app.secret_key = CONFIG['secret_key']
domain = CONFIG['domain']

app.config.update(
    CELERY_BROKER_URL=REDIS['broker'],
    CELERY_RESULT_BACKEND=REDIS['backend']
)
celery = make_celery(app)

from app.mod_NN.views import nn_blueprint
app.register_blueprint(nn_blueprint, url_prefix='/neural-net')

@app.route("/")
@app.route("/index.html")
@app.route("/index")
def index():
	
	return render_template('index.html', nn_server=domain)

@app.route("/low_cost.html")
@app.route("/low_cost")
def low_cost():
	
	return render_template('low-cost.html')

@app.route("/droplet_based.html")
@app.route("/droplet_based")
def droplet_based():
	
	return render_template('droplet-based.html')

@app.route("/single_cell.html")
@app.route("/single_cell")
def single_cell():
	
	return render_template('single-cell.html')

@app.route("/tips.html")
@app.route("/tips")
def tips():
	
	return render_template('tips.html')

@app.route("/team.html")
@app.route("/team")
def team():
	
	return render_template('team.html')

@app.route("/collaborate.html")
@app.route("/collaborate")
def collaborate():
	
	return render_template('collaborate.html')

@app.route("/publications.html")
@app.route("/publications")
def publications():
	
	return render_template('publications.html')


'''The following part below is for celery test'''
@celery.task()
def add_together(a, b):
    return a + b

@app.route("/celery-test")
def celery_test():
	
	result = add_together.delay(10, 20)
	print(result.wait())
	
	return 'Welcome to celery test!'