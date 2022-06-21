from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np
import json
from web3 import Web3

ganache_url = "http://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

web3.isConnected()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/forecast')
def home():
    return render_template('forecast.html')

@app.route('/tracking')
def tracking():
    return render_template('tracking.html')
 
@app.route('/service')
def services():
    return render_template('service.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    scaler_path= r'models\sc.sav'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path=r'models\lr.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)

    return jsonify({'Prediction': float(Y_pred)})

if __name__ == "__main__":
    app.run(debug=True, port=9457)
