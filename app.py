import os
import sys

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask,request,render_template
import numpy as np
import pandas as pd
	
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
	
app=application
	
## Route for a home page
	
@app.route('/')
def index():
	return render_template('index.html') 
	
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
	if request.method=='GET':
	    return render_template('home.html')
	else:
	    data=CustomData(
	    age=int(request.form.get('age')),
	    gender=int(request.form.get('gender')),
	    chest_pain=int(request.form.get('chest_pain')),
	    rest_bps=int(request.form.get('rest_bps')),
	    cholestrol=int(request.form.get('cholestrol')),
	    fasting_blood_sugar=int(request.form.get('fasting_blood_sugar')),
	    rest_ecg=int(request.form.get('rest_ecg')),
	    thalach=int(request.form.get('thalach')),
	    exer_angina=int(request.form.get('exer_angina')),
	    old_peak=float(request.form.get('old_peak')),
	    slope=int(request.form.get('slope')),
	    ca=int(request.form.get('ca')),
	    thalassemia=int(request.form.get('thalassemia'))
	    
	    )
     
	pred_df=data.get_data_as_data_frame()
	print(pred_df)
	print("Before Prediction")
	
	predict_pipeline=PredictPipeline()
	print("Mid Prediction")
	results=predict_pipeline.predict(pred_df)
	print("after Prediction")
	return render_template('home.html',results=results[0])
	    
	
if __name__=="__main__":
    app.run(host="0.0.0.0",port=8890)   