from flask import Flask,render_template,jsonify,request,send_file
from src.exception import CustomException
from src.logger import logging as lg
import os,sys

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/train",methods=["GET","POST"])
def train_route():
    try:
        if request.method=="POST" or request.method=="GET":

            train_pipeline = TrainingPipeline()
            train_file_detail=train_pipeline.run_pipeline()
            lg.info("training Completed,Downloading model file")

            return send_file(train_file_detail,download_name="model.pkl",as_attachment=True)

    except Exception as e:
        raise CustomException(e,sys)
    
@app.route("/predict",methods=["POST","GET"])
def predict():

    try:
        if request.method=="POST":
            prediction_pipeline=PredictionPipeline(request)
            prediction_file_detail=prediction_pipeline.run_pipeline()

            lg.info("Predictioncompleted.Downloadingpredction file.")

            return send_file(prediction_file_detail.prediction_file_path,download_name=prediction_file_detail.prediction_file_name,as_attachment=True)
        return render_template("prediction.html")
        
    except Exception as e:
        raise CustomException(e,sys)

if __name__=="__main__":
    host="0.0.0.0"
    port=int(os.environ.get("PORT",5000))
    print(f"App running on:http://{host}:{port}")
    app.run(host=host,port=port,debug=False)
        
        