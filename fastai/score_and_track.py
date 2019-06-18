import json
import torch
import os, base64
from fastai import *
from fastai.vision import *
from shutil import copyfile
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector

def init():
    global learn
    
    model_file = Model.get_model_path("cats_vs_dogs")
    model_path = os.path.dirname(model_file)
    print(model_path)
    learn = load_learner(model_path)
    
    global prediction_dc
    prediction_dc = ModelDataCollector("best_model", identifier="predictions", feature_names=["prediction"])

def run(raw_data):
    base64_string = json.loads(raw_data)['data']
    base64_bytes = base64.b64decode(base64_string)
    with open(os.path.join(os.getcwd(),"score.jpg"), 'wb') as f:
        f.write(base64_bytes)
    
    # make prediction
    img = open_image(os.path.join(os.getcwd(),"score.jpg"))
    result = learn.predict(img)
    prediction_dc.collect(result)
    return json.dumps({'category':str(result[0]), 'confidence':result[2].data[1].item()})