import json
import numpy as np
from azureml.core.model import Model
from fastai.vision import *


# load the model
def init():
    global learn
    # retrieve the path to the model file using the model name
    try:
        model_path = Model.get_model_path('cats_vs_dogs')
        learn = load_learner(model_path)
    except:
        print("Model doesn't exist or can't be loaded")

# Passes data to the model and returns the prediction
def run(raw_data):
    if (learn):
        tfms = tfms_from_model(resnet34, sz)
        img = tfms(json.loads(raw_data)['data'])
        # make prediction
        result = learn.predict_array(img[None])
        pred = np.argmax(result)
        return json.dumps(pred)
    else:
        return json.dumps({ "error": "The learner coudn't be initialized"})