import torch
import numpy as np
import fastai
from fastai import *
from fastai.vision import *

print("PyTorch version %s" % torch.__version__)
print("fastai version: %s" % fastai.__version__)
print("CUDA supported: %s" % torch.cuda.is_available())
print("CUDNN enabled: %s" % torch.backends.cudnn.enabled)

path = untar_data(URLs.DOGS)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=accuracy)

learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-5,3e-4), pct_start=0.05)

saved_model_path = learn.save('cats-vs-dogs', return_path = True)
learn.export()
saved_model_pkl = str(learn.path) + '/export.pkl'

from azureml.core import Run
run = Run.get_context()

def reduce_list(all_values):
    return [np.max(all_values[i:i+10]) for i in range(0,len(all_values)-1,10)]

losses_values = [tensor.item() for tensor in learn.recorder.losses] 
accuracy_value = np.float(accuracy(*learn.TTA()))

run.log('training_acc', accuracy_value)
run.log('pytorch', torch.__version__)
run.log('fastai', fastai.__version__)
run.log('base_model', 'resnet50')
run.log_list('Learning_rate', reduce_list(learn.recorder.lrs))
run.log_list('Loss', reduce_list(losses_values))

from shutil import copyfile
copyfile(saved_model_pkl, './outputs/cats-vs-dogs.pkl')