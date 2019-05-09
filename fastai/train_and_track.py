import torch
import time
import numpy as np

print("PyTorch version %s" % torch.__version__)
print("CUDA supported: %s" % torch.cuda.is_available())
print("CUDNN enabled: %s" % torch.backends.cudnn.enabled)

import fastai
print("fastai version: %s" % fastai.__version__)

from fastai import *
from fastai.vision import *

path = untar_data(URLs.DOGS)
data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=accuracy)

start_time = time.time()
print('Training starting. Start time is %d' % start_time)

learn.fit_one_cycle(1)
learn.unfreeze()
learn.fit_one_cycle(1, slice(1e-5,3e-4), pct_start=0.05)

elapsed_time = time.time() - start_time
print('Training completed. elapsed time was %d' % elapsed_time)

from azureml.core import Run
# Get a reference to the current run
run = Run.get_context()

run.log('training_acc', np.float(accuracy(*learn.TTA())))
run.log('pytorch', torch.__version__)
run.log('base_model', 'resnet')

# Save the model (for fastai, everything will be in a file named export.pkl in the folder learn.path)
learn.export()
run.upload_file('model', str(learn.path) + '/export.pkl')

# Save can also be done in the output directory
from shutil import copyfile
import os
copyfile(str(learn.path) + '/export.pkl', './outputs/export.pkl')