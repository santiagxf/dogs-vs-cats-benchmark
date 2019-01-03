import torch
import time

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.backends.cudnn.enabled)

import fastai
print(fastai.__version__)

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

#accuracy(*learn.TTA())