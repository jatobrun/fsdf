from flask import Flask, url_for
from pathlib import Path
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        module="torch.nn.functional")

classes = ['NORMAL', 'PNEUMONIA VIRAL', 'PNEUMONIA BACTERIANA']
bs = 32
data = ImageDataBunch.single_from_classes(
    Path(''), classes, ds_tfms=get_transforms(), size=224).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.load('model_best.pth')
print(learn)
# print(learner)
# print(type(learner))
