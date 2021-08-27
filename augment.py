
"""
File name: augmentation.py
Author: Jonghak Moon

"""

# In[2]:

import numpy as np
import PIL
import imageio
import sys, os, pathlib
import random
import cv2
import hyper_parameter_kfold
import scipy
from PIL import ImageEnhance


def cag_aug_method_combination(train_array, train_label, train_path,train_index, train_cropping_option, control_intensity_option):

    img_temp = np.array(train_array)
    crop_h, crop_w = hyper_parameter_kfold.crop_h, hyper_parameter_kfold.crop_w
    mean_image = 0

    ##1. Add intensity
    if control_intensity_option['control_intensity']:
        data_final = []
        values_to_add = list(control_intensity_option['values'])
        value = int(random.choice(values_to_add))
        for i in img_temp:
            image = i.astype(np.float64)
            image += value
            data_final.append(image)
        img_temp = data_final
        print("After random intensity shape", np.array(img_temp).shape)
    else:
        pass

    ## 2. Cropping
    if train_cropping_option['random_crop_option']:
        w1, h1 = random.choice(hyper_parameter_kfold.value_of)
        random_crop = [img[h1:h1 + crop_h, w1:w1 + crop_w] for img in np.array(img_temp)]
        img_temp = random_crop
    else:
        pass

    return np.array(img_temp), np.array(train_label), train_path, train_index, np.array(img_temp)
