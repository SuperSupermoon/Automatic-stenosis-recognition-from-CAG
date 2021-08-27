
"""
File name: hyper parameter tuning.py
Author: Jonghak Moon

"""


import numpy as np
import os
import datetime
import keras
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

def pause():
    input('press the enter key to continue...')

row_col_size = [510, 510]
crop_h, crop_w = (500, 500)
value_of = [(0, 0), (0, 12), (12, 0), (6, 6), (12, 12)]
img_brightness_delta = np.linspace(0.4, 2.0, num=10)
add_intensity_values = np.arange(-10, 11, step=5)
resize_option = {'resize': False}
mean_subtract = {'mean_subtract': True}
learning_rate = 1e-5
epochs = 10
""""""""""""""""""""""""""""""""""""""""
              Aug Option
"""""""""""""""""""""""""""""""""""""""""
train_control_intensity_option = {'control_intensity': True, 'values': add_intensity_values}
train_cropping_option = {'random_crop_option': True, 'times_of_crop': value_of}
train_adjust_brightness_option = {'adjust_brightness': False, 'delta': img_brightness_delta}
mean = {'mean': True}
"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
# training_mode = 'aug_ori'
training_mode = 'aug_per_patient'
gpu_number = 4
testmode = 'clip_kfold'
# kernel_initializer = 'glorot_uniform' # xavier
kernel_initializer = 'he_normal'

valid_file_batch = 1
no_total_train_records = 1
batch_size = 15
series_batch_size = 30
number_of_fold = 5
get_size = 5
dout_rate = 0.5
channel = 3
external_path = '/data/inception_v3/clip_kfold/ext_pickle_file'

model = InceptionV3
if model == InceptionV3:
    model_name = 'inceptionV3'

elif model == VGG19:
    model_name = 'VGG19'

elif model == VGG16:
    model_name = 'VGG16'

elif model == ResNet50:
    model_name = 'ResNet50'

opt_name = 'Adam'

print("training_mode", training_mode)
print("testmode'",testmode)

# optimizer = keras.optimizers.RMSprop(lr=learning_rate, decay=0.5)
# optimizer = keras.optimizers.Adam(lr=learning_rate, decay=0.01)
optimizer = keras.optimizers.Adam(lr=learning_rate)
# optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# optimizer = keras.optimizers.SGD(lr=learning_rate, decay=1e-3, momentum=0.5, nesterov=True)

verbose_option = 2
validation_batch_size = batch_size
log_path = '/data/inception_v3/clip_kfold/pretrained_log'
log_txt_front_name = str(epochs) + 'epochs' + '_' + training_mode
model_save_name = str(epochs) + 'epochs' + training_mode
logs_dir_path = os.path.join(log_path, 'results',
                             log_txt_front_name + '_' + 'GPU_' + str(
                                 gpu_number) + str(datetime.date.today()))
resize_logs_dir_path = os.path.join(log_path, 'resize_results',
                                    log_txt_front_name + '_' + '_logs_GPU_' + str(
                                        gpu_number) + '/')
log_txt_name = log_txt_front_name + str(datetime.date.today()) + '_logs_GPU_' + str(gpu_number) + '.txt'
model_save_path = logs_dir_path + '/' + model_save_name +'_'+ str(gpu_number) +'gpu'

if training_mode == 'aug_ori' or training_mode == 'ori':
    train_dump_path = '/data/inception_v3/clip_kfold/new_ori_512_2'
    fixed_dump_path = '/data/inception_v3/clip_kfold/new_ori_512_2'
    print("train_dump_path", train_dump_path)

if training_mode == 'aug_ori':
    augmentation_option = {'augmentation': True}
    no_total_train_records = 1

    if train_control_intensity_option['control_intensity']:
        no_total_train_records *= len(train_control_intensity_option['values'])
        print("control_intensity", no_total_train_records)
    else:
        pass
    if train_cropping_option['random_crop_option']:
        no_total_train_records *= len(
            train_cropping_option['times_of_crop'])  # Only can calculate times of crop.
        print("random_crop_option", no_total_train_records)
    else:
        pass
    if train_adjust_brightness_option['adjust_brightness']:
        no_total_train_records *= len(train_adjust_brightness_option['delta'])
    else:
        pass
    multiply_by_aug = int(no_total_train_records)
    print("Augmentation times", multiply_by_aug)

    fold_file_path = [
                '/data/inception_v3/'+str(testmode)+'/' + str(1) + 'fold/' + str(multiply_by_aug) + 'aug_ori',
                '/data/inception_v3/'+str(testmode)+'/' + str(2) + 'fold/' + str(multiply_by_aug) + 'aug_ori',
                '/data/inception_v3/'+str(testmode)+'/' + str(3) + 'fold/' + str(multiply_by_aug) + 'aug_ori',
                '/data/inception_v3/'+str(testmode)+'/' + str(4) + 'fold/' + str(multiply_by_aug) + 'aug_ori',
                '/data/inception_v3/'+str(testmode)+'/' + str(5) + 'fold/' + str(multiply_by_aug) + 'aug_ori']
    data_size = None


elif training_mode == 'ori':
    augmentation_option = {'augmentation': False}
    fold_file_path = ['/data/inception_v3/'+str(testmode)+'/' + str(1) + 'fold/' + 'new_ori',
            '/data/inception_v3/'+str(testmode)+'/' + str(2) + 'fold/' + 'new_ori',
            '/data/inception_v3/'+str(testmode)+'/' + str(3) + 'fold/' + 'new_ori',
            '/data/inception_v3/'+str(testmode)+'/' + str(4) + 'fold/' + 'new_ori',
            '/data/inception_v3/'+str(testmode)+'/' + str(5) + 'fold/' + 'new_ori']
    data_size = 512



"""
File name: hyper parameter tuning.py
Author: Jonghak Moon

"""