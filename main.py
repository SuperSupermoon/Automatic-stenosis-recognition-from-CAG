
"""
File name: main.py
Author: Jonghak Moon

"""

from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
import keras
import numpy as np
from keras.layers import Input
import os
import pickle
from keras.utils import np_utils
# import  batch_loader_frangi_filter_0detect
import datetime
import matplotlib
import csv

matplotlib.use('agg')
import matplotlib.pyplot as plt
import hyper_parameter_kfold
import random
import scipy
import keras.applications.inception_v3 as inception

# ---------------------------------------------#
augmentation_option = hyper_parameter_kfold.augmentation_option
resize_option = hyper_parameter_kfold.resize_option

gpu_number = hyper_parameter_kfold.gpu_number
mean_subtract = hyper_parameter_kfold.mean_subtract

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)

channel = hyper_parameter_kfold.channel
batch_size = hyper_parameter_kfold.batch_size
validation_batch_size = hyper_parameter_kfold.validation_batch_size
epochs = hyper_parameter_kfold.epochs
log_path = hyper_parameter_kfold.log_path
log_txt_front_name = hyper_parameter_kfold.log_txt_front_name

logs_dir_path = hyper_parameter_kfold.logs_dir_path
resize_logs_dir_path = hyper_parameter_kfold.resize_logs_dir_path

log_txt_name = hyper_parameter_kfold.log_txt_name
learning_rate = hyper_parameter_kfold.learning_rate
number_of_fold = hyper_parameter_kfold.number_of_fold

# ---------------------------------------------#

train_control_intensity_option = hyper_parameter_kfold.train_control_intensity_option
train_cropping_option = hyper_parameter_kfold.train_cropping_option
train_adjust_brightness_option = hyper_parameter_kfold.train_adjust_brightness_option
valid_control_intensity_option = {'control_intensity': False, 'values': hyper_parameter_kfold.add_intensity_values}
valid_cropping_option = {'random_crop_option': False, 'row_col_size': hyper_parameter_kfold.row_col_size}

# ---------------------------------------------#
if resize_option['resize'] == True:

    ### Data Pickling and Loading ###
    log_text = open(os.path.join(resize_logs_dir_path, log_txt_name), 'a')
else:
    ### Data Pickling and Loading ###
    log_text = open(os.path.join(logs_dir_path, log_txt_name), 'a')

normal_list = []
abnormal_list = []
valid_list = []

if resize_option['resize'] == True:

    ### Data Pickling and Loading ###
    log_text = open(os.path.join(resize_logs_dir_path, log_txt_name), 'a')
else:
    ### Data Pickling and Loading ###
    log_text = open(os.path.join(logs_dir_path, log_txt_name), 'a')

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print('Run in %s' % datetime.datetime.now())
print('Run in %s' % datetime.datetime.now(), file=log_text)

# Path of log file
print('Logs path: ' + logs_dir_path + '\r')
print('Logs path: ' + logs_dir_path + '\r', file=log_text)

# Save parameter logs
print('augmentation_option: ' + str(augmentation_option['augmentation']), file=log_text)
print('resize_option: ' + str(resize_option['resize']), file=log_text)

print('##Augmentation Parameters##\r', file=log_text)
print('GPU: ' + str(gpu_number) + '\r', file=log_text)
print('training_train_batch_size: ' + str(batch_size) + '\r', file=log_text)
print('learning_rate: ' + str(learning_rate) + '\r', file=log_text)
print('The number of epochs: ' + str(epochs) + '\r', file=log_text)
print('drop out rate: ' + str(hyper_parameter_kfold.dout_rate) + '\r', file=log_text)
print('\n', file=log_text)

### augmentation alarm

if augmentation_option['augmentation'] == True:
    print('train_control_intensity_option: %s' % train_control_intensity_option, '\r', file=log_text)
    print('valid_control_intensity_option: %s' % valid_control_intensity_option, '\r', file=log_text)

    print('train_cropping_option: %s' % train_cropping_option, '\r', file=log_text)
    print('valid_cropping_option: %s' % valid_cropping_option, '\r', file=log_text)
    print('\r', file=log_text)

else:
    print('Augmentation option is False.')
    print('Augmentation option is False.', file=log_text)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if resize_option['resize'] == True:
    print('resize 512 to %s' % hyper_parameter_kfold.data_size)
    print('resize 512 to %s' % hyper_parameter_kfold.data_size, '\r', file=log_text)

else:
    print('Resize option is False.')
    print('Resize option is False.', file=log_text)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
input_tensor = Input(shape=(hyper_parameter_kfold.data_size, hyper_parameter_kfold.data_size, hyper_parameter_kfold.channel))

base_model = inception.InceptionV3(input_tensor=input_tensor, weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, kernel_initializer=str(hyper_parameter_kfold.kernel_initializer),
                          bias_initializer='zeros',
                          activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

fold_file_path = hyper_parameter_kfold.fold_file_path

count_kfold = 1

for path_itr in fold_file_path:
    train_path_list = []
    valid_path_list = []
    mean_image_path = []
    # valid_mean_image_path = []
    for Par_dir, Sub_name, File_name in os.walk(path_itr):

        if File_name:
            for itr in range(len(File_name)):
                file_path = os.path.join(Par_dir, File_name[itr])
                if 'train' in file_path.split('/'):
                    train_path_list.append(file_path)
                elif 'validation_clip.pickle' in file_path.split('/'):
                    print("file_path", file_path)
                    valid_path_list.append(file_path)
                elif 'mean_image.pickle' in file_path.split('/'):
                    print("file_path", file_path)
                    mean_image_path.append(file_path)
    train_acc = []
    train_loss = []
    valid_acc = []
    valid_loss = []
    time_start = datetime.datetime.now()
    print("     Training Start at", time_start)
    optimizer = hyper_parameter_kfold.optimizer
    model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=['acc'])

    mean_image_sub = []
    for val_file_itr in mean_image_path:
        with open(str(val_file_itr), 'rb') as f:
            valid = pickle.load(f)
            mean_image_sub.append(np.array(valid))

    for epoch_num in range(1, epochs + 1):
        print('\n')
        print('#######################################################')
        print('[Train epoch: %s]' % epoch_num)

        stack_hist_acc = []
        stack_hist_loss = []
        array_bundle = []

        count_aug = 1

        if augmentation_option['augmentation'] == True:
            increase = 0

            for file_itr in range(int(len(train_path_list)/int(hyper_parameter_kfold.series_batch_size))):
                list_array_sum = []
                list_label_sum = []

                for file_itr2 in train_path_list[increase:increase+int(hyper_parameter_kfold.series_batch_size)]:
                    with open(str(file_itr2), 'rb') as f:
                        train = pickle.load(f)
                        np_array_values = train[0:-3]
                        get_label = train[-3]
                        train_array = np_array_values[0]
                        label = np_utils.to_categorical(get_label, 2)
                        real_train_array = np.array(train_array)
                        real_train_label = np.array(label)
                        list_array_sum.extend(real_train_array)
                        list_label_sum.extend(real_train_label)
                increase += int(hyper_parameter_kfold.series_batch_size)
                subtracted_train = np.array(list_array_sum) - np.array(mean_image_sub)

                hist = model.fit(x=subtracted_train[0], y=np.array(list_label_sum),
                                 batch_size=hyper_parameter_kfold.batch_size, epochs=1,
                                 verbose=hyper_parameter_kfold.verbose_option)  # , callbacks=[checkpoint])
                stack_hist_acc.append(hist.history['acc'])
                stack_hist_loss.append(hist.history['loss'])
                count_aug += 1

        else:
            for file_itr in train_path_list:
                with open(str(file_itr), 'rb') as f:
                    train = pickle.load(f)
                    np_array_values = train[0:-3]
                    get_label = train[-3]
                    train_array = np_array_values[0]
                    label = np_utils.to_categorical(get_label, 2)
                    real_train_array = np.array(train_array)
                    real_train_label = np.array(label)
                    subtracted_train = real_train_array - np.array(mean_image_sub)
                    hist = model.fit(x=subtracted_train[0], y=real_train_label, batch_size=hyper_parameter_kfold.batch_size, epochs=1,verbose=hyper_parameter_kfold.verbose_option)
                    stack_hist_acc.append(hist.history['acc'])
                    stack_hist_loss.append(hist.history['loss'])
                    count_aug += 1
        avg_train = np.mean(stack_hist_acc)
        avg_loss = np.mean(stack_hist_loss)
        train_acc.append(avg_train)
        train_loss.append(avg_loss)

        for val_file_itr in valid_path_list:
            with open(str(val_file_itr), 'rb') as f:
                valid = pickle.load(f)
                np_array_values = valid[0:-3]
                get_label = valid[-3]
                vali_array_bundle = np_array_values[0]
                vali_label_bundle = get_label
                vali_label = np_utils.to_categorical(vali_label_bundle, 2)
                real_test_array = np.array(vali_array_bundle)
                real_test_label = np.array(vali_label)
                subtracted_test = real_test_array - np.array(mean_image_sub[0])
                print("mean image validation size : ", np.array(mean_image_sub[0]).shape)

                print("subtracted_test",subtracted_test.shape)

                scores = model.evaluate(x=subtracted_test, y=real_test_label, batch_size=5,
                                verbose=hyper_parameter_kfold.verbose_option)
                valid_acc.append(scores[1])
                valid_loss.append(scores[0])

                if resize_option['resize'] == True:
                    with open(os.path.join(resize_logs_dir_path, os.path.splitext(log_txt_name)[0] + '_test.csv'), 'a') as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        writer.writerow(np.array([count_kfold, epoch_num, scores[1]]))
                else:
                    with open(os.path.join(logs_dir_path, os.path.splitext(log_txt_name)[0]+'_test.csv'), 'a') as csv_file:
                        writer = csv.writer(csv_file, delimiter=',')
                        writer.writerow(np.array([count_kfold, epoch_num, scores[1]]))

    if augmentation_option['augmentation'] == True:
        if resize_option['resize'] == True:
            hist_fig = plt.figure('hist')
            hist_fig_subplot = hist_fig.add_subplot(1, 1, 1)
            hist_fig_subplot.plot(train_acc, 'g', label='train acc')
            hist_fig_subplot.plot(valid_acc, 'b', label='test acc')
            hist_fig_subplot.legend(loc="upper left")
            hist_fig_subplot.set_xlabel('epochs')
            hist_fig_subplot.set_ylabel('accuracy')
            hist_fig.savefig(os.path.join(resize_logs_dir_path, 'fold_Augmentation_' + hyper_parameter_kfold.opt_name + '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                             bbox_inches='tight')
            hist_fig.clf()

            hist_fig2 = plt.figure('hist2')
            hist_fig2_subplot = hist_fig2.add_subplot(1, 1, 1)
            hist_fig2_subplot.plot(train_loss, 'r', label='train loss')
            hist_fig2_subplot.plot(valid_loss, 'b', label='test loss')
            hist_fig2_subplot.legend(loc="upper left")
            hist_fig2_subplot.set_xlabel('epochs')
            hist_fig2_subplot.set_ylabel('loss')
            hist_fig2.savefig(os.path.join(resize_logs_dir_path, 'fold_Augmentation_loss' + hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' + '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                              bbox_inches='tight')
            hist_fig2.clf()
        else:

            hist_fig = plt.figure('hist')
            hist_fig_subplot = hist_fig.add_subplot(1, 1, 1)
            hist_fig_subplot.plot(train_acc, 'g', label='train acc')
            hist_fig_subplot.plot(valid_acc, 'b', label='test acc')
            hist_fig_subplot.legend(loc="upper left")
            hist_fig_subplot.set_xlabel('epochs')
            hist_fig_subplot.set_ylabel('accuracy')
            hist_fig.savefig(os.path.join(logs_dir_path, 'fold_Augmentation_' + hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' + '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                             bbox_inches='tight')
            hist_fig.clf()

            hist_fig2 = plt.figure('hist2')
            hist_fig2_subplot = hist_fig2.add_subplot(1, 1, 1)
            hist_fig2_subplot.plot(train_loss, 'r', label='train loss')
            hist_fig2_subplot.plot(valid_loss, 'b', label='test loss')
            hist_fig2_subplot.legend(loc="upper left")
            hist_fig2_subplot.set_xlabel('epochs')
            hist_fig2_subplot.set_ylabel('loss')
            hist_fig2.savefig(os.path.join(logs_dir_path, 'fold_Augmentation_loss'+ hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' + '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                              bbox_inches='tight')
            hist_fig2.clf()
    else:
        if resize_option['resize'] == True:
            hist_fig = plt.figure('hist')
            hist_fig_subplot = hist_fig.add_subplot(1, 1, 1)
            hist_fig_subplot.plot(train_acc, 'g', label='train acc')
            hist_fig_subplot.plot(valid_acc, 'b', label='test acc')
            hist_fig_subplot.legend(loc="upper left")
            hist_fig_subplot.set_xlabel('epochs')
            hist_fig_subplot.set_ylabel('accuracy')
            hist_fig.savefig(os.path.join(resize_logs_dir_path, 'No_fold_Aug_' + hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                             bbox_inches='tight')
            hist_fig.clf()

            hist_fig2 = plt.figure('hist2')
            hist_fig2_subplot = hist_fig2.add_subplot(1, 1, 1)
            hist_fig2_subplot.plot(train_loss, 'r', label='train loss')
            hist_fig2_subplot.plot(valid_loss, 'b', label='test loss')
            hist_fig2_subplot.legend(loc="upper left")
            hist_fig2_subplot.set_xlabel('epochs')
            hist_fig2_subplot.set_ylabel('loss')
            hist_fig2.savefig(os.path.join(resize_logs_dir_path, 'No_fold_Aug_loss' + hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' + '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                              bbox_inches='tight')
            hist_fig2.clf()

        else:
            hist_fig = plt.figure('hist')
            hist_fig_subplot = hist_fig.add_subplot(1, 1, 1)
            hist_fig_subplot.plot(train_acc, 'g', label='train acc')
            hist_fig_subplot.plot(valid_acc, 'b', label='test acc')
            hist_fig_subplot.legend(loc="upper left")
            hist_fig_subplot.set_xlabel('epochs')
            hist_fig_subplot.set_ylabel('accuracy')
            hist_fig.savefig(os.path.join(logs_dir_path, 'No_fold_Aug_'+ hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                             bbox_inches='tight')
            hist_fig.clf()

            hist_fig2 = plt.figure('hist2')
            hist_fig2_subplot = hist_fig2.add_subplot(1, 1, 1)
            hist_fig2_subplot.plot(train_loss, 'r', label='train loss')
            hist_fig2_subplot.plot(valid_loss, 'b', label='test loss')
            hist_fig2_subplot.legend(loc="upper left")
            hist_fig2_subplot.set_xlabel('epochs')
            hist_fig2_subplot.set_ylabel('loss')
            hist_fig2.savefig(os.path.join(logs_dir_path, 'No_fold_Aug_loss'+ hyper_parameter_kfold.opt_name + str(
                hyper_parameter_kfold.get_size) + 'frame' + '_' + 'batch' + str(batch_size) + '_epochs' + str(
                epochs) + '_gpu' + str(gpu_number) + '_lr' + str(learning_rate) + '_fold' + str(count_kfold) + '.png'),
                              bbox_inches='tight')
            hist_fig2.clf()

    end = datetime.datetime.now()


    count_kfold += 1

print('==============================================================================')
print('==============================================================================\r',
      file=log_text)

print('##Parameters##')
print('GPU: ' + str(os.environ['CUDA_VISIBLE_DEVICES']))
print('learning_rate: ' + str(learning_rate))
print('Optimizer: ' + str(hyper_parameter_kfold.opt_name))
print('The number of epochs: ' + str(epochs))
print('batch_size: ' + str(batch_size))
print("Finshied Train (%s). \n" % datetime.datetime.now())
print('\n')
print('##Parameters##')
print('GPU: ' + str(os.environ['CUDA_VISIBLE_DEVICES']) + '\r', file=log_text)
print('learning_rate: ' + str(learning_rate) + '\r', file=log_text)
print('Optimizer: ' + str(hyper_parameter_kfold.opt_name) + '\r', file=log_text)
print('The number of epochs: ' + str(epochs) + '\r', file=log_text)
print('batch_size: ' + str(batch_size) + '\r', file=log_text)
print("Finshied Train (%s). \n" % datetime.datetime.now() + '\r', file=log_text)

print('Done')