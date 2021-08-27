
"""
File name: data loader for k-fold
Author: Jonghak Moon

"""

import numpy as np
import os
import pickle
import shutil
# import  batch_loader_frangi_filter_0detect
import datetime
import matplotlib

matplotlib.use('agg')
import hyper_parameter_kfold
import random
import scipy
from sklearn.model_selection import StratifiedKFold
import augment

def pause():
    input('press the enter key to continue...')


resize_option = hyper_parameter_kfold.resize_option
logs_dir_path = hyper_parameter_kfold.logs_dir_path
resize_logs_dir_path = hyper_parameter_kfold.resize_logs_dir_path
log_txt_name = hyper_parameter_kfold.log_txt_name
number_of_fold = hyper_parameter_kfold.number_of_fold
augmentation_option = hyper_parameter_kfold.augmentation_option
train_mode = hyper_parameter_kfold.training_mode

""""""""""""""""""""""""""""""""""""""""
              Aug Option
"""""""""""""""""""""""""""""""""""""""""
train_control_intensity_option = hyper_parameter_kfold.train_control_intensity_option
train_cropping_option = hyper_parameter_kfold.train_cropping_option
train_adjust_brightness_option = hyper_parameter_kfold.train_adjust_brightness_option

mean = {'mean': True}
"""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""
if resize_option['resize'] == True:

    ### Data Pickling and Loading ###
    log_text = open(os.path.join(resize_logs_dir_path, log_txt_name), 'a')
else:
    ### Data Pickling and Loading ###
    log_text = open(os.path.join(logs_dir_path, log_txt_name), 'a')

# ---------------------------------------------#

train_acc = []
train_loss = []
valid_acc = []
test_acc = []
test_loss = []

if resize_option['resize'] == True:
    duplicated_list = os.listdir(hyper_parameter_kfold.fixed_dump_path)

    print('Pickle normal and abnormal data size for resized : %s' % len(duplicated_list))
    print('Pickle normal and abnormal data size for resized : %s' % len(duplicated_list), file=log_text)

    print('=== Pickling the image path and label ===')
    print('=== Pickling the image path and label === \r', file=log_text)

    time_start = datetime.datetime.now()
    print("     Training Start at", time_start)

    # print("list_all",list_all)
    "아래 random에서 normal, abnormal 피클 이름들을 랜덤하게 모두 섞는다"
    random.shuffle(duplicated_list)

    count = 0
    divide = []

    print("duplicated_list_train for 256 size", len(duplicated_list))
    # pause()
    batch = []
    batch_label = []
    for itr in range(0, len(duplicated_list)):

        with open((hyper_parameter_kfold.train_dump_path + '/' + str(duplicated_list[count])),
                  'rb') as f:
            train_normal_abnormal_list = pickle.load(f)
            np_array_values = train_normal_abnormal_list[0:-1]
            get_label = train_normal_abnormal_list[-1]

            for_bundle = []
            for_bundle_label = []
            for itr2 in np.array(np_array_values[0]):
                img = itr2.astype(np.float32)
                for_bundle.append(img)

            count += 1
        batch.append(for_bundle)
        batch_label.append(get_label)
    train = np.array(batch)
    train_label = np.array(batch_label)

else:
    print('Resize option is False.')
    print('Resize option is False.', file=log_text)

    duplicated_list = os.listdir(hyper_parameter_kfold.fixed_dump_path)
    # duplicated_list = os.listdir('/data/inception_v3/clip_kfold/new_ori_512_2')

    print('Pickle normal and abnormal data size for resized : %s' % len(duplicated_list))
    print('Pickle normal and abnormal data size for resized : %s' % len(duplicated_list), file=log_text)

    print('=== Pickling the image path and label ===')
    print('=== Pickling the image path and label === \r', file=log_text)
    time_start = datetime.datetime.now()
    print("     Training Start at", time_start)

    for_bundle_label = []
    for_bundle_valid = []
    for_bundle_label_valid = []
    count = 0
    divide = []
    batch = []
    batch_label = []
    count_for_image = 0
    batch_path = []
    batch_index = []
    count_ab = 0
    count_normal = 0

    for itr in range(0, len(duplicated_list)):

        with open((hyper_parameter_kfold.train_dump_path + '/' + str(duplicated_list[itr])),
                  'rb') as f:

            train_normal_abnormal_list = pickle.load(f)
            np_array_values = train_normal_abnormal_list[0]
            for_bundle = []
            for_bundle_label = []
            for itr2 in np.array(np_array_values):
                img = itr2.astype(np.float32)
                for_bundle.append(img)
            get_label = train_normal_abnormal_list[-3]
            path = [train_normal_abnormal_list[-2], train_normal_abnormal_list[-1]]
            if get_label == 0:
                count_ab += 1
            elif get_label == 1:
                count_normal += 1

            count += 1
        batch.append(for_bundle)
        batch_label.append(get_label)
        batch_path.append(path)

    all = np.array(batch)
    all_label = np.array(batch_label)
    batch_path = np.array(batch_path)

    kfold = StratifiedKFold(n_splits=number_of_fold)

    count_kfold = 1

    for train, test in kfold.split(all, all_label, batch_path):

        train_array = np.array(all)[train]
        train_label = np.array(all_label)[train]
        train_path = np.array(batch_path)[train]
        # train_index = np.array(batch_index)[train]
        test_array = np.array(all)[test]
        test_label = np.array(all_label)[test]
        test_path = np.array(batch_path)[test]
        array1 = []
        new_label = []
        new_path = []
        mean_image = 0
        new_index = []
        for itr in range(0, train_array.shape[0]):
            for itr2 in range(0, train_array[itr].shape[0]):
                a = train_array[itr][itr2]
                b = train_label[itr]
                new_train_path = train_path[itr][0]
                new_train_index = train_path[itr][1]
                mean_image += a
                img = a
                array1.append(img)
                new_label.append(b)
                new_path.append(new_train_path)
                new_index.append(new_train_index)
        mean_image /= len(array1)

        array2 = []
        new_label2 = []
        new_path2 = []
        new_index2 = []
        for itr3 in range(0, test_array.shape[0]):
            for itr4 in range(0, test_array[itr3].shape[0]):
                c = test_array[itr3][itr4]
                d = test_label[itr3]
                new_test_path = test_path[itr3][0]
                new_test_index = test_path[itr3][1]
                img2 = c

                array2.append(img2)
                new_label2.append(d)
                new_path2.append(new_test_path)
                new_index2.append(new_test_index)

        if augmentation_option['augmentation'] == True:
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

            sec_mean_image = 0
            for ever in range(multiply_by_aug):
                if augmentation_option['augmentation'] == True:
                    aug_train_array, aug_train_label, aug_train_path, aug_train_index = augment.cag_aug_method_combination(array1,new_label, new_path, new_index, control_intensity_option=train_control_intensity_option,train_cropping_option=train_cropping_option)
                    pickle_file_path = os.path.join('/data/inception_v3/clip_kfold/' + str(count_kfold) + 'fold/' + str(
                                multiply_by_aug) + str(train_mode) + '/train')

                    sec_mean_image += mean_image
                    num = 0

                    for iter in range(int(np.array(aug_train_array).shape[0]/5)):
                        with open((pickle_file_path +'/'+ str(ever) + '_' + str(iter) + 'augmented_train_clip' + '.pickle'), 'wb') as f:
                            pickle.dump([aug_train_array[num:num+5], aug_train_label[num:num+5], aug_train_path[num:num+5], aug_train_index[num:num+5]], f)
                        num += 5


            pickle_file_path = os.path.join('/data/inception_v3/clip_kfold/' + str(count_kfold) + 'fold/' + str(
                        multiply_by_aug) + str(train_mode) + '/valid')

            sec_mean_image /= int(multiply_by_aug)
            valid_crop = [img[6:506, 6:506] for img in np.array(array2)]

            if os.path.exists(pickle_file_path):
                print('There is no .pickle file. \n')
                with open((pickle_file_path +'/'+ 'validation_clip' + '.pickle'), 'wb') as f:
                    pickle.dump([valid_crop, new_label2, new_path2, new_index2], f)
                with open((pickle_file_path +'/'+ 'mean_image' + '.pickle'), 'wb') as f:
                    pickle.dump([sec_mean_image], f)
            else:
                print('Saved files are exists! I will use this .pickle \n')


        elif augmentation_option['augmentation'] == False:

            train_pickle_file_path = os.path.join(
            '/data/inception_v3/clip_kfold/' + str(count_kfold) + 'fold/' +'new_'+ str(train_mode) + '/train')
            valid_pickle_file_path = os.path.join(
            '/data/inception_v3/clip_kfold/' + str(count_kfold) + 'fold/' +'new_'+ str(train_mode) + '/valid')

            with open((train_pickle_file_path + '/' + 'original_train_clip.pickle'), 'wb') as f:
                pickle.dump([array1,new_label, new_path, new_index], f)

            with open((valid_pickle_file_path + '/' + 'validation_clip.pickle'), 'wb') as f:
                pickle.dump([array2,new_label2, new_path2, new_index2], f)

            with open((valid_pickle_file_path +'/'+ 'mean_image' + '.pickle'), 'wb') as f:
                pickle.dump([mean_image], f)


        count_kfold += 1