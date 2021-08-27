

"""
File name: original data pickle file generator.py
Author: Jonghak Moon

"""

from skimage.filters import threshold_otsu, frangi
import numpy as np
import os
import re
import scipy.misc
import pickle
import SimpleITK as sitk
import hyper_parameter_kfold
from skimage.morphology import black_tophat, white_tophat
from skimage.morphology import disk
import matplotlib
matplotlib.use('agg')
import time

def pause():
    input('press the enter key to continue...')

filename = '/data/inception_v3/clip_kfold'
threshold = 0
get_size = hyper_parameter_kfold.get_size
position = 'data_all'

resize_option = hyper_parameter_kfold.resize_option
data_size = hyper_parameter_kfold.data_size
number_of_fold = hyper_parameter_kfold.number_of_fold

picture_save_normal = '/data/inception_v3/clip_kfold/per_patient_picture'
picture_save_abnormal = '/data/inception_v3/clip_kfold/per_patient_picture'

resize_normal = picture_save_normal + '/resize'
resize_abnormal = picture_save_abnormal + '/resize'

abnormal_pickle_save_path = '/data/inception_v3/clip_kfold/per_patient_pickle2'
normal_pickle_save_path = '/data/inception_v3/clip_kfold/per_patient_pickle2'

pickle_resize_normal = '/data/inception_v3/clip_kfold/ori_256'
pickle_resize_abnormal = '/data/inception_v3/clip_kfold/ori_256'

density = 10
date = 10.10

pickle_numb = 165

divide = 'whole'

selem = disk(density)
fire_name = str(date) +'_'+ str(position) +'_'+ str(density) + str(divide) +'1of3_updated_'


def file_path_list(filename):
    Normal = []
    Abnormal = []
    # file_path_list=[]

    for Par_dir, Sub_name, File_name in os.walk(filename):

        if File_name:
            for itr in range(len(File_name)):
                file_path = os.path.join(Par_dir, File_name[itr])
                # print(file_path)

                if position in file_path.split('/'):
                    print("file_path", file_path)

                    if 'N' in file_path.split('/'):
                        Normal.append(file_path)

                    elif 'A' in file_path.split('/'):
                        Abnormal.append(file_path)

    return Normal, Abnormal


def training_batch(Normal, Abnormal):

    print("len(Normal)",len(Normal))
    print("len(Abnormal)",len(Abnormal))

    count = 0
    count_abnormal = 0

    max_index1 = []
    max_index2 = []

    # if divide == 'Normal_1' or 'Normal_2':
    for itr in range(len(Normal)):
        print("\n")
        print("NOW 'right normal clip'")
        ds_normal_train = sitk.ReadImage(Normal[itr])
        dcm_pixel_normal = sitk.GetArrayFromImage(ds_normal_train)
        print("I will train this file!!! = ", Normal[itr])
        print("shape of dcm pixel ", dcm_pixel_normal.shape)

        save_wpc = []
        count += 1
        print("training normal count", count)
        start = time.time()
        print("key fame detection start")

        for itr1 in range(len(dcm_pixel_normal)):

            image = dcm_pixel_normal[itr1]
            w_tophat = white_tophat(image, selem)
            b_tophat = black_tophat(image, selem)

            for a in range(b_tophat.shape[0]):
                for b in range(b_tophat.shape[1]):
                    if b_tophat[a][b] > image[a][b]:
                        b_tophat[a][b] = image[a][b]
            this_way = image + w_tophat - b_tophat

            this_way[0:160, 310:512] = 128

            frangi_image = frangi(this_way)
            frangi_thresh = threshold_otsu(frangi_image)
            frangi_binary = (frangi_image > frangi_thresh)
            wpc_thresh = np.unique(frangi_binary, return_counts=True)
            save_wpc.append(wpc_thresh[1][1])

        index = len(save_wpc) / 3
        index2 = (2 * len(save_wpc)) / 3
        max_value_save_wpc = max(save_wpc[int(index):int(index2)])
        max_value_index = save_wpc.index(max_value_save_wpc)

        print("max_value_save_wpc",max_value_save_wpc)
        print("max_value_index",max_value_index)
        max_index1.append(max_value_index)

        end = time.time()

    for itr in range(len(Abnormal)):
        print("\n")
        print('training'and 'abnormal_CAG' in re.split('[/]', Abnormal[itr]))
        print("NOW 'training'and 'abnormal_CAG' in re.split('[/]', train_file_path_list[itr])")
        ds_abnormal_train = sitk.ReadImage(Abnormal[itr])
        dcm_pixel_abnormal = sitk.GetArrayFromImage(ds_abnormal_train)

        print("I will train this file!!! = ", Abnormal[itr])
        print("shape of dcm pixel ", dcm_pixel_abnormal.shape)

        save_wpc_abnormal = []
        count_abnormal += 1
        print("training abnormal_count", count_abnormal)
        start = time.time()
        print("key fame detection start")
        for itr2 in range(len(dcm_pixel_abnormal)):
            itr_abormal = dcm_pixel_abnormal[itr2]
            w_tophat = white_tophat(itr_abormal, selem)
            b_tophat = black_tophat(itr_abormal, selem)
            for a in range(b_tophat.shape[0]):
                for b in range(b_tophat.shape[1]):
                    if b_tophat[a][b] > itr_abormal[a][b]:
                        b_tophat[a][b] = itr_abormal[a][b]

            this_way = itr_abormal + w_tophat - b_tophat
            this_way[0:160, 310:512] = 128

            frangi_image = frangi(this_way)
            frangi_thresh = threshold_otsu(frangi_image)
            frangi_binary = (frangi_image > frangi_thresh)
            wpc_abnormal = np.unique(frangi_binary, return_counts=True)
            save_wpc_abnormal.append(wpc_abnormal[1][1])

        ab_index = len(save_wpc_abnormal) / 3
        ab_index2 = (2 * len(save_wpc_abnormal)) / 3
        max_value_save_wpc_abnormal = max(save_wpc_abnormal[int(ab_index):int(ab_index2)])
        max_value_index_abnormal = save_wpc_abnormal.index(max_value_save_wpc_abnormal)

        print("max_value_save_wpc", max_value_save_wpc_abnormal)
        print("max_value_index", max_value_index_abnormal)
        max_index2.append(max_value_index_abnormal)
        end = time.time()

    print("max_index1",max_index1)
    print("max_index2",max_index2)

    np.save('/data/inception_v3/train_index/'+str(fire_name)+'index1.npy', max_index1)
    np.save('/data/inception_v3/train_index/'+str(fire_name)+'index2.npy', max_index2)

    return max_index1, max_index2

def load_peak_frame(Normal, Abnormal):
    pickle_name = int(pickle_numb)
    i = 0
    j = 0
    training_normal = np.load('/data/inception_v3/train_index/' + str(fire_name) + 'index1.npy')
    training_abnormal = np.load('/data/inception_v3/train_index/' + str(fire_name) + 'index2.npy')

    print("training_normal", training_normal)
    print("training_abnormal", training_abnormal)

    for itr in range(len(Normal)):

        print("Normal[itr]", Normal[itr])
        ds_normal_train = sitk.ReadImage(Normal[itr])
        dcm_pixel_normal = sitk.GetArrayFromImage(ds_normal_train)

        print("itr", itr)
        print("normal")

        pickle_normal_list = []
        resize_pickle_normal_list = []
        array_save = ('anything' + '_clip_' + 'train.pickle')
        # pause()

        for get in range(get_size):
            print("peak_frame_count", get)
            if len(dcm_pixel_normal) < (training_normal[i] + int(get_size)):
                peak_frame = dcm_pixel_normal[training_normal[i] - get]
                print("changed index", training_normal[i] - get)

            else:
                peak_frame = dcm_pixel_normal[training_normal[i] + get - int(np.floor(get_size / 2))]
                pickle_normal_list.append(peak_frame)
                print("index ", training_normal[i] + get - int(np.floor(get_size / 2)))
        print("shape resize_pickle_normal", np.array(resize_pickle_normal_list).shape)
        print("shape pickle_normal", np.array(pickle_normal_list).shape)
        print("pickle_normal_path", Normal[itr])

        if resize_option['resize'] == True:

            pickle_file_path = os.path.join(pickle_resize_normal, array_save)
            if not os.path.exists(pickle_file_path):
                print('There is no .pickle file. \n')
                with open((pickle_resize_normal + '/' + str(pickle_name) + '_clip_' + 'normal.pickle'),
                          'wb') as f:
                    pickle.dump([resize_pickle_normal_list, 0, Normal[itr], training_normal[i]], f)
            else:
                print('Saved files are exists! I will use this .pickle \n')

        else:

            pickle_file_path = os.path.join(normal_pickle_save_path, array_save)

            if not os.path.exists(pickle_file_path):
                print('There is no .pickle file. \n')
                with open((normal_pickle_save_path + '/' + str(pickle_name) + '_clip_' + 'normal.pickle'), 'wb') as f:
                    pickle.dump([pickle_normal_list, 0, Normal[itr], training_normal[i]], f)
            else:
                print('Saved files are exists! I will use this .pickle \n')

        i += 1
        pickle_name += 1

    for itr in range(len(Abnormal)):

        ds_abnormal_train = sitk.ReadImage(Abnormal[itr])
        dcm_pixel_abnormal = sitk.GetArrayFromImage(ds_abnormal_train)
        # print("itr",itr)

        array_save = ('anything' + '_clip_' + 'train.pickle')

        pickle_abnormal_list = []
        resize_pickle_abnormal_list = []
        # pickle_abnormal_path = []

        for get2 in range(get_size):
            print("peak_frame_count", get2)
            if len(dcm_pixel_abnormal) < (training_abnormal[j] + int(get_size)):
                peak_frame = dcm_pixel_abnormal[training_abnormal[j] - get2]
                print("changed index", training_abnormal[j] - get2)
            else:
                peak_frame = dcm_pixel_abnormal[training_abnormal[j] + get2 - int(np.floor(get_size / 2))]
                pickle_abnormal_list.append(peak_frame)
                print("index ", training_abnormal[j] + get2 - int(np.floor(get_size / 2)))


            if resize_option['resize'] == True:

                resize_img2 = scipy.misc.imresize(peak_frame, size=(data_size, data_size), interp='bilinear')
                scipy.misc.imsave(resize_abnormal + '/' + str(j) + '_' + str(get2) + 'resize_peak_frame.png',
                                  resize_img2)
                resize_pickle_abnormal = resize_img2
                resize_pickle_abnormal_list.append(resize_pickle_abnormal)
            else:
                scipy.misc.imsave(picture_save_abnormal + '/' + str(j) + '_' + str(get2) + 'abnormal_peak_frame.png',peak_frame)


        if resize_option['resize'] == True:

            pickle_file_path = os.path.join(pickle_resize_abnormal, array_save)

            if not os.path.exists(pickle_file_path):
                print('There is no .pickle file. \n')
                with open((pickle_resize_abnormal + '/' + str(pickle_name) + '_clip_' + 'abnormal.pickle'),
                          'wb') as f:
                    pickle.dump([resize_pickle_abnormal_list, 1, Abnormal[itr], training_abnormal[j]], f)
            else:
                print('Saved files are exists! I will use this .pickle \n')

        else:
            ab_pickle_file_path = os.path.join(abnormal_pickle_save_path, array_save)

            if not os.path.exists(ab_pickle_file_path):
                print('There is no .pickle file. \n')
                with open((abnormal_pickle_save_path + '/' + str(pickle_name) + '_clip_' + 'abnormal.pickle'), 'wb') as f:
                    pickle.dump([pickle_abnormal_list, 1, Abnormal[itr],training_abnormal[j]], f)
            else:
                print('Saved files are exists! I will use this .pickle \n')

        j += 1
        pickle_name += 1


if __name__ == '__main__':
    Normal, Abnormal = file_path_list(filename)
    print("Length Normal", len(Normal))
    print("Length Abnormal", len(Abnormal))
    #
    # training_normal_index, training_abnormal_index = training_batch(Normal, Abnormal)
    # #
    # load_peak_frame(Normal, Abnormal)
    # print("max_index1",a)

    # print("training_normal",training_normal)
    # print("training_abnormal",training_abnormal)
    # print("validation_normal",validation_normal)
    # print("validation_abnormal", validation_abnormal)