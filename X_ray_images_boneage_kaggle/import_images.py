"""
This tool is developed for the study of deep learning with tensorflow-gpu on the
project of bone age and gender predictions of X-ray images by Alvin Pei-Yan, Li.
Please contact him, d05548014@ntu.edu.tw / Alvin.Li@acer.com / a4624393@gmail.com,
for futher authorizatoin of the use of this tool.
"""
from __future__ import absolute_import, division, print_function
from PIL import Image
from random import shuffle

import csv
import os
import math

import numpy as np
import tensorflow as tf

#tf.enable_eager_execution()

####################################################################################

"Global Variables"

#Boneage_range = 228
dir_datasheet = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset.csv'
dir_dataset = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset'

num_classes = 2
img_iso_resolution = 500
ratio_of_separate = 0.8 # data of 80% for training, the rest for evaluation

####################################################################################

def get_inputs(flag_usage, FLAGS):

    datasheet = get_datasheet()
    num_example_per_epoch_for_train = math.ceil(len(datasheet) * ratio_of_separate)
    num_example_per_epoch_for_eval = len(datasheet) - num_example_per_epoch_for_train

    index_card = [i for i in range(0, len(datasheet))]

    filenames, labels, num_examples_per_epoch = get_filenames(flag_usage)

    with tf.name_scope('get_inputs'):
        filenames = tf.convert_to_tensor(filenames, dtype=tf.string)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        queue_image, queue_label = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)

        queue_image = tf.read_file(queue_image)
        queue_image = tf.image.decode_image(queue_image, channels = 1)

        queue_image = tf.image.resize_images(queue_image,
                                             size = [img_iso_resolution, img_iso_resolution],
                                             preserve_aspect_ratio = True
                                             )

        input_images, input_labels = tf.train.batch(
                            [queue_image, queue_label],
                            batch_size = FLAGS.batch_size,
                            num_threads = 4,
                            capacity = batch_size + int(num_examples_per_epoch*0.4)
                            )

    return input_images, input_labels

    def get_datasheet():

        try:
            print("Loading the datasheet from {}.".format(dir_datasheet))
            datasheet = list(csv.reader(open(dir_datasheet), delimiter=','))
            datasheet.pop(0)
            datasheet = datasheet_cleaning(datasheet)
            print("Got it!\n")

        except Exception as err:
            print("Here occured an error: {}\n".format(err))

        return datasheet

    def datasheet_cleaning(datasheet):

        print("Cleaning the data to remove the rows with blank or NaN elements")

        for index in range(0, len(datasheet)):
            for member in datasheet[index]:
                if member is "" or member == 'NaN':
                    datasheet.pop[index]

        return datasheet

    def get_filenames(flag_usage):

        if flag_usage is 'train':
            training_indices = index_card[:num_example_per_epoch_for_train]
            filenames = []
            labels = []
            for index in training_indices:
                filenames.append(dir_dataset + '\\' + datasheet[index][0] + '.png')
                if datasheet[index][2] is 'False':
                    labels.append(0)
                else:
                    labels.append(1)
            num_examples_per_epoch = num_example_per_epoch_for_train

        elif flag_usage is 'eval':
            testing_indices = index_card[-num_example_per_epoch_for_eval:]
            filenames = []
            labels = []
            for index in testing_indices:
                filenames.append(dir_dataset + '\\' + datasheet[index][0] + '.png')
                if datasheet[index][2] is 'False':
                    labels.append(0)
                else:
                    labels.append(1)
            num_examples_per_epoch = num_example_per_epoch_for_eval

        return filenames, labels, num_examples_per_epoch
