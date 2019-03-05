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
import matplotlib.pyplot as plt
import math

import numpy as np
import tensorflow as tf

#tf.enable_eager_execution()

"Global Variables"

global dir_datasheet
global dir_dataset
global num_classes # male or female, 2 classes
global num_example_per_epoch_for_train
global num_example_per_epoch_for_eval
#global Boneage_range = 228

"Function Definitions"

def main():
    """

    """
    print("Start.\n")

    datasheet = load_datasheet()
    deeplearning_model(datasheet)

    print("\nDone.")

def load_datasheet():
    """
    This function load the .csv file recorded the information of training data.
    """

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

def deeplearning_model(datasheet):
    """
    Model inputs, Model prediction, Model training
    # This model do not rely on multiple GPUs training.
    # This model do not distort the training images.
    """
    print("\n *Log in the deep learning model based on tensorflow frameworks ")

    "# Model inputes #"

    print("\n *Import and parse the data set\n ")

    _init_ratio_of_separate = 0.8
    training_features, testing_features, training_inputs, testing_inputs \
                        = parse_the_data(datasheet, _init_ratio_of_separate)

    print("\n *Initialize features columns\n ")

    feature_initializer()

    "# Model prediction #"

    print("\n *Setting model parameters basically:\n ")

    FLAGS, num_example_per_epoch_for_train, num_example_per_epoch_for_eval \
                        = set_model_parameters(datasheet, _init_ratio_of_separate)

    moving_ave_decay, num_epochs_per_decay, Learning_rate_decay_factor \
                        , initital_learning_rate = set_training_process_parameters()

    img_iso_resolution = 500
    print("The size of resized images: {}".format((img_iso_resolution, img_iso_resolution)))

    " select the type of model: define data, define model, define loss "

    #tensor_img = image_resizing(image_index, datasheet, img_iso_resolution)

    "# Model training #"

    " train the model "

    " evaluate the model's effectiveness "

    " use the trained model to make predictions "


def parse_the_data(datasheet, ratio_of_separate):

    training_indices, testing_indices = shuffle_data(datasheet, ratio_of_separate)
    training_features = get_training_feature_dict(training_indices, datasheet)
    testing_features = get_testing_feature_dict(testing_indices, datasheet)

    gender_column = tf.feature_column.categorical_column_with_vocabulary_list('Gender', ['M', 'F'])
    gender_column = tf.feature_column.indicator_column(gender_column)

    columns = [
        tf.feature_column.numeric_column('Boneage'),
        gender_column
    ]

    training_inputs = tf.feature_column.input_layer(training_features, columns)
    testing_inputs = tf.feature_column.input_layer(testing_features, columns)

    return training_features, testing_features, training_inputs, testing_inputs

def shuffle_data(datasheet, ratio_of_separate):
    """
    shuffle the data and separate it to the training set and testing/validation set.
    # (ratio_of_separate)% for training, and the rest for test/validation.
    """
    shuffle_card = [i for i in range(0, len(datasheet))]
    shuffle(shuffle_card)
    training_amount = math.ceil(len(shuffle_card) * ratio_of_separate)
    testing_amount = len(shuffle_card) - training_amount

    training_indices = shuffle_card[:training_amount]
    if testing_amount == 0:
        testing_indices = []
    else:
        testing_indices = shuffle_card[-testing_amount:]

    return training_indices, testing_indices

def get_training_feature_dict(training_indices, datasheet):

    Gender_values = []
    Boneage_values = []
    for index in training_indices:
        if datasheet[index][2] is 'False':
            gender = 'F'
            Gender_values.append(gender)
        else:
            gender = 'M'
            Gender_values.append(gender)

        boneage = [int(datasheet[index][1])]
        Boneage_values.append(boneage)

    training_features = {}
    training_features['Boneage'] = Boneage_values
    training_features['Gender'] = Gender_values

    return training_features

def get_testing_feature_dict(testing_indices, datasheet):

    Gender_values = []
    Boneage_values = []
    for index in testing_indices:
        if datasheet[index][2] is 'False':
            gender = 'F'
            Gender_values.append(gender)
        else:
            gender = 'M'
            Gender_values.append(gender)

        boneage = [int(datasheet[index][1])]
        Boneage_values.append(boneage)

    testing_features = {}
    testing_features['Boneage'] = Boneage_values
    testing_features['Gender'] = Gender_values

    return testing_features

def image_resizing(image_index, datasheet, img_iso_resolution):
    """
    resize image with same aspect ratio and return that tensor
    """
    open_image_dir = dir_dataset + '\\' + datasheet[image_index][0] + '.png'

    try:
        img = Image.open(open_image_dir)

    except Exception as err:
        print('An error occured trying to read {}.'.format(open_image_dir))
        print("Error: {}\n".format(err))

    resize_dimention = (img_iso_resolution, img_iso_resolution)
    img.thumbnail(resize_dimention)
    img = np.asarray(img)
    (width, height) = img.shape
    new_img = np.zeros((resize_dimention))

    if width < img_iso_resolution:
        shift_start = math.ceil( (img_iso_resolution - width) / 2 )
        shift_end = shift_start + width
        new_img[shift_start:shift_end, :] = img

    elif height < img_iso_resolution:
        shift_start = math.ceil( (img_iso_resolution - height) / 2 )
        shift_end = shift_start + height
        new_img[:, shift_start:shift_end] = img

    return tf.convert_to_tensor(img)

def feature_initializer():
    """
    Initialize features columns
    """

    var_init = tf.global_variables_initializer()
    table_init = tf.tables_initializer()

    sess = tf.Session()
    sess.run((var_init, table_init))

def set_model_parameters(datasheet, ratio_of_separate):
    """
    Setting basic parameters of the deep learning model
    """

    FLAGS = tf.app.flags.FLAGS
    #Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
    tf.app.flags.DEFINE_integer('batch_size', 128,
                                """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_string('dir_dataset', dir_dataset,
                               """Path to the CIFAR-10 data directory.""")
    tf.app.flags.DEFINE_boolean('use_fp16', False,
                                """Train the model using fp16.""")

    amount_img_pool = math.ceil(len(datasheet) * ratio_of_separate)
    num_example_per_epoch_for_train = math.ceil(amount_img_pool * ratio_of_separate)
    num_example_per_epoch_for_eval = amount_img_pool - num_example_per_epoch_for_train

    print("Batch size: {}".format(FLAGS.batch_size))
    print("The number of training images per epoch: {}".format(num_example_per_epoch_for_train))
    print("The number of evaluation images per epoch: {}".format(num_example_per_epoch_for_eval))

    return FLAGS, num_example_per_epoch_for_train, num_example_per_epoch_for_eval

def set_training_process_parameters():
    """
    Setting training process parameters of the deep learning model
    """

    # The decay to use for the moving average.
    moving_ave_decay = 0.9999
    # Epochs after which learning rate decays.
    num_epochs_per_decay = 350.0
    # Learning rate decay factor.
    Learning_rate_decay_factor = 0.1
    # Initial learning rate.
    initital_learning_rate = 0.1

    print("The decay to use for the moving average: {}".format(moving_ave_decay))
    print("Epochs after which learning rate decays: {}".format(num_epochs_per_decay))
    print("Learning rate decay factor: {}".format(Learning_rate_decay_factor))
    print("Initial learning rate: {}".format(initital_learning_rate))

    return moving_ave_decay, num_epochs_per_decay, Learning_rate_decay_factor, initital_learning_rate

"Execution"

import time
if __name__ == '__main__':

    dir_datasheet = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset.csv'
    dir_dataset = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset'

    start = time. time()

    main()

    end = time. time()

    duration = end - start

    print('\nThis code runs so fast that only spends {} in second.'.format(duration))
