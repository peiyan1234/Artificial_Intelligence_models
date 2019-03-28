from PIL import Image
from random import shuffle

import os
import csv
import math
import time

import numpy as np

dir_datasheet = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset.csv'
dir_dataset = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset'

IMAGE_SIZE = 256

dir_batches = os.path.join(dir_dataset, 'batches')
dir_train_batch = os.path.join(dir_batches, 'data_batch')
dir_test_batch = os.path.join(dir_batches, 'test_batch')

ratio_of_separate = 0.8

def main():
    os.mkdir(dir_batches)
    datasheet = get_datasheet()
    index_card = [i for i in range(0, len(datasheet))]
    shuffle(index_card)

    print("Separate data for the model training")
    get_train_batch(index_card, datasheet)
    print("Separate data for the model evaluation")
    get_eval_batch(index_card, datasheet)

    print("\nDone")

def get_datasheet():

    try:
        print("Loading the datasheet from {}.".format(dir_datasheet))
        datasheet = list(csv.reader(open(dir_datasheet), delimiter=','))
        datasheet.pop(0)
        print("Got it!\n")
        print("Cleaning the data to remove the rows with blank or NaN elements")

        for index in range(0, len(datasheet)):
            for member in datasheet[index]:
                if member is "" or member == 'NaN':
                    datasheet.pop[index]
        return datasheet

    except Exception as err:
        print("Here occured an error: {}\n".format(err))

def get_train_batch(index_card, datasheet):
    os.mkdir(dir_train_batch)
    num_example_per_epoch_for_train = math.ceil(len(datasheet) * ratio_of_separate)

    train_filenames = []
    training_indices = index_card[:num_example_per_epoch_for_train]
    for index in training_indices:
        filename = dir_dataset + '\\' + datasheet[index][0] + '.png'

        try:
            original_im = Image.open(filename)
            print("Read the image from {}".format(filename))
        except Exception as err:
            print('An error occured trying to read {}.'.format(filename))
            print("Error: {}\n".format(err))

        print("Resize this image with the original apsect ration preserved")
        resize_dimention = (IMAGE_SIZE, IMAGE_SIZE)
        original_im.thumbnail(resize_dimention)
        resized_image = np.asarray(original_im)
        (width, height) = resized_image.shape
        new_img = np.zeros((resize_dimention))

        if width < IMAGE_SIZE:
            shift_start = math.ceil( (IMAGE_SIZE - width) / 2 )
            shift_end = shift_start + width
            new_img[shift_start:shift_end, :] = resized_image
            #new_img = new_img / 255.0

        elif height < IMAGE_SIZE:
            shift_start = math.ceil( (IMAGE_SIZE - height) / 2 )
            shift_end = shift_start + height
            new_img[:, shift_start:shift_end] = resized_image
            #new_img = new_img / 255.0

        savingdir = dir_train_batch + '\\' + datasheet[index][0] + '.png'
        new_img = Image.fromarray(np.uint8(new_img))
        print("Save the image to {}".format(savingdir))
        new_img.save(savingdir)
        #train_filenames.append(filename)

def get_eval_batch(index_card, datasheet):
    os.mkdir(dir_test_batch)
    num_example_per_epoch_for_eval = len(datasheet) -  math.ceil(len(datasheet) * ratio_of_separate)

    test_filenames = []
    testing_indices = index_card[-num_example_per_epoch_for_eval:]
    for index in testing_indices:
        filename = dir_dataset + '\\' + datasheet[index][0] + '.png'

        try:
            original_im = Image.open(filename)
            print("Read the image from {}".format(filename))

        except Exception as err:
            print('An error occured trying to read {}.'.format(filename))
            print("Error: {}\n".format(err))

        print("Resize this image with the original apsect ration preserved")
        resize_dimention = (IMAGE_SIZE, IMAGE_SIZE)
        original_im.thumbnail(resize_dimention)
        resized_image = np.asarray(original_im)
        (width, height) = resized_image.shape
        new_img = np.zeros((resize_dimention))

        if width < IMAGE_SIZE:
            shift_start = math.ceil( (IMAGE_SIZE - width) / 2 )
            shift_end = shift_start + width
            new_img[shift_start:shift_end, :] = resized_image
            #new_img = new_img / 255.0

        elif height < IMAGE_SIZE:
            shift_start = math.ceil( (IMAGE_SIZE - height) / 2 )
            shift_end = shift_start + height
            new_img[:, shift_start:shift_end] = resized_image
            #new_img = new_img / 255.0

        savingdir = dir_test_batch + '\\' + datasheet[index][0] + '.png'
        new_img = Image.fromarray(np.uint8(new_img))
        print("Save the image to {}".format(savingdir))
        new_img.save(savingdir)
        #test_filenames.append(dir_dataset + '\\' + datasheet[index][0] + '.png')

if __name__ == '__main__':
    start = time. time()
    main()
    end = time. time()

    duration = end - start
    print('\nThis code runs so fast that only spends {} in second.'.format(duration))
