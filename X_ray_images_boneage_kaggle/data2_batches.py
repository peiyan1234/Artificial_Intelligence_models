from PIL import Image
from random import shuffle

import os
import csv
import math

dir_datasheet = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset.csv'
dir_dataset = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset'

IMAGE_SIZE = 500

goback_dir = os.getcwd()
os.chdir(dir_dataset)

dir_batches = os.path.join(dir_dataset, 'batches')
dir_train_batch = os.path.join(dir_batches, 'data_batch')
dir_test_batch = os.path.join(dir_batches, 'test_batch')

os.mkdir(dir_savingfiles)
os.mkdir(dir_train_batch)
os.mkdir(dir_test_batch)

datasheet = get_datasheet()
index_card = [i for i in range(0, len(datasheet))]
shuffle(index_card)

train_filenames = []
training_indices = index_card[:num_example_per_epoch_for_train]
for index in training_indices:
    filename = dir_dataset + '\\' + datasheet[index][0] + '.png'
    original_im = Image.open(filename)

    

    savingdir = dir_train_batch + '\\' + datasheet[index][0] + '.png'
    resized_image.save(savingdir)
    #train_filenames.append(filename)


test_filenames = []
testing_indices = index_card[-num_example_per_epoch_for_eval:]
for index in testing_indices:
    filename = dir_dataset + '\\' + datasheet[index][0] + '.png'
    original_im = Image.open(filename)

    savingdir = dir_test_batch + '\\' + datasheet[index][0] + '.png'
    resized_image.save(savingdir)
    #test_filenames.append(dir_dataset + '\\' + datasheet[index][0] + '.png')

os.chdir(goback_dir)

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
