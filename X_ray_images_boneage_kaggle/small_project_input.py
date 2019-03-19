from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import math
import glob

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

dir_datasheet = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset.csv'

IMAGE_SIZE = 500

NUM_CLASSES = 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10089
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2522

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for small_project training using the Reader ops.

  Args:
    data_dir: Path to the small_project data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = []
  for filename in glob.glob(data_dir + '\\' + '*.png'):
      filenames.append(filename)

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  datasheet = list(csv.reader(open(dir_datasheet), delimiter=','))
  datasheet.pop(0)
  for index in range(0, len(datasheet)):
      for member in datasheet[index]:
          if member is "" or member == 'NaN':
              datasheet.pop[index]

  labels_dic = {}
  for index in range(0, len(datasheet)):
      if datasheet[index][2] == 'False':
          key = datasheet[index][0] + '.png'
          labels_dic[key] = 0
      elif datasheet[index][2] == 'True':
          key = datasheet[index][0] + '.png'
          labels_dic[key] = 1

  labels = []
  goback_dir = os.getcwd()
  os.chdir(data_dir)
  for image_name in glob.glob('*.png'):
      genderlabel = labels_dic[str(image_name)]
      labels.append(genderlabel)
  os.chdir(goback_dir)
  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_data(filename_queue, labels)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    #distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # NOTE: since per_image_standardization zeros the mean and makes
    # the stddev unit, this likely has no effect see tensorflow#1458.
    #distorted_image = tf.image.random_brightness(distorted_image,
    #                                             max_delta=63)
    #distorted_image = tf.image.random_contrast(distorted_image,
    #                                           lower=0.2, upper=1.8)

    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(distorted_image)

    # Set the shapes of tensors.
    #float_image.set_shape([height, width, 3])
    #reshaped_image.set_shape([height, width, 1])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d small_project images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  #return _generate_image_and_label_batch(float_image, read_input.label,
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=True)
    return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                          min_queue_examples, batch_size,
                                          shuffle=True)

def inputs(eval_data, data_dir, batch_size):
  """Construct input for small_project evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the small_project data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = []
  for filename in glob.glob(data_dir + '\\' + '*.png'):
      filenames.append(filename)

  if not eval_data:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  else:
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  datasheet = list(csv.reader(open(dir_datasheet), delimiter=','))
  datasheet.pop(0)
  for index in range(0, len(datasheet)):
      for member in datasheet[index]:
          if member is "" or member == 'NaN':
              datasheet.pop[index]

  labels_dic = {}
  for index in range(0, len(datasheet)):
      if datasheet[index][2] == 'False':
          key = datasheet[index][0] + '.png'
          labels_dic[key] = 0
      elif datasheet[index][2] == 'True':
          key = datasheet[index][0] + '.png'
          labels_dic[key] = 1

  labels = []
  goback_dir = os.getcwd()
  os.chdir(data_dir)
  for image_name in glob.glob('*.png'):
      genderlabel = labels_dic[str(image_name)]
      labels.append(genderlabel)
  os.chdir(goback_dir)

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_data(filename_queue, labels)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,
    #                                                       height, width)

    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(resized_image)

    # Set the shapes of tensors.
    #float_image.set_shape([height, width, 3])
    #reshaped_image.set_shape([height, width, 1])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  #return _generate_image_and_label_batch(float_image, read_input.label,
    #                                     min_queue_examples, batch_size,
    #                                     shuffle=False)
    return _generate_image_and_label_batch(reshaped_image, read_input.label,
                                          min_queue_examples, batch_size,
                                          shuffle=True)

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 1] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 1] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def read_data(filename_queue, labels):
  """Reads and parses examples from small_project data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (500)
      width: number of columns in the result (500)
      depth: number of color channels in the result (1)
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class dataRecord(object):
    pass
  result = dataRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  result.height = 500
  result.width = 500
  result.depth = 1

  labels = tf.convert_to_tensor(labels, dtype=tf.int32)
  image, label = tf.train.slice_input_producer([filename_queue, labels],
                                                 shuffle=False)
  image = tf.read_file(image)
  image = tf.image.decode_image(image)
  image /= 255.0  # normalize to [0,1] range
  result.uint8image = image

  return result
