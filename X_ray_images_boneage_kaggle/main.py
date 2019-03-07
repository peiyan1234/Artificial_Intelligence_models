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
import time

import numpy as np
import tensorflow as tf

#tf.enable_eager_execution()

"Global Variables"

global dir_datasheet
global dir_dataset
global num_classes # male or female, 2 classes
global num_example_per_epoch_for_train
global num_example_per_epoch_for_eval
global num_epochs_per_decay
global initital_learning_rate
global Learning_rate_decay_factor
global moving_ave_decay

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
    def infterence(training_input_images):

        print("""\nDefine model:
            \n The architecture of networks \n
                layer 1: conv2d
                layer 2: max_pool
                layer 3: local_response_normalization
                layer 4: conv2d
                layer 5: local_response_normalization
                layer 6: max_pool
                layer 7: fully connected layer with rectified linear activation.
                layer 8: softmax_linear
        """)

        layer_1_conv_output = model_layer_1(training_input_images)
        layer_2_maxpool_output = model_layer_2(layer_1_conv_output)
        layer_3_local_response_normalization_output = model_layer_3(layer_2_maxpool_output)
        layer_4_conv_output = model_layer_4(layer_3_local_response_normalization_output)
        layer_5_local_response_normalization_output = model_layer_5(layer_4_conv_output)
        layer_6_maxpool_output = model_layer_6(layer_5_local_response_normalization_output)
        layer_7_FC_rectified_output = model_layer_7(layer_6_maxpool_output, tf.shape(training_input_images), FLAGS)
        layer_8_softmax_output = model_layer_8(layer_7_FC_rectified_output, num_classes = 2)

        return layer_8_softmax_output

    def loss_calculation(training_input_images, logits):

        print("""\n Define loss:
            \n L2 loss to all trainable variables\n
        """)

        total_loss = loss(logits, training_input_labels)

        return total_loss

    def train(training_input_images, training_input_labels, FLAGS):

        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()

            logits = infterence(training_input_images)
            total_loss = loss_calculation(training_input_images, logits)
            train_op = get_train_op(total_loss, FLAGS.batch_size, global_step)

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=FLAGS.train_dir,
                    hooks = [
                            tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                            tf.train.NanTensorHook(total_loss),
                            _LoggerHook()
                            ],
                    config = tf.ConfigProto(
                            log_device_placement = log_device_placement
                            )) as mon_sess:
                while not mon_sess.should_stop():
                    mon_sess.run(train_op)

            class _LoggerHook(tf.train.SessionRunHook):
                """Logs loss and runtime"""

                def begin(self):
                  self._step = -1
                  self._start_time = time.time()

                def before_run(self, run_context):
                  self._step += 1
                  return tf.train.SessionRunArgs(total_loss)

                def after_run(self, run_context, run_values):
                  if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print (format_str % (datetime.now(), self._step, loss_value,
                                         examples_per_sec, sec_per_batch))

    def run_train():
        if tf.gfile.Exists(FLAGS.dir_train):
            tf.gfile.DeleteRecursively(FLAGS.dir_train)
        tf.gfile.MakeDirs(FLAGS.train_dir)

        print("\n *Log in the deep learning model based on tensorflow frameworks ")

        print("\n *Setting model parameters basically:\n ")

        ratio_of_separate = 0.8

        FLAGS, num_example_per_epoch_for_train, num_example_per_epoch_for_eval \
                            = set_model_parameters(datasheet, ratio_of_separate)

        moving_ave_decay, num_epochs_per_decay, Learning_rate_decay_factor \
                            , initital_learning_rate = set_training_process_parameters()

        img_iso_resolution = 500

        print("The size of resized images: {}".format((img_iso_resolution, img_iso_resolution)))

        training_indices, testing_indices = shuffle_data(datasheet, ratio_of_separate)

        training_input_images, training_input_labels = get_training_inputs(
                                    batch_size = FLAGS.batch_size,
                                    datasheet,
                                    ratio_of_separate,
                                    img_iso_resolution,
                                    training_indices
                                    )

        print("\n train the model")

        train(training_input_images, training_input_labels, FLAGS)

    " train the model "

    if __name__ == '__run_train__':
        tf.app.run()

def set_model_parameters(datasheet, ratio_of_separate):
    """
    Setting basic parameters of the deep learning model
    """

    FLAGS = tf.app.flags.FLAGS
    #Max batch size= available GPU memory bytes / 4 / (size of tensors + trainable parameters)
    tf.app.flags.DEFINE_integer('batch_size', 128,
                                """Number of images to process in a batch.""")
    tf.app.flags.DEFINE_integer('max_steps', 200000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_integer('log_frequency', 10,
                                """How often to log results to the console.""")
    tf.app.flags.DEFINE_string('dir_dataset', dir_dataset,
                               """Path to the CIFAR-10 data directory.""")
    tf.app.flags.DEFINE_string('dir_train', dir_dataset + '/train',
                               """Used to write event logs and checkpoint.""")
    tf.app.flags.DEFINE_boolean('use_fp16', False,
                                """Train the model using fp16.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")

    #amount_img_pool = math.ceil(len(datasheet) * ratio_of_separate)
    #num_example_per_epoch_for_train = math.ceil(amount_img_pool * ratio_of_separate)
    num_example_per_epoch_for_train = math.ceil(len(datasheet) * ratio_of_separate)
    num_example_per_epoch_for_eval = len(datasheet) - num_example_per_epoch_for_train

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

def get_training_inputs(batch_size, datasheet, ratio_of_separate, img_iso_resolution, training_indices):

    print("\n *Import and parse the data set\n ")

    img = image_resizing(image_index, datasheet, img_iso_resolution)
    #training_features, testing_features, training_inputs, testing_inputs \
    #                    = parse_the_data(datasheet, ratio_of_separate)


    #print("\n *Initialize features columns\n ")

    #feature_initializer()

    #tensor_img = image_resizing(image_index, datasheet, img_iso_resolution)

    training_input_images = tf.convert_to_tensor(images)
    training_input_labels = tf.convert_to_tensor(labels)

    return training_input_images, training_input_labelss

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

    return img

#def feature_initializer():
#    """
#    Initialize features columns
#    """
#
#    var_init = tf.global_variables_initializer()
#    table_init = tf.tables_initializer()
#
#    sess = tf.Session()
#    sess.run((var_init, table_init))
#
#def get_training_feature_dict(training_indices, datasheet):
#
#    Gender_values = []
#    Boneage_values = []
#    for index in training_indices:
#        if datasheet[index][2] is 'False':
#            gender = 'F'
#            Gender_values.append(gender)
#        else:
#            gender = 'M'
#            Gender_values.append(gender)
#
#        boneage = [int(datasheet[index][1])]
#        Boneage_values.append(boneage)
#
#    training_features = {}
#    training_features['Boneage'] = Boneage_values
#    training_features['Gender'] = Gender_values
#
#    return training_features
#
#def get_testing_feature_dict(testing_indices, datasheet):
#
#    Gender_values = []
#    Boneage_values = []
#    for index in testing_indices:
#        if datasheet[index][2] is 'False':
#            gender = 'F'
#            Gender_values.append(gender)
#        else:
#            gender = 'M'
#            Gender_values.append(gender)
#
#        boneage = [int(datasheet[index][1])]
#        Boneage_values.append(boneage)
#
#    testing_features = {}
#    testing_features['Boneage'] = Boneage_values
#    testing_features['Gender'] = Gender_values
#
#    return testing_features
#
#def parse_the_data(datasheet, ratio_of_separate):
#
#
#    training_features = get_training_feature_dict(training_indices, datasheet)
#    testing_features = get_testing_feature_dict(testing_indices, datasheet)
#
#    gender_column = tf.feature_column.categorical_column_with_vocabulary_list('Gender', ['M', 'F'])
#    gender_column = tf.feature_column.indicator_column(gender_column)
#
#    columns = [
#        tf.feature_column.numeric_column('Boneage'),
#        gender_column
#    ]
#
#    training_inputs = tf.feature_column.input_layer(training_features, columns)
#    testing_inputs = tf.feature_column.input_layer(testing_features, columns)
#
#    return training_features, testing_features, training_inputs, testing_inputs

def model_layer_1(training_input_images):

    with tf.variable_scope('conv1') as scope:
        conv = tf.layers.conv2d(
                        input = training_input_images,
                        filters = 64,
                        kernel_size = (5, 5),
                        strides=(1, 1),
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=(1, 1),
                        activation=None,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name=None,
                        reuse=None
                        )

        biases = tf.get_variable('biases',
                                 shape = [64],
                                 initializer = tf.initializers.constant(dtype = dtype)
                                )

        conv1 = tf.nn.relu(tf.nn.bias_add(conv, biases),
                           name = scope.name
                          )

        _activation_summary(conv1)

    return conv1

def model_layer_2(layer_1_conv_output):

    pool1 = tf.nn.max_pool(layer_1_conv_output,
                           ksize = tf.shape(layer_1st_conv_output),
                           strides = [1, 2, 2, 1],
                           padding = 'SAME',
                           nmae = 'pool1'
                          )

    return pool1

def model_layer_3(layer_2_maxpool_output):

    norm1 = tf.nn.local_response_normalization(layer_2_maxpool_output, name='norm1')

    return norm1

def model_layer_4(layer_3_local_response_normalization_output):

    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
                        input = layer_3_local_response_normalization_output,
                        filters = 64,
                        kernel_size = (5, 5),
                        strides=(1, 1),
                        padding='same',
                        data_format='channels_last',
                        dilation_rate=(1, 1),
                        activation=None,
                        use_bias=True,
                        kernel_initializer=None,
                        bias_initializer=tf.zeros_initializer(),
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        activity_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None,
                        trainable=True,
                        name=None,
                        reuse=None
                        )

        biases = tf.get_variable('biases',
                                 shape = [64],
                                 initializer = tf.initializers.constant(dtype = dtype)
                                )

        conv2 = tf.nn.relu(tf.nn.bias_add(conv, biases),
                           name = scope.name
                          )

        _activation_summary(conv2)

    return conv2

def model_layer_5(layer_4_conv_output):

    norm2 = tf.nn.local_response_normalization(layer_4_conv_output, name='norm2')

    return norm2

def model_layer_6(layer_5_local_response_normalization_output):

    pool2 = tf.nn.max_pool(layer_5_local_response_normalization_output,
                           ksize = tf.shape(layer_5_local_response_normalization_output),
                           strides = [1, 2, 2, 1],
                           padding = 'SAME',
                           nmae = 'pool2'
                          )

    return pool2

def model_layer_7(layer_6_maxpool_output, images_shape, FLAGS):

    with tf.variable_scope('FC_rectified_activation_1') as scope:
        reshape_pool2 = tf.reshape(layer_6_maxpool_output,
                                 [images_shape.as_list()[0], -1])

        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

        weights = tf.get_variable('weights',
                                  shape = [tf.shape(reshape_pool2)[1], 64],
                                  initializer = tf.initializers.random_normal(dtype = dtype),
                                  dtype = dtype
                                 )

        biases = tf.get_variable('biases',
                                 shape = [64],
                                 initializer = tf.initializers.constant(dtype = dtype)
                                )

        FC_rectified_output = tf.nn.relu(tf.matmul(reshape_pool2, weights) + biases,
                                         name=scope.name
                                        )
        _activation_summary(FC_rectified_output)

    return FC_rectified_output

def model_layer_8(layer_7_FC_rectified_output, num_classes):

    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights',
                                  shape = [64, num_classes],
                                  initializer = tf.initializers.random_normal(dtype = dtype),
                                  dtype = dtype
                                 )

        biases = tf.get_variable('biases',
                                 shape = [num_classes],
                                 initializer = tf.initializers.constant(dtype = dtype)
                                )

        softmax_linear = tf.add(tf.matmul(layer_7_FC_rectified_output, weights),
                                biases,
                                name=scope.name
                                )
        _activation_summary(softmax_linear)

    return softmax_linear

def _activation_summary(input_tensor):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    """

    tf.summary.histogram(input_tensor.op.name + '/activations', input_tensor)
    tf.summary.scalar(input_tensor.op.name + '/sparsity', tf.nn.zero_fraction(input_tensor))

def loss(layer_8_softmax_output, training_input_labels):

    training_input_labels = tf.cast(training_input_labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels = training_input_labels,
                        logits = layer_8_softmax_output,
                        name = 'cross_entropy_per_example'
                        )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name = 'cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss

def _add_loss_summaries(total_loss):

    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    for L in losses + [total_loss]:

        tf.summary.scalar(L.op.name + ' (raw)', L)
        tf.summary.scalar(L.op.name, loss_averages.average(L))

    return loss_averages_op

def get_train_op(total_loss, batch_size, global_step):

    num_batches_per_epoch = num_example_per_epoch_for_train / batch_size
    decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
                        initital_learning_rate,
                        global_step,
                        decay_steps,
                        Learning_rate_decay_factor,
                        staircase = True
                        )

    tf.summary.scalar('learning_rate', learning_rate)
    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(total_loss)

    apply_gradient_op = optimizer.apply_gradients(gradients, global_step = global_step)

    for variable in tf.trainable_variables():
        tf.summary.histogram(variable.op.name, variable)

    for gradient, variable in gradients:
        if gradient is not None:
            tf.summary.histogram(variable.op.name + '/gradients', gradient)

    variable_moving_averages = tf.train.ExponentialMovingAverage(
                        moving_ave_decay,
                        global_step
                        )

    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_moving_averages.apply(tf.trainable_variables())

    return variables_averages_op

"Execution"

if __name__ == '__main__':

    dir_datasheet = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset.csv'
    dir_dataset = 'C:/Users/Alvin.Li/Desktop/small_project/dataset/boneage-training-dataset'

    start = time. time()

    main()

    end = time. time()

    duration = end - start

    print('\nThis code runs so fast that only spends {} in second.'.format(duration))
