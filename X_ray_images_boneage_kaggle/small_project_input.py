from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv
import math

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

IMAGE_SIZE = 500

NUM_CLASSES = 2

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 10089
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2522
