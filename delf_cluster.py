
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import app
from delf import feature_io
from delf.python.detect_to_retrieve import dataset

IS_QUERY = False

# Extensions.
_DELF_EXTENSION = '.delf'

# Default DELF dimensionality.
_DELF_DIM = 30

# Pace to report log when collecting features.
_STATUS_CHECK_ITERATIONS = 100

OUTPUT_PATH = 'output/cluster/index'
FEATURE_PATH = 'output/index'

if IS_QUERY : 
    OUTPUT_PATH = 'output/cluster/test'
    FEATURE_PATH = 'output/test'
else :
    OUTPUT_PATH = 'output/cluster/index'
    FEATURE_PATH = 'output/index'

if os.path.exists(OUTPUT_PATH):
    raise RuntimeError(
            'output_cluster_dir = %s already exists. This may indicate that a '
            'previous run already wrote checkpoints in this directory, which would '
            'lead to incorrect training. Please re-run this script by specifying an'
            ' inexisting directory.' % OUTPUT_PATH)
else : 
    os.makedirs(OUTPUT_PATH)

# Read list of index images from dataset file.
print('Reading list of index images from dataset file...')

if IS_QUERY : 
    image_list = os.listdir("image_test")
else :
    image_list = os.listdir("image_index")

num_images = len(image_list)
print('done! Found %d images' % num_images)


# Loop over list of index images and collect DELF features.
features_for_clustering = []
start = time.clock()
print('Starting to collect features from index images...')
for i in range(num_images):
    if i > 0 and i % _STATUS_CHECK_ITERATIONS == 0:
        elapsed = (time.clock() - start)
        print('Processing index image %d out of %d, last %d '
                    'images took %f seconds' %
                    (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
        start = time.clock()

    features_filename = image_list[i] + _DELF_EXTENSION
    features_fullpath = os.path.join(FEATURE_PATH, features_filename)
    _, _, features, _, _ = feature_io.ReadFromFile(features_fullpath)
    if features.size != 0:
        assert features.shape[1] == _DELF_DIM
    for feature in features:
        features_for_clustering.append(feature)

features_for_clustering = np.array(features_for_clustering, dtype=np.float32)
print('All features were loaded! There are %d features, each with %d '
            'dimensions' %
            (features_for_clustering.shape[0], features_for_clustering.shape[1]))

class _IteratorInitHook(tf.train.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(_IteratorInitHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)

# Run K-means clustering.
def _get_input_fn():
    """Helper function to create input function and hook for training.

    Returns:
        input_fn: Input function for k-means Estimator training.
        init_hook: Hook used to load data during training.
    """
    init_hook = _IteratorInitHook()

    def _input_fn():
        """Produces tf.data.Dataset object for k-means training.

        Returns:
            Tensor with the data for training.
        """
        features_placeholder = tf.placeholder(tf.float32, features_for_clustering.shape)
        delf_dataset = tf.data.Dataset.from_tensor_slices((features_placeholder))
        delf_dataset = delf_dataset.shuffle(1000).batch(features_for_clustering.shape[0])
        iterator = delf_dataset.make_initializable_iterator()

        def _initializer_fn(sess):
            """Initialize dataset iterator, feed in the data."""
            sess.run(
                    iterator.initializer,
                    feed_dict={features_placeholder: features_for_clustering})

        init_hook.iterator_initializer_fn = _initializer_fn
        return iterator.get_next()

    return _input_fn, init_hook

input_fn, init_hook = _get_input_fn()

# kmeans = tf.estimator.experimental.KMeans(
#         num_clusters=1000, # cluster숫자
#         model_dir=OUTPUT_PATH,
#         use_mini_batch=False,
# )

kmeans = tf.contrib.factorization.KMeansClustering(
    num_clusters= 1000,
    use_mini_batch=False,
    model_dir=OUTPUT_PATH
)

iteration_count = 50

print('Starting K-means clustering...')
start = time.clock()
for i in range(iteration_count): #임의의 숫자로 클러스러이
    kmeans.train(input_fn, hooks=[init_hook])
    average_sum_squared_error = kmeans.evaluate(
            input_fn, hooks=[init_hook])['score'] / features_for_clustering.shape[0]
    elapsed = (time.clock() - start)
    print('K-means iteration %d (out of %d) took %f seconds, '
                'average-sum-of-squares: %f' %
                (i, iteration_count, elapsed, average_sum_squared_error))
    start = time.clock()

print('K-means clustering finished!')