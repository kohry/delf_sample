from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
from delf import extract_features

import traceback

# Extensions.
_DELF_EXTENSION = '.delf'

OUTPUT_PATH = 'output/index'
IMAGE_DIR = 'image_index'

# To avoid PIL crashing for truncated (corrupted) images.
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _PilLoader(path):
    """Helper function to read image with PIL.

    Args:
        path: Path to image to be loaded.

    Returns:
        PIL image in RGB format.
    """
    with tf.gfile.GFile(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# Read list of query images from dataset file.
tf.logging.info('Reading list of query images and boxes from dataset file...')
query_list = os.listdir(IMAGE_DIR)
num_images = len(query_list)
tf.logging.info('done! Found %d images', num_images)

# Parse DelfConfig proto.
config = delf_config_pb2.DelfConfig()
with tf.gfile.GFile('delf_gld_config.pbtxt', 'r') as f:
    text_format.Merge(f.read(), config)

# Create output directory if necessary.
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

with tf.Graph().as_default():
    with tf.Session() as sess:
        # Initialize variables, construct DELF extractor.
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        extractor_fn = extract_features.MakeExtractor(sess, config)

        start = time.clock()
        for i in range(num_images):
            
            try:
            
                query_image_name = query_list[i]
                input_image_filename = os.path.join(IMAGE_DIR,query_image_name)
                output_feature_filename = os.path.join(OUTPUT_PATH, query_image_name + _DELF_EXTENSION)
                if tf.gfile.Exists(output_feature_filename):
                    tf.logging.info('Skipping %s', query_image_name)
                    continue

                # Crop query image according to bounding box.
                # bbox = [int(round(b)) for b in ground_truth[i]['bbx']]
                # im = np.array(_PilLoader(input_image_filename).crop(bbox))

                im = np.array(_PilLoader(input_image_filename))

                # Extract and save features.
                (locations_out, descriptors_out, feature_scales_out,attention_out) = extractor_fn(im)

                feature_io.WriteToFile(output_feature_filename, locations_out,feature_scales_out, descriptors_out,attention_out)

                elapsed = (time.clock() - start)
                
                if i % 100 == 0 :
                    print('Processed %d query images in %f seconds' % (i, elapsed))
            
            except :
                traceback.print_exc()
