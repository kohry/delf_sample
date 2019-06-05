from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import sys
import time

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import aggregation_config_pb2
from delf import datum_io
from delf import feature_aggregation_extractor
from delf import feature_io
from delf.python.detect_to_retrieve import dataset

import traceback

# Aliases for aggregation types.
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR

# Extensions.
_DELF_EXTENSION = '.delf'
_VLAD_EXTENSION_SUFFIX = 'vlad'
_ASMK_EXTENSION_SUFFIX = 'asmk'
_ASMK_STAR_EXTENSION_SUFFIX = 'asmk_star'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100

#QUERY일 경우, 그리고 INDEX일 경우 ################ 이부분 동적변경필요
IS_QUERY = True

OUTPUT_QUERY = 'output/agg/query'
OUTPUT_INDEX = 'output/agg/index'

INPUT_QUERY = 'image_test'
INPUT_INDEX = 'image_index'

if IS_QUERY == True:
    # QUERY의 Feature를 담고있음
    FEATURE_FILE_PATH = 'output/test'
else:
    FEATURE_FILE_PATH = 'output/index'

def _ReadMappingBasenameToBoxNames(input_path, index_image_names):
    """Reads mapping from image name to DELF file names for each box.

    Args:
        input_path: Path to CSV file containing mapping.
        index_image_names: List containing index image names, in order, for the
            dataset under consideration.

    Returns:
        images_to_box_feature_files: Dict. key=string (image name); value=list of
            strings (file names containing DELF features for boxes).
    """
    images_to_box_feature_files = {}
    with tf.gfile.GFile(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index_image_name = index_image_names[int(row['index_image_id'])]
            if index_image_name not in images_to_box_feature_files:
                images_to_box_feature_files[index_image_name] = []

            images_to_box_feature_files[index_image_name].append(row['name'])

    return images_to_box_feature_files

# Read list of images from dataset file.
print('Reading list of images from dataset file...')

if IS_QUERY:
    image_list = os.listdir(INPUT_QUERY)
else:
    image_list = os.listdir(INPUT_INDEX)
    
num_images = len(image_list)
print('done! Found %d images' % num_images)

# Parse AggregationConfig proto, and select output extension.
config = aggregation_config_pb2.AggregationConfig()

if IS_QUERY:
    with tf.gfile.GFile('query_aggregation_config.pbtxt', 'r') as f:
        text_format.Merge(f.read(), config)
else:
    with tf.gfile.GFile('index_aggregation_config.pbtxt', 'r') as f:
        text_format.Merge(f.read(), config)

output_extension = '.'
if config.use_regional_aggregation:
    output_extension += 'r'
if config.aggregation_type == _VLAD:
    output_extension += _VLAD_EXTENSION_SUFFIX
elif config.aggregation_type == _ASMK:
    output_extension += _ASMK_EXTENSION_SUFFIX
elif config.aggregation_type == _ASMK_STAR:
    output_extension += _ASMK_STAR_EXTENSION_SUFFIX
else:
    raise ValueError('Invalid aggregation type: %d' % config.aggregation_type)


# Read index mapping path, if provided.
# if IS_QUERY == False:
#     images_to_box_feature_files = _ReadMappingBasenameToBoxNames('index_mapping_1.csv', image_list)

output_dir = ""

if IS_QUERY:
    output_dir = OUTPUT_QUERY
    if not os.path.exists(OUTPUT_QUERY):
        os.makedirs(OUTPUT_QUERY)
else:
    output_dir = OUTPUT_INDEX
    if not os.path.exists(OUTPUT_INDEX):
        os.makedirs(OUTPUT_INDEX)

with tf.Session() as sess:
    extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(sess, config)

    start = time.clock()
    for i in range(num_images):
        if i == 0:
            print('Starting to extract aggregation from images...')
        elif i % _STATUS_CHECK_ITERATIONS == 0:
            elapsed = (time.clock() - start)
            print('Processing image %d out of %d, last %d '
                        'images took %f seconds' %
                        (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
            start = time.clock()

        image_name = image_list[i]

        output_aggregation_filename = os.path.join(output_dir, image_name + output_extension)
        if tf.io.gfile.exists(output_aggregation_filename):
            print('Skipping %s' % image_name)
            continue

        # Load DELF features.
        if config.use_regional_aggregation:
            if not 'index_mapping_1.csv':
                raise ValueError(
                        'Requested regional aggregation, but index_mapping_path was not '
                        'provided')
            descriptors_list = []
            num_features_per_box = []
            for box_feature_file in images_to_box_feature_files[image_name]:
                
                try :
                
                    delf_filename = os.path.join(FEATURE_FILE_PATH, box_feature_file + _DELF_EXTENSION)
                    _, _, box_descriptors, _, _ = feature_io.ReadFromFile(delf_filename)
                    # If `box_descriptors` is empty, reshape it such that it can be
                    # concatenated with other descriptors.
                    if not box_descriptors.shape[0]:
                        box_descriptors = np.reshape(box_descriptors,[0, config.feature_dimensionality])
                    descriptors_list.append(box_descriptors)
                    num_features_per_box.append(box_descriptors.shape[0])
                except :
                    traceback.print_exc()

            descriptors = np.concatenate(descriptors_list)
        else:
            input_delf_filename = os.path.join(FEATURE_FILE_PATH,image_name + _DELF_EXTENSION)
            _, _, descriptors, _, _ = feature_io.ReadFromFile(input_delf_filename)
            num_features_per_box = None

        # Extract and save aggregation. If using VLAD, only
        # `aggregated_descriptors` needs to be saved.
        
        try:
            (aggregated_descriptors,feature_visual_words) = extractor.Extract(descriptors,num_features_per_box)
        except:
            traceback.print_exc()
            continue

        if config.aggregation_type == _VLAD:
            datum_io.WriteToFile(aggregated_descriptors,output_aggregation_filename)
        else:
            datum_io.WritePairToFile(aggregated_descriptors,feature_visual_words.astype('uint32'),output_aggregation_filename)

