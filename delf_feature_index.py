from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import math
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
from delf import box_io
from delf import feature_io
from delf.python.detect_to_retrieve import dataset
from delf import extract_boxes
from delf import extract_features

import traceback
from tqdm import tqdm 

# Extension of feature files.
_BOX_EXTENSION = '.boxes'
_DELF_EXTENSION = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 10

# To avoid crashing for truncated (corrupted) images.
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_DIR = 'image_index'
OUTPUT_PATH_FEATURE = 'output/feature'
OUTPUT_PATH_BOX = 'output/box'
DETECTOR_MODEL_PATH = 'parameters/d2r_frcnn_20190411'
THRESHOLD = 0.1


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



def _WriteMappingBasenameToIds(index_names_ids_and_boxes, output_path):
    """Helper function to write CSV mapping from DELF file name to IDs.

    Args:
        index_names_ids_and_boxes: List containing 3-element lists with name, image
            ID and box ID.
        output_path: Output CSV path.
    """
    with tf.gfile.GFile(output_path, 'w') as f:
        csv_writer = csv.DictWriter(
                f, fieldnames=['name', 'index_image_id', 'box_id'])
        csv_writer.writeheader()
        for name_imid_boxid in index_names_ids_and_boxes:
            csv_writer.writerow({
                    'name': name_imid_boxid[0],
                    'index_image_id': name_imid_boxid[1],
                    'box_id': name_imid_boxid[2],
            })


tf.logging.info('Reading list of index images from dataset file...')
#index_list = ['image_index/' + s for s in os.listdir("image_index")]
index_list = os.listdir('image_index')
num_images = len(index_list)
tf.logging.info('done! Found %d images', num_images)

config = delf_config_pb2.DelfConfig()
with tf.gfile.GFile('delf_gld_config.pbtxt', 'r') as f:
    text_format.Merge(f.read(), config)

if not os.path.exists(OUTPUT_PATH_FEATURE):
    os.makedirs(OUTPUT_PATH_FEATURE)
if not os.path.exists(OUTPUT_PATH_BOX):
    os.makedirs(OUTPUT_PATH_BOX)

index_names_ids_and_boxes = []

with tf.Graph().as_default():
    with tf.Session() as sess:
        # Initialize variables, construct detector and DELF extractor.
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        detector_fn = extract_boxes.MakeDetector(sess, DETECTOR_MODEL_PATH, import_scope='detector')
        delf_extractor_fn = extract_features.MakeExtractor(sess, config, import_scope='extractor_delf')

        start = time.clock()
        for i in range(num_images):
            
            try :

                if i == 0:
                    print('Starting to extract features/boxes from index images...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = (time.clock() - start)
                    print('Processing index image %d out of %d, last %d '
                                'images took %f seconds' %
                                (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
                    start = time.clock()

                index_image_name = index_list[i]
                input_image_filename = os.path.join(IMAGE_DIR,index_image_name)
                output_feature_filename_whole_image = os.path.join(
                        OUTPUT_PATH_FEATURE, index_image_name + _DELF_EXTENSION)
                output_box_filename = os.path.join(OUTPUT_PATH_BOX,index_image_name + _BOX_EXTENSION)
                
                pil_im = _PilLoader(input_image_filename)
                width, height = pil_im.size

                # Extract and save boxes.
                if tf.gfile.Exists(output_box_filename):
                    tf.logging.info('Skipping box computation for %s', index_image_name)
                    (boxes_out, scores_out,class_indices_out) = box_io.ReadFromFile(output_box_filename)
                else:
                    (boxes_out, scores_out,class_indices_out) = detector_fn(np.expand_dims(pil_im, 0))
                    # Using only one image per batch.
                    boxes_out = boxes_out[0]
                    scores_out = scores_out[0]
                    class_indices_out = class_indices_out[0]
                    box_io.WriteToFile(output_box_filename, boxes_out, scores_out, class_indices_out)

                # Select boxes with scores greater than threshold. Those will be the
                # ones with extracted DELF features (besides the whole image, whose DELF
                # features are extracted in all cases).
                num_delf_files = 1
                selected_boxes = []
                for box_ind, box in enumerate(boxes_out):
                    if scores_out[box_ind] >= THRESHOLD:
                        selected_boxes.append(box)
                num_delf_files += len(selected_boxes)

                # Extract and save DELF features.
                for delf_file_ind in range(num_delf_files):
                    if delf_file_ind == 0:
                        index_box_name = index_image_name
                        output_feature_filename = output_feature_filename_whole_image
                    else:
                        index_box_name = index_image_name + '_' + str(delf_file_ind - 1)
                        output_feature_filename = os.path.join(OUTPUT_PATH_FEATURE, index_box_name + _DELF_EXTENSION)

                    index_names_ids_and_boxes.append(
                            [index_box_name, i, delf_file_ind - 1])

                    if tf.gfile.Exists(output_feature_filename):
                        tf.logging.info('Skipping DELF computation for %s', index_box_name)
                        continue

                    if delf_file_ind >= 1:
                        bbox_for_cropping = selected_boxes[delf_file_ind - 1]
                        bbox_for_cropping_pil_convention = [
                                int(math.floor(bbox_for_cropping[1] * width)),
                                int(math.floor(bbox_for_cropping[0] * height)),
                                int(math.ceil(bbox_for_cropping[3] * width)),
                                int(math.ceil(bbox_for_cropping[2] * height))
                        ]
                        pil_cropped_im = pil_im.crop(bbox_for_cropping_pil_convention)
                        im = np.array(pil_cropped_im)
                    else:
                        im = np.array(pil_im)

                    (locations_out, descriptors_out, feature_scales_out,attention_out) = delf_extractor_fn(im)

                    feature_io.WriteToFile(output_feature_filename, locations_out,feature_scales_out, descriptors_out,attention_out)
            except :
                traceback.print_exc()

# Save mapping from output DELF name to index image id and box id.
_WriteMappingBasenameToIds(index_names_ids_and_boxes, 'index_mapping_1.csv')
