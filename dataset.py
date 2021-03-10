import tensorflow as tf
import os
import glob
import random
import math
import collections

from utils import *

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
train_tfrecords_dir = os.path.join(SCRIPT_PATH, 'train-tfrecords-output')
test_tfrecords_dir = os.path.join(SCRIPT_PATH, 'test-tfrecords-output')
trg_font_path = os.path.join(SCRIPT_PATH, 'trg_font')

CROP_SIZE = 256
# parameters for style embedding
train_num_styles = len(glob.glob1(trg_font_path,"*.ttf"))
fine_tune_styles = 0
total_styles = train_num_styles + fine_tune_styles

Examples = collections.namedtuple("Examples", "paths, src_font, trg_font, trg_skeleton, count, steps_per_epoch, style_labels")

############################################################################################
# _parse_function() is called from the load_examples() below.  
# It extracts images, style, and character labels from the TFRecord files.  
# It the applies further preprocessing like normalization, random jittering,
# breaking inoput and target image, cropping scalling etc. 
# Finally returns input_image, target_image, style_label, character_label, path as a batch
############################################################################################ 

def _parse_function(example, a):
    features = tf.parse_single_example(
        example,
        features={
            'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/path': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'image/style_label': tf.FixedLenFeature([], tf.int64),
            'image/character_label': tf.FixedLenFeature([], tf.int64)
        })
    # Get the data.
    image_encoded = features['image/encoded']
    path = features['image/path']
    style_label = features['image/style_label']
    # character_label = features['image/character_label']

    # Decode the JPEG.
    image = tf.image.decode_png(image_encoded, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    with tf.name_scope("load_images"):
        # Check if images have 3 channels or not i.e. rgb or not
        assertion = tf.assert_equal(tf.shape(image)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            image = tf.identity(image)

        image.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(image)[1] # [height, width, channels]
        a_images = preprocess(image[:,:width//3,:])
        b_images = preprocess(image[:,width//3:width//3 * 2,:])
        c_images = preprocess(image[:,width//3 * 2:, :])

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    # Transform function simply applies some preprocessing on input and target image to upscale the size etc
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        # Just flip image of hangul or skeleton from left to right
        # if a.flip:
        #     r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("source_font"):
        src_font = transform(a_images)

    with tf.name_scope("target_font"):
        trg_font = transform(b_images)

    with tf.name_scope("target_skeleton"):
        trg_skeleton = transform(c_images)

    # Represent the label as a one-hot vector.
    style_label = tf.stack(tf.one_hot(style_label, total_styles, dtype=tf.float32))
    # character_label = tf.stack(tf.one_hot(character_label, total_characters, dtype=tf.float32))
    # print("labels shape in parser functions ", label.shape)

    return src_font, trg_font, trg_skeleton, style_label, path

##################################################################################
# Load TFRecord files for training or testing and apply preprocessing on images
# Preprocessing is done using the _parse_function() defined above
# Finally the "named tuple" Examples is returned to the main function
################################################################################## 
def load_examples(args):
    total_records = 0
    if args.mode == "test":
            print('Processing the Test TFRecord File')
            tf_record_pattern = os.path.join(test_tfrecords_dir, '%s-*' % 'test')
            test_data_files = tf.gfile.Glob(tf_record_pattern)

            # Create testing dataset input pipeline.
            test_dataset = tf.data.TFRecordDataset(test_data_files) \
                .map(lambda example: _parse_function(example, args)) \
                .batch(args.batch_size) \
                .prefetch(1)

            iterator = test_dataset.make_one_shot_iterator()
            batch = iterator.get_next()

            # Function for getting the total no of records
            for fn in test_data_files:
                for record in tf.python_io.tf_record_iterator(fn):
                   total_records += 1
    else:
        print('Processing the Train TFRecord File')
        tf_record_pattern = os.path.join(train_tfrecords_dir, '%s-*' % 'train')
        train_data_files = tf.gfile.Glob(tf_record_pattern)

        # Create training dataset input pipeline.
        train_dataset = tf.data.TFRecordDataset(train_data_files) \
            .map(lambda example: _parse_function(example, args)) \
            .shuffle(1000) \
            .repeat(count=None) \
            .batch(args.batch_size) \
            .prefetch(1)

        iterator = train_dataset.make_one_shot_iterator()
        batch = iterator.get_next()

        # Function for getting the total no of records
        for fn in train_data_files:
            for record in tf.python_io.tf_record_iterator(fn):
               total_records += 1

   # batch contains the input images , labels and target images for the model
    src_font, trg_font, trg_skeleton, style_label, path = batch
    steps_per_epoch = int(math.ceil(total_records / args.batch_size))

   # Finally Examples named tuple is returned to the main function for feeding into the model
    return Examples(
        paths=path,
        src_font=src_font,
        trg_font=trg_font,
        trg_skeleton=trg_skeleton,
        count=total_records,
        steps_per_epoch=steps_per_epoch,
        style_labels = style_label,
    )