# imprts related to creating paths
import os
import argparse

# imports related to preprocess from pix2pix
import tfimage as im
import time
import tensorflow as tf
import numpy as np
import threading
import shutil

# setting global variable for counter 
total_count = 0

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/256-common-hangul.txt')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../combine-image-data')

def remove_dir(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def combine(src, src_path):
    if args.b_dir is None:
        raise Exception("missing b_dir")
    if args.c_dir is None:
        raise Exception("missing c_dir")

    # Find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(args.b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # Find corresponding file in c_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path2 = os.path.join(args.c_dir, basename + ext)
        if os.path.exists(sibling_path2):
            sibling2 = im.load(sibling_path2)
            break
    else:
        raise Exception("could not find sibling2 image for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1] or height != sibling2.shape[0] or width != sibling2.shape[1]:
        raise Exception("differing sizes")
    
    # convert all images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    if sibling2.shape[2] == 1:
        sibling2 = im.grayscale_to_rgb(images=sibling2)

    # remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]
    
    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    if sibling2.shape[2] == 4:
        sibling2 = sibling2[:,:,:3]

    return np.concatenate([src, sibling, sibling2], axis=1)

def process(src_path, dst_path, image_dir):
    global total_count

    total_count += 1
    src = im.load(src_path)

    if args.operation == "combine":
        dst = combine(src, src_path)
    else:
        raise Exception("invalid operation")
    im.save(dst, dst_path)

complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now

def generate_font_skeleton_combine_images(label_file, output_dir):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set the path of hangul-skeleton-combine images in output directory. It will be used later for 
    # setting up hangul-skeleton-combine images path for hangul-skeleton-combine labels
    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    src_paths = []
    dst_paths = []

    # Check if the directory and images already exsist?
    # If yes then skip those images else create the paths list
    skipped = 0
    for src_path in sorted(im.find(args.input_dir), key=os.path.getmtime):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(image_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)
    
    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()

    if args.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path in zip(src_paths, dst_paths):
                process(src_path, dst_path, image_dir)
                complete()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, 
                            help="path to folder containing source font images")
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                            default=DEFAULT_OUTPUT_DIR,
                            help='Output directory to store combine images.')
    parser.add_argument("--operation", required=True, choices=["combine"])
    parser.add_argument("--workers", type=int, default=1, help="number of workers")
    # combine
    parser.add_argument("--b_dir", type=str, help="path to folder containing target font images for combine operation")
    parser.add_argument("--c_dir", type=str, help="path to folder containing target font's skeleton images for combine operation")
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    args = parser.parse_args()

    generate_font_skeleton_combine_images(args.label_file, args.output_dir)

    # remove the src and target directories
    src_head, _ = os.path.split(args.input_dir)
    trg_head, _ = os.path.split(args.b_dir)
    trg_skl_head, _ = os.path.split(args.c_dir)
    print("Removing the directories")
    remove_dir(src_head)
    remove_dir(trg_head)
    remove_dir(trg_skl_head)
    print("*** DONE ***")