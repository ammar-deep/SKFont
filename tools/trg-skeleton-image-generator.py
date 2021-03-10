import argparse
import io
import glob
import os

from skimage import img_as_uint, io as ioo
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize_3d, binary_closing

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/256-common-hangul.txt')
DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../trg-image-data/images')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../skel-image-data')

def get_binary(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary

def generate_skeleton_images(label_file, fonts_image_dir, output_dir):
    """Generate skeleton images.
    This function takes two arguments, i.e. font images whoose skeletons we want to generate
    and output directory where we will store these generated skeleton images. 
    Please make sure that the images are of 256*256 (PNG) size with black backgorund and white 
    character text.
    """
    # Set the path of skeleton images in output directory. It will be used later for
    # setting up skeleton images path for skeleton labels
    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    font_images = glob.glob(os.path.join(fonts_image_dir, '*.png'))
    # check if the images are jpeg
    if len(font_images) == 0:
        font_images = glob.glob(os.path.join(fonts_image_dir, '*.jpg'))

    # If input directory is empty or no images are found with .png, jpeg and .jpg extension
    if len(font_images) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # If the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in font_images):
        font_images = sorted(font_images, key=lambda path: int(get_name(path)))
    else:
        font_images = sorted(font_images)

    # Set the count so that we can view the progress on the terminal
    total_count = 0
    prev_count = 0

    for font_image in font_images:
        img_path = font_image
        # Split names and labels
        name, typeis = os.path.splitext(os.path.basename(img_path))
        img_name = name + typeis

        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} skeleton images generated...'.format(total_count))

        total_count += 1

        # Read the images one by one from the font images directory
        # Convert them from rgb to gray and then convert it into bool
        # Then apply skeletonize function and wrap it into binary_closing
        # for converting them into binary
        image = rgb2gray(imread(font_image))

        # Convert gray image to binary
        image = get_binary(image)

        # Skeletonize (otsu + skeletonize_3d)
        skeleton = skeletonize_3d(image)
        skeleton = binary_closing(skeleton)

        # convert image as uint before saving in output directory
        skeleton = img_as_uint(skeleton)

        file_string = img_name
        file_path = os.path.join(image_dir, file_string)     
        ioo.imsave(fname=file_path, arr=skeleton)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-image-dir', type=str, dest='fonts_image_dir',
                        default=DEFAULT_FONTS_IMAGE_DIR,
                        help='Directory of images to use for extracting skeletons.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated skeleton images.')
    args = parser.parse_args()

    generate_skeleton_images(args.label_file, args.fonts_image_dir, args.output_dir)