from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

from utils import *
from dataset import *
from ops import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=1000, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument("--classification_penalty", type=float, default=1.0, help="weight for classification loss")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
args = parser.parse_args()
CROP_SIZE = 256

def main():
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == "test" or args.mode == "export":
        if args.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(args.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(args, key, val)
        # disable these features in test mode
        args.scale_size = CROP_SIZE
        args.flip = False

    for k, v in args._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    # Load dataset
    examples = load_examples(args)
    print("examples count = %d" % examples.count)

    # src_font, trg_font, trg_skeleton are [batch_size, height, width, channels]
    model = create_model(examples.src_font, examples.trg_font, examples.trg_skeleton, examples.style_labels, args)

    src_font = deprocess(examples.src_font)
    trg_font = deprocess(examples.trg_font)
    trg_skeleton = deprocess(examples.trg_skeleton)
    f2f_outputs = deprocess(model.f2f_outputs)
    f2s_outputs = deprocess(model.f2s_outputs)
    s2f_outputs = deprocess(model.s2f_outputs)

    def convert(image):
        if args.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * args.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_src_font"):
        converted_src_font = convert(src_font)

    with tf.name_scope("convert_trg_font"):
        converted_trg_font = convert(trg_font)

    with tf.name_scope("convert_trg_skeleton"):
        converted_trg_skeleton = convert(trg_skeleton)

    with tf.name_scope("convert_f2f_outputs"):
        converted_f2f_outputs = convert(f2f_outputs)

    with tf.name_scope("convert_f2s_outputs"):
        converted_f2s_outputs = convert(f2s_outputs)

    with tf.name_scope("convert_s2f_outputs"):
        converted_s2f_outputs = convert(s2f_outputs)


    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "src_font": tf.map_fn(tf.image.encode_png, converted_src_font, dtype=tf.string, name="f2f_input_pngs"),
            "trg_font": tf.map_fn(tf.image.encode_png, converted_trg_font, dtype=tf.string, name="f2s_input_pngs"),
            "trg_skeleton": tf.map_fn(tf.image.encode_png, converted_trg_skeleton, dtype=tf.string, name="s2f_input_pngs"),
            "f2f_outputs": tf.map_fn(tf.image.encode_png, converted_f2f_outputs, dtype=tf.string, name="f2f_output_pngs"),
            "f2s_outputs": tf.map_fn(tf.image.encode_png, converted_f2s_outputs, dtype=tf.string, name="f2s_output_pngs"),
            "s2f_outputs": tf.map_fn(tf.image.encode_png, converted_s2f_outputs, dtype=tf.string, name="s2f_output_pngs"),
        }

    # summaries
    with tf.name_scope("src_font_summary"):
        tf.summary.image("src_font", converted_src_font)

    with tf.name_scope("trg_font_summary"):
        tf.summary.image("trg_font", converted_trg_font)

    with tf.name_scope("trg_skeleton_summary"):
        tf.summary.image("trg_skeleton", converted_trg_skeleton)

    with tf.name_scope("f2f_predict_real_summary"):
        tf.summary.image("f2f_predict_real", tf.image.convert_image_dtype(model.f2f_predict_real, dtype=tf.uint8))

    with tf.name_scope("f2f_predict_fake_summary"):
        tf.summary.image("f2f_predict_fake", tf.image.convert_image_dtype(model.f2f_predict_fake, dtype=tf.uint8))

    # F2F tensorboard summaries
    tf.summary.scalar("f2f_discriminator_loss_fake", model.f2f_disc_fake_loss)
    tf.summary.scalar("f2f_discriminator_loss_real", model.f2f_disc_real_loss) 
    tf.summary.scalar("f2f_discriminator_loss_real_styl", model.f2f_disc_loss_real_styl)
    tf.summary.scalar("f2f_generator_loss_GAN", model.f2f_gen_loss_GAN)
    tf.summary.scalar("f2f_generator_loss_L1", model.f2f_gen_loss_L1)

    # F2S tensorboard summaries
    tf.summary.scalar("f2s_discriminator_loss_fake", model.f2s_disc_fake_loss)
    tf.summary.scalar("f2s_discriminator_loss_real", model.f2s_disc_real_loss) 
    tf.summary.scalar("f2s_discriminator_loss_real_styl", model.f2s_disc_loss_real_styl)
    tf.summary.scalar("f2s_generator_loss_GAN", model.f2s_gen_loss_GAN)
    tf.summary.scalar("f2s_generator_loss_L1", model.f2s_gen_loss_L1)

    # S2F tensorboard summaries
    tf.summary.scalar("s2f_discriminator_loss_fake", model.s2f_disc_fake_loss)
    tf.summary.scalar("s2f_discriminator_loss_real", model.s2f_disc_real_loss) 
    tf.summary.scalar("s2f_discriminator_loss_real_styl", model.s2f_disc_loss_real_styl)
    tf.summary.scalar("s2f_generator_loss_GAN", model.s2f_gen_loss_GAN)
    tf.summary.scalar("s2f_generator_loss_L1", model.s2f_gen_loss_L1)


    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.f2f_discrim_grads_and_vars + model.f2f_gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    for grad, var in model.f2s_discrim_grads_and_vars + model.f2s_gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    for grad, var in model.s2f_discrim_grads_and_vars + model.s2f_gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = args.output_dir if (args.trace_freq > 0 or args.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        if args.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(args.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if args.max_epochs is not None:
            max_steps = examples.steps_per_epoch * args.max_epochs
        if args.max_steps is not None:
            max_steps = args.max_steps

        if args.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results, args, step=None)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets, args, step=False)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(args.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }

                if should(args.progress_freq):
                    #F2F progress parameters
                    fetches["f2f_disc_real_loss"] = model.f2f_disc_real_loss
                    fetches["f2f_disc_fake_loss"] = model.f2f_disc_fake_loss 
                    fetches["f2f_disc_loss_real_styl"] = model.f2f_disc_loss_real_styl
                    fetches["f2f_gen_loss_GAN"] = model.f2f_gen_loss_GAN
                    fetches["f2f_gen_loss_L1"] = model.f2f_gen_loss_L1

                    #F2S progress parameters
                    fetches["f2s_disc_real_loss"] = model.f2s_disc_real_loss
                    fetches["f2s_disc_fake_loss"] = model.f2s_disc_fake_loss 
                    fetches["f2s_disc_loss_real_styl"] = model.f2s_disc_loss_real_styl
                    fetches["f2s_gen_loss_GAN"] = model.f2s_gen_loss_GAN
                    fetches["f2s_gen_loss_L1"] = model.f2s_gen_loss_L1

                    #S2F progress parameters
                    fetches["s2f_disc_real_loss"] = model.s2f_disc_real_loss
                    fetches["s2f_disc_fake_loss"] = model.s2f_disc_fake_loss 
                    fetches["s2f_disc_loss_real_styl"] = model.s2f_disc_loss_real_styl
                    fetches["s2f_gen_loss_GAN"] = model.s2f_gen_loss_GAN
                    fetches["s2f_gen_loss_L1"] = model.s2f_gen_loss_L1

                if should(args.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(args.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(args.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(args.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], args, step=results["global_step"])
                    append_index(filesets, args, step=True)

                if should(args.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(args.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * args.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * args.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print()
                    print("f2f_disc_real_loss", results["f2f_disc_real_loss"])
                    print("f2f_disc_fake_loss", results["f2f_disc_fake_loss"]) 
                    print("f2f_disc_loss_real_styl", results["f2f_disc_loss_real_styl"])
                    print("f2f_gen_loss_GAN", results["f2f_gen_loss_GAN"])
                    print("f2f_gen_loss_L1", results["f2f_gen_loss_L1"])
                    print()

                    print()
                    print("f2s_disc_real_loss", results["f2s_disc_real_loss"])
                    print("f2s_disc_fake_loss", results["f2s_disc_fake_loss"]) 
                    print("f2s_disc_loss_real_styl", results["f2s_disc_loss_real_styl"])
                    print("f2s_gen_loss_GAN", results["f2s_gen_loss_GAN"])
                    print("f2s_gen_loss_L1", results["f2s_gen_loss_L1"])
                    print()

                    print()
                    print("s2f_disc_real_loss", results["s2f_disc_real_loss"])
                    print("s2f_disc_fake_loss", results["s2f_disc_fake_loss"]) 
                    print("s2f_disc_loss_real_styl", results["s2f_disc_loss_real_styl"])
                    print("s2f_gen_loss_GAN", results["s2f_gen_loss_GAN"])
                    print("s2f_gen_loss_L1", results["s2f_gen_loss_L1"])
                    print()
                if should(args.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

main()