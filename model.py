import tensorflow as tf
import collections
import os
import glob

from ops import *

EPS = 1e-12

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
trg_font_path = os.path.join(SCRIPT_PATH, 'trg_font')

Model = collections.namedtuple("Model", "f2f_outputs, f2f_predict_real, f2f_predict_fake, f2f_disc_real_loss, f2f_disc_fake_loss, f2f_disc_loss_real_styl, f2f_discrim_grads_and_vars, f2f_gen_loss_GAN, f2f_gen_loss_L1, f2f_gen_grads_and_vars, f2s_outputs, f2s_predict_real, f2s_predict_fake, f2s_disc_real_loss, f2s_disc_fake_loss, f2s_disc_loss_real_styl, f2s_discrim_grads_and_vars, f2s_gen_loss_GAN, f2s_gen_loss_L1, f2s_gen_grads_and_vars, s2f_outputs, s2f_predict_real, s2f_predict_fake, s2f_disc_real_loss, s2f_disc_fake_loss, s2f_disc_loss_real_styl, s2f_discrim_grads_and_vars, s2f_gen_loss_GAN, s2f_gen_loss_L1, s2f_gen_grads_and_vars, train")

# parameters for style embedding
train_num_styles = len(glob.glob1(trg_font_path,"*.ttf"))
fine_tune_styles = 0
total_styles = train_num_styles + fine_tune_styles

##################################################################################
# Generator
# It is a U-Net architecture i.e. and encoder decoder with skip connections
##################################################################################

def create_generator(generator_inputs, generator_outputs_channels, styl_label, args):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, args.ngf, args)
        layers.append(output)

    layer_specs = [
        args.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        args.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        args.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        args.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        args.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        args.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        args.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels, args)
            output = batchnorm(convolved)
            layers.append(output)

    # Adding style labels here.
    styl_labels = tf.reshape(styl_label, [-1, 1, 1, total_styles])
    # We are concatinating styles with enocder output.   
    layers[-1] = tf.concat([layers[-1], styl_labels], axis=3)

    layer_specs = [
        (args.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (args.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (args.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (args.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (args.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (args.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (args.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels, args)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels, args)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

##################################################################################
# Discriminator
# Its a PatchGAN with outputs a patch of N*N dimension. N*N here is 30*30
# Each pixel in the N*N patch is actually telling whether the corresponding patch 
# in the input image is Real or Fake
##################################################################################

def create_discriminator(discrim_inputs, discrim_targets, args):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, args.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = args.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        # Adding Fully connected layer after our Encoder as we want to add classification loss
        output_flat = tf.layers.flatten(layers[-1])

        with tf.variable_scope('layer_fc_s'):
            styl_y = tf.layers.dense(output_flat, total_styles,
                                          kernel_initializer=tf.random_normal_initializer(0, 0.02))

        return layers[-1], styl_y

##################################################################################
# Build Model
# Run the Generator, then the discriminator two times for real and fake image respectively. 
# Two loss functions are used 1) GAN loss 2) L1 loss
# Then Generator and Discriminator are trained using the Adam Optimizer
# Then apply ExponentailMovingAverage while training the weights
# global_step then just keeps track of the number of batches seen so far.
##################################################################################

def create_model(src_font, trg_font, trg_skeleton, style_labels, args):
    ################################################## 
    # F2F cGAN -> A uNet generator and a discriminator
    ##################################################

    with tf.name_scope("f2f_generator"):
        with tf.variable_scope("generator"):
            out_channels = int(trg_font.get_shape()[-1])
            f2f_outputs = create_generator(src_font, out_channels, style_labels, args)

    # create two copies of f2f_discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("f2f_real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            f2f_predict_real, f2f_real_styl = create_discriminator(src_font, trg_font, args)

    with tf.name_scope("f2f_fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            f2f_predict_fake, f2f_fake_styl = create_discriminator(src_font, f2f_outputs, args)


    ################################################## 
    # F2S cGAN -> A uNet generator and a discriminator
    ##################################################
    
    with tf.name_scope("f2s_generator"):
        with tf.variable_scope("generator", reuse=True):
            out_channels = int(trg_font.get_shape()[-1])
            f2s_outputs = create_generator(trg_font, out_channels, style_labels, args)

    # create two copies of f2s_discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("f2s_real_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            f2s_predict_real, f2s_real_styl = create_discriminator(trg_font, trg_skeleton, args)

    with tf.name_scope("f2s_fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            f2s_predict_fake, f2s_fake_styl = create_discriminator(trg_font, f2s_outputs, args)

    ################################################## 
    # S2F cGAN -> A uNet generator and a discriminator
    ##################################################
    
    with tf.name_scope("s2f_generator"):
        with tf.variable_scope("generator", reuse=True):
            out_channels = int(trg_font.get_shape()[-1])
            s2f_outputs = create_generator(trg_skeleton, out_channels, style_labels, args)

    # create two copies of s2f_discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("s2f_real_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            s2f_predict_real, s2f_real_styl = create_discriminator(trg_skeleton, trg_font, args)

    with tf.name_scope("s2f_fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 7, 7, 1]
            s2f_predict_fake, s2f_fake_styl = create_discriminator(trg_skeleton, s2f_outputs, args)

    ##################################################
    # F2F Loss Functions
    ##################################################
    with tf.name_scope("f2f_discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # f2f_predict_real => 1
        # f2f_predict_fake => 0
        f2f_disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f2f_predict_real), logits=f2f_predict_real))
        f2f_disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(f2f_predict_fake), logits=f2f_predict_fake))
        
        f2f_disc_loss_real_styl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=style_labels, logits=f2f_real_styl))

        # f2f_Discriminator Final Loss
        f2f_discrim_loss =  f2f_disc_real_loss + f2f_disc_fake_loss + f2f_disc_loss_real_styl * args.classification_penalty

    with tf.name_scope("f2f_generator_loss"):
        # f2f_predict_fake => 1
        # abs(targets - outputs) => 0
        f2f_gen_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f2f_predict_fake), logits=f2f_predict_fake))
        f2f_gen_loss_L1 = tf.reduce_mean(tf.abs(trg_font - f2f_outputs))
        f2f_gen_loss = f2f_gen_loss_GAN * args.gan_weight + f2f_gen_loss_L1 * args.l1_weight


    ##################################################
    # F2S Loss Functions
    ##################################################
    with tf.name_scope("f2S_discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # f2s_predict_real => 1
        # f2s_predict_fake => 0
        f2s_disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f2s_predict_real), logits=f2s_predict_real))
        f2s_disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(f2s_predict_fake), logits=f2s_predict_fake))
        
        f2s_disc_loss_real_styl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=style_labels, logits=f2s_real_styl))

        # f2s_Discriminator Final Loss
        f2s_discrim_loss =  f2s_disc_real_loss + f2s_disc_fake_loss + f2s_disc_loss_real_styl * args.classification_penalty

    with tf.name_scope("f2s_generator_loss"):
        # f2s_f2s_predict_fake => 1
        # abs(targets - outputs) => 0
        f2s_gen_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(f2s_predict_fake), logits=f2s_predict_fake))
        f2s_gen_loss_L1 = tf.reduce_mean(tf.abs(trg_skeleton - f2s_outputs))
        f2s_gen_loss = f2s_gen_loss_GAN * args.gan_weight + f2s_gen_loss_L1 * args.l1_weight


    ##################################################
    # S2F Loss Functions
    ##################################################
    with tf.name_scope("s2f_discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # s2f_predict_real => 1
        # s2f_predict_fake => 0
        s2f_disc_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(s2f_predict_real), logits=s2f_predict_real))
        s2f_disc_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(s2f_predict_fake), logits=s2f_predict_fake))
        
        s2f_disc_loss_real_styl = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=style_labels, logits=s2f_real_styl))

        # s2f_Discriminator Final Loss
        s2f_discrim_loss =  s2f_disc_real_loss + s2f_disc_fake_loss + s2f_disc_loss_real_styl * args.classification_penalty

    with tf.name_scope("s2f_generator_loss"):
        # s2f_predict_fake => 1
        # abs(targets - outputs) => 0
        s2f_gen_loss_GAN = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(s2f_predict_fake), logits=s2f_predict_fake))
        s2f_gen_loss_L1 = tf.reduce_mean(tf.abs(trg_font - s2f_outputs))
        s2f_gen_loss = s2f_gen_loss_GAN * args.gan_weight + s2f_gen_loss_L1 * args.l1_weight

    ##################################################
    # Training f2f_D and f2f_G using Adam
    ##################################################
    with tf.name_scope("f2f_discriminator_train"):
        f2f_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        f2f_discrim_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
        f2f_discrim_grads_and_vars = f2f_discrim_optim.compute_gradients(f2f_discrim_loss, var_list=f2f_discrim_tvars)
        f2f_discrim_train = f2f_discrim_optim.apply_gradients(f2f_discrim_grads_and_vars)

    with tf.name_scope("f2f_generator_train"):
        with tf.control_dependencies([f2f_discrim_train]):
            f2f_gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            f2f_gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            f2f_gen_grads_and_vars = f2f_gen_optim.compute_gradients(f2f_gen_loss, var_list=f2f_gen_tvars)
            f2f_gen_train = f2f_gen_optim.apply_gradients(f2f_gen_grads_and_vars)

    ##################################################
    # Training f2s_D and f2s_G using Adam
    ##################################################
    with tf.name_scope("f2s_discriminator_train"):
        f2s_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        f2s_discrim_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
        f2s_discrim_grads_and_vars = f2s_discrim_optim.compute_gradients(f2s_discrim_loss, var_list=f2s_discrim_tvars)
        f2s_discrim_train = f2s_discrim_optim.apply_gradients(f2s_discrim_grads_and_vars)

    with tf.name_scope("f2s_generator_train"):
        with tf.control_dependencies([f2s_discrim_train]):
            f2s_gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            f2s_gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            f2s_gen_grads_and_vars = f2s_gen_optim.compute_gradients(f2s_gen_loss, var_list=f2s_gen_tvars)
            f2s_gen_train = f2s_gen_optim.apply_gradients(f2s_gen_grads_and_vars)


    ##################################################
    # Training s2f_D and s2f_G using Adam
    ##################################################
    with tf.name_scope("s2f_discriminator_train"):
        s2f_discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        s2f_discrim_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
        s2f_discrim_grads_and_vars = s2f_discrim_optim.compute_gradients(s2f_discrim_loss, var_list=s2f_discrim_tvars)
        s2f_discrim_train = s2f_discrim_optim.apply_gradients(s2f_discrim_grads_and_vars)

    with tf.name_scope("s2f_generator_train"):
        with tf.control_dependencies([s2f_discrim_train]):
            s2f_gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            s2f_gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            s2f_gen_grads_and_vars = s2f_gen_optim.compute_gradients(s2f_gen_loss, var_list=s2f_gen_tvars)
            s2f_gen_train = s2f_gen_optim.apply_gradients(s2f_gen_grads_and_vars)


    ##################################################
    # Weight update f2f network
    ##################################################

    f2f_ema = tf.train.ExponentialMovingAverage(decay=0.99)
    f2f_update_losses = f2f_ema.apply([f2f_disc_real_loss, f2f_disc_fake_loss, f2f_disc_loss_real_styl, f2f_gen_loss_GAN, f2f_gen_loss_L1])

    f2f_global_step = tf.train.get_or_create_global_step()
    f2f_incr_global_step = tf.assign(f2f_global_step, f2f_global_step+1)


    ##################################################
    # Weight update f2s network
    ##################################################

    f2s_ema = tf.train.ExponentialMovingAverage(decay=0.99)
    f2s_update_losses = f2s_ema.apply([f2s_disc_real_loss, f2s_disc_fake_loss, f2s_disc_loss_real_styl, f2s_gen_loss_GAN, f2s_gen_loss_L1])

    f2s_global_step = tf.train.get_or_create_global_step()
    f2s_incr_global_step = tf.assign(f2s_global_step, f2s_global_step+1)

    ##################################################
    # Weight update s2f network
    ##################################################

    s2f_ema = tf.train.ExponentialMovingAverage(decay=0.99)
    s2f_update_losses = s2f_ema.apply([s2f_disc_real_loss, s2f_disc_fake_loss, s2f_disc_loss_real_styl, s2f_gen_loss_GAN, s2f_gen_loss_L1])

    s2f_global_step = tf.train.get_or_create_global_step()
    s2f_incr_global_step = tf.assign(s2f_global_step, s2f_global_step+1)

    return Model(
        f2f_predict_real=f2f_predict_real,
        f2f_predict_fake=f2f_predict_fake,
        f2f_disc_real_loss=f2f_ema.average(f2f_disc_real_loss),
        f2f_disc_fake_loss=f2f_ema.average(f2f_disc_fake_loss),
        f2f_disc_loss_real_styl=f2f_ema.average(f2f_disc_loss_real_styl),
        f2f_discrim_grads_and_vars=f2f_discrim_grads_and_vars,
        f2f_gen_loss_GAN=f2f_ema.average(f2f_gen_loss_GAN),
        f2f_gen_loss_L1=f2f_ema.average(f2f_gen_loss_L1),
        f2f_gen_grads_and_vars=f2f_gen_grads_and_vars,
        f2f_outputs=f2f_outputs,
        f2s_predict_real=f2s_predict_real,
        f2s_predict_fake=f2s_predict_fake,
        f2s_disc_real_loss=f2s_ema.average(f2s_disc_real_loss),
        f2s_disc_fake_loss=f2s_ema.average(f2s_disc_fake_loss),
        f2s_disc_loss_real_styl=f2s_ema.average(f2s_disc_loss_real_styl),
        f2s_discrim_grads_and_vars=f2s_discrim_grads_and_vars,
        f2s_gen_loss_GAN=f2s_ema.average(f2s_gen_loss_GAN),
        f2s_gen_loss_L1=f2s_ema.average(f2s_gen_loss_L1),
        f2s_gen_grads_and_vars=f2s_gen_grads_and_vars,
        f2s_outputs=f2s_outputs,
        s2f_predict_real=s2f_predict_real,
        s2f_predict_fake=s2f_predict_fake,
        s2f_disc_real_loss=s2f_ema.average(s2f_disc_real_loss),
        s2f_disc_fake_loss=s2f_ema.average(s2f_disc_fake_loss),
        s2f_disc_loss_real_styl=s2f_ema.average(s2f_disc_loss_real_styl),
        s2f_discrim_grads_and_vars=s2f_discrim_grads_and_vars,
        s2f_gen_loss_GAN=s2f_ema.average(s2f_gen_loss_GAN),
        s2f_gen_loss_L1=s2f_ema.average(s2f_gen_loss_L1),
        s2f_gen_grads_and_vars=s2f_gen_grads_and_vars,
        s2f_outputs=s2f_outputs,
        train=tf.group(f2f_update_losses, f2f_incr_global_step, f2f_gen_train, f2s_update_losses, f2s_incr_global_step, f2s_gen_train, s2f_update_losses, s2f_incr_global_step, s2f_gen_train),
    )