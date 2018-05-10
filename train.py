# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:53:18 2018

@author: Jakub
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:42:15 2018

@author: Jakub
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:13:45 2018

@author: Jakub
"""
import os
import numpy as  np
import numpy.linalg as LAG
import tensorflow as tf
from PIL import Image
import matplotlib.pylab as plt
import time

tf.reset_default_graph()  

EPS = 1e-12 #epsilon, some very small number
NGF=64 #number of generator filters in first convolutional layer
NDF=64 #number of discriminator filters in first convolutional layer
GAN_WEIGHT=1.0
L1_WEIGHT=100.0
LR=0.002 #Learning rate for Adam
BETA1=0.5 #momentum term of Adam
DROPOUT=0.0

SERVER_PATHS = True

if SERVER_PATHS:
    DATA_TRAIN_DIR  ="/home/shared/pizza/data/pizza_data" 
    DATA_TEST_DIR   ="/home/shared/pizza/data/pizza_test"
    DATA_OUTPUT_DIR ="/home/shared/pizza/output"
    MODEL_DIR       ="/home/shared/pizza/models"
else:
    DATA_TRAIN_DIR  ="C:/Users/Jakub/Desktop/DL_project/pizza_data"
    DATA_TEST_DIR   ="C:/Users/Jakub/Desktop/DL_project/pizza_test"
    DATA_OUTPUT_DIR ="C:/Users/Jakub/Desktop/DL_project/output"
    MODEL_DIR       ="C:/Users/Jakub/Desktop/DL_project/models"

EXAMPLES_TRAIN=100000
EPOCHS=3
SAVE_EACH=1
TEST_EACH=1


#implementation of leaky ReLU
def lrelu(x, a):
    with tf.name_scope("lrelu"):        
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))

def gen_conv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels, separable_conv=False):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)
    
def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, NGF)
        layers.append(output)

    layer_specs = [
        NGF * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        NGF * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        NGF * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        NGF * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        NGF * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        NGF * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        NGF * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (NGF * 8, DROPOUT),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (NGF * 8, DROPOUT),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (NGF * 8, DROPOUT),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (NGF * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (NGF * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (NGF * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (NGF, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
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
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, NDF, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = NDF * min(2**(i+1), 8)
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

        return layers[-1]


#inputs - batch of edges images
#targets - batch of ground truth output images
def create_model(inputs, targets):
    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    with tf.name_scope("discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * GAN_WEIGHT + gen_loss_L1 * L1_WEIGHT

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(LR, BETA1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(LR, BETA1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return gen_train, discrim_train, outputs

input_batch=tf.placeholder(dtype=tf.float32, shape=[None, 256,256,3])
target_batch=tf.placeholder(dtype=tf.float32, shape=[None, 256,256,3])
gen_train, discrim_train,  outputs=create_model(input_batch, target_batch)



"""
im = plt.imread("C:/Users/Jakub/pix2pix-tensorflow/facades/train/1.jpg")
im=np.asarray(im, dtype=np.float32)
im/=256

im1=im[:,256:512,:]
im2=im[:,0:256,:]


batch1=np.zeros((1,256,256,3))
batch2=np.zeros((1,256,256,3))
batch1[0,:,:,:]=im1
batch2[0,:,:,:]=im2
"""

batches=[]
batches_test=[]
loaded=0

for filename in os.listdir(DATA_TRAIN_DIR):   
    batch_input=np.zeros((1,256,256,3), dtype=np.float32)
    batch_target=np.zeros((1,256,256,3), dtype=np.float32)
    im=plt.imread(DATA_TRAIN_DIR+"/"+filename)
    im=np.asarray(im, dtype=np.float32)
    im/=256
    batch_input[0,:,:,:]= im[:,256:512,:]
    batch_target[0,:,:,:]=im[:,0:256,:]
    batches.append([batch_input,batch_target])
    
    loaded+=1
    if loaded>EXAMPLES_TRAIN:
        break
    
for filename in os.listdir(DATA_TEST_DIR): 
    batch_input=np.zeros((1,256,256,3), dtype=np.float32)
    batch_target=np.zeros((1,256,256,3), dtype=np.float32)
    im=plt.imread(DATA_TEST_DIR+"/"+filename)
    im=np.asarray(im, dtype=np.float32)
    im/=256
    batch_input[0,:,:,:]= im[:,256:512,:]
    batch_target[0,:,:,:]=im[:,0:256,:]
    batches_test.append([batch_input,batch_target])
    
    
    

        
saver = tf.train.Saver()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())   
    for epoch in range(EPOCHS):           
        print("epoch",epoch) 
        b=0
        for batch in batches: 
            print("batch:",b)
            b+=1
            
            gen_train.run(feed_dict={input_batch: batch[0], target_batch: batch[1]})
            discrim_train.run(feed_dict={input_batch: batch[0], target_batch: batch[1]})
            
       
        if epoch%SAVE_EACH==0:            
            saver.save(sess,MODEL_DIR+"/model"+str(epoch)+".ckpt") 
            
        if epoch%TEST_EACH==0:
            i=0
            for batch in batches_test:
                print("Processing test image ",i)
                ret=np.zeros((256,512,3))
                o=outputs.eval(feed_dict={input_batch: batch[0]})
                gt=np.copy(batch[0])
                ret[:,0:256,:]=o
                ret[:,256:512,:]=gt
                ret*=256
                ret=np.asarray(ret, dtype=np.uint8)
                img = Image.fromarray(ret, 'RGB')
                img.save(DATA_OUTPUT_DIR+"/"+str(i)+'.jpg')
                i+=1
                


