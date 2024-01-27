from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *


# Source:
# https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py


def one_side_pad(x):
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.Lambda(lambda x: x[:, :-1, :-1, :])(x)
    return x

def resize_image(inp, s):
    return layers.Lambda(lambda x: tf.keras.backend.resize_images(x, height_factor=s[0], width_factor=s[1], data_format='channels_last', interpolation='bilinear'))(inp)


def depth_pool(inp):
    return layers.Lambda(lambda x: tf.math.reduce_mean(x, 3, keepdims=True))(inp)

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x_cut = x

    x = layers.Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2a')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same', use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=True, kernel_initializer='he_uniform', name=conv_name_base + '2c')(x)
    x = layers.add([x, x_cut])

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x_cut = x

    x = layers.Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2a')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=bn_name_base + '_2_pad')(x)
    x = layers.Conv2D(filters2, kernel_size, strides=strides, use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=True, kernel_initializer='he_uniform', name=conv_name_base + '2c')(x)
    shortcut = layers.Conv2D(filters3, (1, 1), use_bias=True, kernel_initializer='he_uniform', name=conv_name_base + '1')(x_cut)
    x = layers.add([x, shortcut])
    return x


def conv_block_str(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x_cut = x

    x = layers.Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2a')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 0), (1, 0)), name=bn_name_base + '_1_pad')(x)
    x = layers.Conv2D(filters2, kernel_size, strides=strides, use_bias=False, kernel_initializer='he_uniform',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=True, kernel_initializer='he_uniform', name=conv_name_base + '2c')(x)
    shortcut = layers.MaxPooling2D(1, strides=strides, name=conv_name_base + '-1')(x_cut)
    x = layers.add([x, shortcut])
    return x

def conv_block_atrous(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at
                     main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with
    strides=(2,2) and the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.BatchNormalization(name=bn_name_base + '2a')(input_tensor)
    x = layers.Activation('relu')(x)
    x_cut = x

    x = layers.Conv2D(filters1, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2a')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',dilation_rate=(2, 2), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(name=bn_name_base + '2c')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False, kernel_initializer='he_uniform', name=conv_name_base + '2c')(x)
    shortcut = layers.Conv2D(filters3, (1, 1), use_bias=True, kernel_initializer='he_uniform', name=conv_name_base + '1')(x_cut)
    x = layers.add([x, shortcut])

    return x


def supres50_a(img_a):
    x = layers.AveragePooling2D((4, 4), strides=(4, 4))(img_a)
    x = layers.Conv2D(16, (7, 7), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_uniform',
               name='conv1a')(x)

    x = conv_block(x, 3, [32, 32, 128], stage=2, block='aa')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='ab')
    x = conv_block_str(x, 3, [32, 32, 128], stage=2, block='ac')

    x = conv_block_atrous(x, 3, [64, 64, 256], stage=3, block='aa')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='ab')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='ac')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='ad')
    Sa1 = x

    return Sa1

def supres50_b(img_b, Sa1):
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(img_b)
    x = layers.Conv2D(16, (7, 7), strides=(1, 1), padding='same', use_bias=False,
               kernel_initializer='he_uniform', name='conv1b')(x)

    x = layers.BatchNormalization(name='bn_conv1b')(x)
    x = layers.Activation('relu')(x)

    x = conv_block(x, 3, [32, 32, 128], stage=2, block='ba')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='bb')
    x = conv_block_str(x, 3, [32, 32, 128], stage=2, block='bc')

    x = conv_block(x, 3, [64, 64, 256], stage=3, block='ba')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='bb')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='bc')
    x = conv_block_str(x, 3, [64, 64, 256], stage=3, block='bd')

    x = layers.Concatenate(axis=-1)([x, Sa1])

    x = conv_block_atrous(x, 3, [128, 128, 512], stage=4, block='ba')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='bb')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='bc')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='bd')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='be')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='bf')

    Sb1 = x

    return Sb1

def supres50_c(img_c, Sb1):
    x = layers.Conv2D(64, (7, 7), strides=(1, 1), padding='same', use_bias=False, kernel_initializer='he_uniform', name='conv1c')(img_c)

    x = conv_block(x, 3, [32, 32, 128], stage=2, block='ca')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='cb')
    x = conv_block_str(x, 3, [32, 32, 128], stage=2, block='cc')
    #x = one_side_pad(x)

    x = conv_block(x, 3, [64, 64, 256], stage=3, block='ca')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='cb')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='cc')
    x = conv_block_str(x, 3, [64, 64, 256], stage=3, block='cd')

    x = conv_block(x, 3, [128, 128, 512], stage=4, block='ca')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='cb')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='cc')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='cd')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='ce')
    x = conv_block_str(x, 3, [128, 128, 512], stage=4, block='cf')

    x = layers.Concatenate(axis=-1)([x, Sb1])

    x = conv_block_atrous(x, 3, [256, 256, 1024], stage=5, block='ca')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='cb')
    x = identity_block(x, 3, [256, 256, 1024], stage=5, block='cc')
    Sc = x

    return Sc

def poolRes50_attention(img):
    Sa1 = supres50_a(img)
    Sb1 = supres50_b(img, Sa1)
    Sc = supres50_c(img, Sb1)
    return Sa1, Sb1,  Sc