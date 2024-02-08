import json
import glob
import six
import numpy as np
import tensorflow as tf
import random as rn
import os
#from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
#from sklearn.metrics import roc_auc_score, confusion_matrix
from pandas import DataFrame
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.optimizers import SGD, RMSprop, Adadelta, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
import platform
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Self3CrossAttention_Res507v2 import selfCrossPooling
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

np.random.seed(2023)
tf.random.set_seed(2023)
rn.seed(2023)


def scheduler(epoch, lr):
    if epoch <10:
        print("lr changed to {}".format(lr))
        return lr

    elif epoch >= 10 and epoch <50:
        print("lr changed to {}".format(0.0001))
        return 0.0001

    else:
        print("lr changed to {}".format(0.00001))
        return 0.00001

def to_categorical_mask(multi_label, nClasses):
    categorical_mask = np.zeros((multi_label.shape[0], multi_label.shape[1], nClasses))
    for c in range(nClasses):
        categorical_mask[:, :, c] = (multi_label == c).astype(int)
    categorical_mask = np.reshape(categorical_mask, (multi_label.shape[0] * multi_label.shape[1], nClasses))
    return categorical_mask


def random_adjust_saturation(image, min_delta=0.8, max_delta=2.0, max_delta_hue=0.2,seed=None):
    delta = tf.random.uniform([], -max_delta_hue, max_delta_hue, seed=seed)
    image = tf.image.adjust_hue(image / 255.0, delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    saturation_factor = tf.random.uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image, saturation_factor)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

input_dir = r'T:\project\tcga_tnbc\public_train\patch768_tcga20x\image'
target_dir = r'T:\project\tcga_tnbc\public_train\patch768_tcga20x\maskPng'
img_size = (384, 384)
nClasses = 8
batch_size = 8


input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".png")
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png")
    ]
)

print("Number of samples:", len(input_img_paths))

img = keras.Input(shape=(384, 384, 3))

o = selfCrossPooling(nClasses, img)
o = (layers.Activation('softmax'))(o)
stromaPSmodel = keras.Model(img, o)
stromaPSmodel.summary()

train_samples = len(input_img_paths)

rn.Random(2023).shuffle(input_img_paths)
rn.Random(2023).shuffle(target_img_paths)
train_input_img_paths = input_img_paths
train_target_img_paths = target_img_paths

df_train = DataFrame(train_input_img_paths,columns=['filename'])
df_train_target = DataFrame(train_target_img_paths,columns=['filename'])

# we create two instances with the same arguments-rotation_range, width/height_shift_range, zoom_range, fill_mode
data_gen_args = dict(rotation_range=90,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     fill_mode='constant',
                     preprocessing_function=random_adjust_saturation)

data_gen_args_mask = dict(rescale=1.,
                     rotation_range=90,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=0.2,
                     fill_mode='constant'
                     )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args_mask)
# Provide the same seed and keyword arguments to the fit and flow methods
seed = 2023
image_generator = image_datagen.flow_from_dataframe(
    df_train,
    target_size=img_size,
    class_mode=None,
    batch_size=batch_size,
    seed=seed)

mask_generator = mask_datagen.flow_from_dataframe(
    df_train_target,
    target_size=img_size,
    color_mode='grayscale',
    class_mode=None,
    batch_size=batch_size,
    seed=seed)




# combine generators into one which yields image and masks
#train_generator = (pair for pair in zip(image_generator_a, image_generator_b,  image_generator_c, mask_generator))
def data_generator_bin(image_generator_c, mask_generator):
    while True:
        yield (image_generator_c.next(), mask_generator.next())

def data_generator(image_generator_c, mask_generator, nClasses=2):
    while True:
        yield (image_generator_c.next(), tf.one_hot(tf.cast(tf.squeeze(mask_generator.next(), axis=3), dtype=tf.int32), nClasses))


# combine generators into one which yields image and masks
#sgd = SGD(lr=0.001, momentum=0.9, decay=0, nesterov=True)
adam = Adam(lr=0.001)
categorical_focal_dice_loss = ['sparse_categorical_crossentropy']
stromaPSmodel.compile(optimizer=adam, loss=categorical_focal_dice_loss,
                         metrics='sparse_categorical_accuracy') #sample_weight_mode="temporal",

modelpath = "TMEseg_tcgaScratch20x_resnetby2_sum6_e5060_sCE" + ".h5"  #train: using training set percentage, otherwise, using ALL
callbacks = [LearningRateScheduler(scheduler)]


hist = stromaPSmodel.fit(
    data_generator_bin(image_generator, mask_generator),
    steps_per_epoch=int(np.ceil(train_samples/batch_size)),
    #validation_steps=int(np.ceil(val_samples/batch_size)),
    callbacks=callbacks,
    epochs=60, verbose=1)

stromaPSmodel.save(modelpath)

