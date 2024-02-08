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
#import tensorflow_advanced_segmentation_models as tasm
tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()


if os.name == 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def random_adjust_saturation(image, min_delta=0.8, max_delta=2.0, max_delta_hue=0.1,seed=None):
    delta = tf.random.uniform([], -max_delta_hue, max_delta_hue, seed=seed)
    image = tf.image.adjust_hue(image / 255.0, delta)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    saturation_factor = tf.random.uniform([], min_delta, max_delta, seed=seed)
    image = tf.image.adjust_saturation(image, saturation_factor)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image

np.random.seed(2023)
tf.random.set_seed(2023)
rn.seed(2023)

# Step 0: load new dataset
input_dir = r'T:\project\tcga_tnbc\public_train\patch768_384\image'
target_dir = r'T:\project\tcga_tnbc\public_train\patch768_384\maskPng'
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


def data_generator_bin(image_generator_c, mask_generator):
    while True:
        yield (image_generator_c.next(), mask_generator.next())

def data_generator(image_generator_c, mask_generator, nClasses=2):
    while True:
        yield (image_generator_c.next(), tf.one_hot(tf.cast(tf.squeeze(mask_generator.next(), axis=3), dtype=tf.int32), nClasses))

# Step 1: Load the pre-trained model
pretrained_model=load_model(r'T:\pipelines\artemis\artemis_tme_sum12_e50_sCE_img768_penmark636.h5', custom_objects={'tf': tf}, compile=False)
modelpath = "TMElung_artemisTcgaAll_sum12_e60_sCE_img768x20penmark" + ".h5"
# Step 2: Replace the top layers (if necessary)
o = pretrained_model.layers[-2].output
o = Dense(nClasses, activation='softmax')(o)
finetuned_model = keras.Model(inputs=pretrained_model.input, outputs=o)


# Step 3: Compile the model
#metrics = [tasm.metrics.IOUScore(threshold=0.5)]
categorical_focal_dice_loss = ['sparse_categorical_crossentropy']
finetuned_model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_focal_dice_loss,
                         metrics='sparse_categorical_accuracy')



# Fine-tune the model
hist = finetuned_model.fit(
    data_generator_bin(image_generator, mask_generator),
    steps_per_epoch=int(np.ceil(train_samples/batch_size)),
    epochs=60, verbose=1)

finetuned_model.save(modelpath)