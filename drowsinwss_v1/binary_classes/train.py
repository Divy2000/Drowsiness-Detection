import math
import os
import random
from glob import glob

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint

# Dataset Parameters - CHANGE HERE

# Change path to the folder where images are extracted from the video
data_dir = "/Users/divy/Downloads/images"

# Image Parameters
N_CLASSES = 2  # CHANGE HERE, total number of classes
IMG_HEIGHT = 512  # CHANGE HERE, the image height to be resized to
IMG_WIDTH = 512  # CHANGE HERE, the image width to be resized to
CHANNELS = 3  # The 3 color channels, change to 1 if grayscale


def make_dataset(path, batch_size):
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=CHANNELS)
        image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
        return image

    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    classes = os.listdir(path)[1:]
    filenames = glob(path + '/0/*')
    filenames = filenames + glob(path + '/10/*')
    random.shuffle(filenames)
    labels = [classes.index(name.split('/')[-2]) for name in filenames]
    filenames_ds = tf.data.Dataset.from_tensor_slices(filenames)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((images_ds, labels_ds))
    ds = configure_for_performance(ds)

    return ds


# def build_model(num_classes):
#     if K.image_data_format() == 'channels_first':
#         input_shape = (3, IMG_HEIGHT, IMG_WIDTH)
#     else:
#         input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
#
#     model = Sequential()
#     model.add(Conv2D(16, (3, 3), input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.15))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(64))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])
#     return model


def build_model(num_classes):
    if K.image_data_format() == 'channels_first':
        input_shape = (3, IMG_HEIGHT, IMG_WIDTH)
    else:
        input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def use_tf_data(path):
    batch_size = 32
    dataset = make_dataset(path, batch_size)
    num_images = len(glob(path + '/*/*'))
    train_size = int(0.8 * num_images)
    train_num_images = 0.8 * num_images
    val_num_images = 0.2 * num_images
    train_dataset = dataset.take(train_size)
    test_dataset = dataset.skip(train_size)
    num_classes = len(os.listdir(path)) - 2
    model = build_model(num_classes)

    checkpoint = ModelCheckpoint("`binary_classes/model_binary.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',
                                 save_weights_only=True)

    history = model.fit(train_dataset, validation_data=test_dataset, batch_size=batch_size, epochs=5,
                        callbacks=[checkpoint],
                        verbose=1,
                        steps_per_epoch=math.ceil(train_num_images / batch_size),
                        validation_steps=math.ceil(val_num_images / batch_size))

    train_loss, train_acc = model.evaluate(train_dataset, batch_size=batch_size, verbose=1,
                                           steps=math.ceil(train_num_images / batch_size))
    val_loss, val_acc = model.evaluate(test_dataset, batch_size=batch_size, verbose=1,
                                       steps=math.ceil(val_num_images / batch_size))
    print(f"train_loss = {train_loss}, train_acc = {train_acc * 100}%")
    print(f"val_loss = {val_loss}, val_acc = {val_acc * 100}%")
    return history


history = use_tf_data(data_dir)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("binary_classes/accuracy_graph.png")
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig("binary_classes/loss_graph.png")
plt.close()
