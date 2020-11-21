import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import CenterCrop, Rescaling
from tensorflow.keras import layers
from cv2 import cv2 as cv


model_path = './model/'
shape = (150, 150, 3)

(ds_train, ds_test), info = tfds.load(
    "rock_paper_scissors",
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize(image, label):
    return tf.cast(cv.resize(image, (150, 150)), tf.float32) / 255., label


# Training pipline
ds_train = ds_train.map(
    normalize,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

# testing pipline
ds_test = ds_test.map(
    normalize,
    num_parallel_calls=tf.data.experimental.AUTOTUNE
)
ds_test = ds_test.cache()
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


# Input layer
inputs = keras.Input(shape=shape)

# Center crop images


x = layers.Conv2D(64, (3, 3), padding='same', activation="relu")(inputs)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(64, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Conv2D(128, (3, 3), activation="relu")(x)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)

x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(ds_train, epochs=6, validation_data=ds_test)

model.evaluate(ds_test)

keras.models.save_model(
    model,
    model_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None,
)
