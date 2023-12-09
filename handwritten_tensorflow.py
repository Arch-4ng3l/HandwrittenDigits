from keras.layers.attention.multi_head_attention import activation
import tensorflow as tf
import pandas as pd
import numpy as np
from play import start


mnist = tf.keras.datasets.mnist

(train_image, train_labels), (test_images, test_labels) = mnist.load_data()
train_image = train_image / 255.0

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(28, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(48, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(48, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.fit(train_image, train_labels, epochs=5)
loss, acc = model.evaluate(test_images, test_labels)
model.save("model.tf")


def make_prediction(arr):
    arr = np.array(arr)
    arr = arr.reshape((28, 28))
    arr = np.expand_dims(arr, axis=0)
    cl = np.argmax(model.predict(arr)[0])
    return cl


start(make_prediction)
