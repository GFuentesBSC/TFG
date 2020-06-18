from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import tensorflow as tf
import horovod.tensorflow as hvd
import horovod.tensorflow.keras as hvd_keras
import math
from tensorflow.keras.optimizers import get as get_optimizer_by_name


def hvd_adapt_steps(steps):
    return max(1, math.ceil((steps // hvd.size())))


def hvd_adapt_epochs(epochs):
    return max(1, math.ceil((epochs // hvd.size())))


def hvd_adapt_callbacks(callbacks, save_checkpoints):
    hvd_callbacks = [
        hvd_keras.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd_keras.callbacks.MetricAverageCallback(),
    ]
    if (hvd.rank() == 0) and save_checkpoints:
        callbacks.append(ModelCheckpoint("./checkpoint-{epoch}.h5"))
    return hvd_callbacks + callbacks


def hvd_adapt_optimizer(opt):
    if "".__class__.__name__ == opt.__class__.__name__:
        opt = get_optimizer_by_name(opt)
    opt_config = opt.get_config()
    try:
        opt_config["learning_rate"] *= hvd.size()
    except KeyError:
        opt_config["lr"] *= hvd.size()
    return hvd.DistributedOptimizer(opt.from_config(opt_config))


hvd.init()
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], "GPU")
batch_size = 32
epochs = 200
data_augmentation = True
num_classes = 10
subtract_pixel_mean = True
n = 3
version = 1
if version == 1:
    depth = (n * 6) + 2
elif version == 2:
    depth = (n * 9) + 2
model_type = "ResNet%dv%d" % (depth, version)
((x_train, y_train), (x_test, y_test)) = cifar10.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
print("y_train shape:", y_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    lr = 0.001
    if epoch > 180:
        lr *= 0.0005
    elif epoch > 160:
        lr *= 0.001
    elif epoch > 120:
        lr *= 0.01
    elif epoch > 80:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr


def resnet_layer(
    inputs,
    num_filters=16,
    kernel_size=3,
    strides=1,
    activation="relu",
    batch_normalization=True,
    conv_first=True,
):
    conv = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(0.0001),
    )
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    if ((depth - 2) % 6) != 0:
        raise ValueError("depth should be 6n+2 (eg 20, 32, 44 in [a])")
    num_filters = 16
    num_res_blocks = int(((depth - 2) / 6))
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if (stack > 0) and (res_block == 0):
                strides = 2
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            if (stack > 0) and (res_block == 0):
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, y])
            x = Activation("relu")(x)
        num_filters *= 2
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        y
    )
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    if ((depth - 2) % 9) != 0:
        raise ValueError("depth should be 9n+2 (eg 56 or 110 in [b])")
    num_filters_in = 16
    num_res_blocks = int(((depth - 2) / 9))
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = "relu"
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:
                    strides = 2
            y = resnet_layer(
                inputs=x,
                num_filters=num_filters_in,
                kernel_size=1,
                strides=strides,
                activation=activation,
                batch_normalization=batch_normalization,
                conv_first=False,
            )
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(
                inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False
            )
            if res_block == 0:
                x = resnet_layer(
                    inputs=x,
                    num_filters=num_filters_out,
                    kernel_size=1,
                    strides=strides,
                    activation=None,
                    batch_normalization=False,
                )
            x = keras.layers.add([x, y])
        num_filters_in = num_filters_out
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation="softmax", kernel_initializer="he_normal")(
        y
    )
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)
model.compile(
    loss="categorical_crossentropy",
    optimizer=hvd_adapt_optimizer(Adam(learning_rate=lr_schedule(0))),
    metrics=["accuracy"],
)
model.summary()
print(model_type)
save_dir = os.path.join(os.getcwd(), "saved_models")
model_name = "cifar10_%s_model.{epoch:03d}.h5" % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(
    filepath=filepath, monitor="val_acc", verbose=1, save_best_only=True
)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(
    factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=5e-07
)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
if not data_augmentation:
    print("Not using data augmentation.")
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=hvd_adapt_epochs(epochs),
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=hvd_adapt_callbacks(callbacks, True),
        verbose=(1 if (hvd.rank() == 0) else 0),
    )
else:
    print("Using real-time data augmentation.")
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode="nearest",
        cval=0.0,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0,
    )
    datagen.fit(x_train)
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=hvd_adapt_epochs(epochs),
        verbose=(1 if (hvd.rank() == 0) else 0),
        workers=4,
        callbacks=hvd_adapt_callbacks(callbacks, True),
    )
if hvd.rank() == 0:
    scores = model.evaluate(x_test, y_test, verbose=1)
    print("Test loss:", scores[0])
    print("Test accuracy:", scores[1])
