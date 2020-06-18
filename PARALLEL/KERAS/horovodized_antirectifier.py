from __future__ import print_function
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import tensorflow as tf
import horovod.tensorflow as hvd
import horovod.tensorflow.keras as hvd_keras
import math
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
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


class Antirectifier(layers.Layer):
    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2
        shape[(-1)] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu((-inputs))
        return K.concatenate([pos, neg], axis=1)


batch_size = 128
num_classes = 10
epochs = 40
((x_train, y_train), (x_test, y_test)) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = Sequential()
model.add(layers.Dense(256, input_shape=(784,)))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes))
model.add(layers.Activation("softmax"))
model.compile(
    loss="categorical_crossentropy",
    optimizer=hvd_adapt_optimizer("rmsprop"),
    metrics=["accuracy"],
)
model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=hvd_adapt_epochs(epochs),
    verbose=(1 if (hvd.rank() == 0) else 0),
    validation_data=(x_test, y_test),
    callbacks=hvd_adapt_callbacks([], True),
)
