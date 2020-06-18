from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import pylab as plt
import tensorflow as tf
import horovod.tensorflow as hvd
import horovod.tensorflow.keras as hvd_keras
import math
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
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
seq = Sequential()
seq.add(
    ConvLSTM2D(
        filters=40,
        kernel_size=(3, 3),
        input_shape=(None, 40, 40, 1),
        padding="same",
        return_sequences=True,
    )
)
seq.add(BatchNormalization())
seq.add(
    ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)
)
seq.add(BatchNormalization())
seq.add(
    ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)
)
seq.add(BatchNormalization())
seq.add(
    ConvLSTM2D(filters=40, kernel_size=(3, 3), padding="same", return_sequences=True)
)
seq.add(BatchNormalization())
seq.add(
    Conv3D(
        filters=1,
        kernel_size=(3, 3, 3),
        activation="sigmoid",
        padding="same",
        data_format="channels_last",
    )
)
seq.compile(loss="binary_crossentropy", optimizer=hvd_adapt_optimizer("adadelta"))


def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    for i in range(n_samples):
        n = np.random.randint(3, 8)
        for j in range(n):
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1
            w = np.random.randint(2, 4)
            for t in range(n_frames):
                x_shift = xstart + (directionx * t)
                y_shift = ystart + (directiony * t)
                noisy_movies[
                    i,
                    t,
                    (x_shift - w) : (x_shift + w),
                    (y_shift - w) : (y_shift + w),
                    0,
                ] += 1
                if np.random.randint(0, 2):
                    noise_f = (-1) ** np.random.randint(0, 2)
                    noisy_movies[
                        i,
                        t,
                        ((x_shift - w) - 1) : ((x_shift + w) + 1),
                        ((y_shift - w) - 1) : ((y_shift + w) + 1),
                        0,
                    ] += (noise_f * 0.1)
                x_shift = xstart + (directionx * (t + 1))
                y_shift = ystart + (directiony * (t + 1))
                shifted_movies[
                    i,
                    t,
                    (x_shift - w) : (x_shift + w),
                    (y_shift - w) : (y_shift + w),
                    0,
                ] += 1
    noisy_movies = noisy_movies[:, :, 20:60, 20:60, :]
    shifted_movies = shifted_movies[:, :, 20:60, 20:60, :]
    noisy_movies[(noisy_movies >= 1)] = 1
    shifted_movies[(shifted_movies >= 1)] = 1
    return (noisy_movies, shifted_movies)


(noisy_movies, shifted_movies) = generate_movies(n_samples=1200)
seq.fit(
    noisy_movies[:1000],
    shifted_movies[:1000],
    batch_size=10,
    epochs=hvd_adapt_epochs(300),
    validation_split=0.05,
    callbacks=hvd_adapt_callbacks([], True),
    verbose=(1 if (hvd.rank() == 0) else 0),
)
which = 1004
track = noisy_movies[which][:7, :, :, :]
for j in range(16):
    if hvd.rank() == 0:
        new_pos = seq.predict(track[np.newaxis, :, :, :, :])
        new = new_pos[:, (-1), :, :, :]
    track = np.concatenate((track, new), axis=0)
track2 = noisy_movies[which][:, :, :, :]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    if i >= 7:
        ax.text(1, 3, "Predictions !", fontsize=20, color="w")
    else:
        ax.text(1, 3, "Initial trajectory", fontsize=20)
    toplot = track[i, :, :, 0]
    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, "Ground truth", fontsize=20)
    toplot = track2[i, :, :, 0]
    if i >= 2:
        toplot = shifted_movies[which][(i - 1), :, :, 0]
    plt.imshow(toplot)
    plt.savefig(("%i_animate.png" % (i + 1)))
