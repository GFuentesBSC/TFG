from __future__ import print_function
import numpy as np
from keras.layers import Input, Convolution1D, Dot, Dense, Activation, Concatenate
from keras.models import Model
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
batch_size = 64
epochs = 100
num_samples = 10000
data_path = "fra-eng/fra.txt"
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
for line in lines[: min(num_samples, (len(lines) - 1))]:
    (input_text, target_text) = line.split("\t")
    target_text = ("\t" + target_text) + "\n"
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)
input_token_index = dict([(char, i) for (i, char) in enumerate(input_characters)])
target_token_index = dict([(char, i) for (i, char) in enumerate(target_characters)])
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
)
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
for (i, (input_text, target_text)) in enumerate(zip(input_texts, target_texts)):
    for (t, char) in enumerate(input_text):
        encoder_input_data[(i, t, input_token_index[char])] = 1.0
    for (t, char) in enumerate(target_text):
        decoder_input_data[(i, t, target_token_index[char])] = 1.0
        if t > 0:
            decoder_target_data[(i, (t - 1), target_token_index[char])] = 1.0
encoder_inputs = Input(shape=(None, num_encoder_tokens))
x_encoder = Convolution1D(256, kernel_size=3, activation="relu", padding="causal")(
    encoder_inputs
)
x_encoder = Convolution1D(
    256, kernel_size=3, activation="relu", padding="causal", dilation_rate=2
)(x_encoder)
x_encoder = Convolution1D(
    256, kernel_size=3, activation="relu", padding="causal", dilation_rate=4
)(x_encoder)
decoder_inputs = Input(shape=(None, num_decoder_tokens))
x_decoder = Convolution1D(256, kernel_size=3, activation="relu", padding="causal")(
    decoder_inputs
)
x_decoder = Convolution1D(
    256, kernel_size=3, activation="relu", padding="causal", dilation_rate=2
)(x_decoder)
x_decoder = Convolution1D(
    256, kernel_size=3, activation="relu", padding="causal", dilation_rate=4
)(x_decoder)
attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
attention = Activation("softmax")(attention)
context = Dot(axes=[2, 1])([attention, x_encoder])
decoder_combined_context = Concatenate(axis=(-1))([context, x_decoder])
decoder_outputs = Convolution1D(64, kernel_size=3, activation="relu", padding="causal")(
    decoder_combined_context
)
decoder_outputs = Convolution1D(64, kernel_size=3, activation="relu", padding="causal")(
    decoder_outputs
)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
model.compile(optimizer=hvd_adapt_optimizer("adam"), loss="categorical_crossentropy")
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=batch_size,
    epochs=hvd_adapt_epochs(epochs),
    validation_split=0.2,
    callbacks=hvd_adapt_callbacks([], True),
    verbose=(1 if (hvd.rank() == 0) else 0),
)
if hvd.rank() == 0:
    model.save("cnn_s2s.h5")
reverse_input_char_index = dict(((i, char) for (char, i) in input_token_index.items()))
reverse_target_char_index = dict(
    ((i, char) for (char, i) in target_token_index.items())
)
nb_examples = 100
in_encoder = encoder_input_data[:nb_examples]
in_decoder = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
in_decoder[:, 0, target_token_index["\t"]] = 1
predict = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
for i in range((max_decoder_seq_length - 1)):
    if hvd.rank() == 0:
        predict = model.predict([in_encoder, in_decoder])
        predict = predict.argmax(axis=(-1))
        predict_ = predict[:, i].ravel().tolist()
    for (j, x) in enumerate(predict_):
        in_decoder[(j, (i + 1), x)] = 1
for seq_index in range(nb_examples):
    output_seq = predict[seq_index, :].ravel().tolist()
    decoded = []
    for x in output_seq:
        if reverse_target_char_index[x] == "\n":
            break
        else:
            decoded.append(reverse_target_char_index[x])
    decoded_sentence = "".join(decoded)
    print("-")
    print("Input sentence:", input_texts[seq_index])
    print("Decoded sentence:", decoded_sentence)
