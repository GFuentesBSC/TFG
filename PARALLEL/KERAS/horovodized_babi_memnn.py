from __future__ import print_function
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Input, Activation, Dense, Permute, Dropout
from tensorflow.keras.layers import add, dot, concatenate
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import get_file
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re
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


def tokenize(sent):
    return [x.strip() for x in re.split("(\\W+)?", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        line = line.decode("utf-8").strip()
        (nid, line) = line.split(" ", 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line:
            (q, a, supporting) = line.split("\t")
            q = tokenize(q)
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[(i - 1)] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append("")
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce((lambda x, y: (x + y)), data)
    data = [
        (flatten(story), q, answer)
        for (story, q, answer) in data
        if ((not max_length) or (len(flatten(story)) < max_length))
    ]
    return data


def vectorize_stories(data):
    (inputs, queries, answers) = ([], [], [])
    for (story, query, answer) in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    return (
        pad_sequences(inputs, maxlen=story_maxlen),
        pad_sequences(queries, maxlen=query_maxlen),
        np.array(answers),
    )


try:
    path = get_file(
        "babi-tasks-v1-2.tar.gz",
        origin="https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz",
    )
except:
    print(
        "Error downloading dataset, please download it manually:\n$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz"
    )
    raise
challenges = {
    "single_supporting_fact_10k": "tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt",
    "two_supporting_facts_10k": "tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt",
}
challenge_type = "single_supporting_fact_10k"
challenge = challenges[challenge_type]
print("Extracting stories for the challenge:", challenge_type)
with tarfile.open(path) as tar:
    train_stories = get_stories(tar.extractfile(challenge.format("train")))
    test_stories = get_stories(tar.extractfile(challenge.format("test")))
vocab = set()
for (story, q, answer) in train_stories + test_stories:
    vocab |= set(((story + q) + [answer]))
vocab = sorted(vocab)
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for (x, _, _) in (train_stories + test_stories))))
query_maxlen = max(map(len, (x for (_, x, _) in (train_stories + test_stories))))
print("-")
print("Vocab size:", vocab_size, "unique words")
print("Story max length:", story_maxlen, "words")
print("Query max length:", query_maxlen, "words")
print("Number of training stories:", len(train_stories))
print("Number of test stories:", len(test_stories))
print("-")
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print("-")
print("Vectorizing the word sequences...")
word_idx = dict(((c, (i + 1)) for (i, c) in enumerate(vocab)))
(inputs_train, queries_train, answers_train) = vectorize_stories(train_stories)
(inputs_test, queries_test, answers_test) = vectorize_stories(test_stories)
print("-")
print("inputs: integer tensor of shape (samples, max_length)")
print("inputs_train shape:", inputs_train.shape)
print("inputs_test shape:", inputs_test.shape)
print("-")
print("queries: integer tensor of shape (samples, max_length)")
print("queries_train shape:", queries_train.shape)
print("queries_test shape:", queries_test.shape)
print("-")
print("answers: binary (1 or 0) tensor of shape (samples, vocab_size)")
print("answers_train shape:", answers_train.shape)
print("answers_test shape:", answers_test.shape)
print("-")
print("Compiling...")
input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=64))
input_encoder_m.add(Dropout(0.3))
input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
input_encoder_c.add(Dropout(0.3))
question_encoder = Sequential()
question_encoder.add(
    Embedding(input_dim=vocab_size, output_dim=64, input_length=query_maxlen)
)
question_encoder.add(Dropout(0.3))
input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)
match = dot([input_encoded_m, question_encoded], axes=(2, 2))
match = Activation("softmax")(match)
response = add([match, input_encoded_c])
response = Permute((2, 1))(response)
answer = concatenate([response, question_encoded])
answer = LSTM(32)(answer)
answer = Dropout(0.3)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation("softmax")(answer)
model = Model([input_sequence, question], answer)
model.compile(
    optimizer=hvd_adapt_optimizer("rmsprop"),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    [inputs_train, queries_train],
    answers_train,
    batch_size=32,
    epochs=hvd_adapt_epochs(120),
    validation_data=([inputs_test, queries_test], answers_test),
    callbacks=hvd_adapt_callbacks([], True),
    verbose=(1 if (hvd.rank() == 0) else 0),
)
