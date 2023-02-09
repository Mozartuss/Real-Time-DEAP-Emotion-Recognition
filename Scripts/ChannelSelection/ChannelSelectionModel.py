import os
from datetime import datetime

import humanize
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.auto import tqdm

from Utils.Constants import FsType, ClassifyType


def build_model(shape=(None, 1)):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=shape))

    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))

    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))

    model.add(tf.keras.layers.LSTM(units=16))

    # model.add(tf.keras.layers.Dense(16, activation="relu"))

    model.add(tf.keras.layers.Dense(8, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    # print(model.summary())
    return model


def build_model_full(shape=(None, 1)):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=shape))
    # model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.LSTM(units=256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.LSTM(units=32))
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(16, activation="relu"))

    model.add(tf.keras.layers.Dense(8, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    # print(model.summary())
    return model


def training(dataset, epochs, batch_size, classify_type: ClassifyType = "", natural_optimizer: FsType = "",
             participant: int = 0,
             full_model: bool = False):
    """
    Training model
    @param full_model: True if the Model is used with all the Participants
    @param participant: Number of Participant
    @param dataset: Dataset for training and testing [x_train, x_test, y_train, y_test]
    @param epochs: Number of epochs
    @param batch_size: Batch size
    @param classify_type: Type of classification [enum ClassifyType]
    @param natural_optimizer: Type of channel selection algorithm [enum FsType]
    @return: none
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU to use
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    # configure the GPU settings

    # initialize tqdm callback with default parameters
    tqdm_callback = tfa.callbacks.TQDMProgressBar(show_epoch_progress=False)

    if participant != 0 and natural_optimizer:
        tqdm.write("Participant {}:\tStart Training with {} and {}...\n".format(participant, classify_type.value,
                                                                                natural_optimizer.value))
    else:

        tqdm.write("Start Training with {}...\n".format(classify_type.value if classify_type else "all"))

    start_time = datetime.now()

    x_train, x_test, y_train, y_test = dataset
    if full_model:
        built_model = build_model_full()
    else:
        built_model = build_model()
    history = None
    if full_model:
        history = built_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                                  validation_data=(x_test, y_test))
    else:
        built_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.25)
    score = built_model.evaluate(x_test, y_test, callbacks=[tqdm_callback], verbose=0)
    acc = score[1]
    loss = score[0]
    tqdm.write("Test accuracy: {:.2%} loss: {}:".format(acc, loss))

    end_time = datetime.now()
    elapsed = end_time - start_time
    tqdm.write("Training took: {}\n".format(humanize.precisedelta(elapsed, minimum_unit="seconds")))

    if history:
        return acc, history
    else:
        return acc


if __name__ == '__main__':
    model = build_model()
    # plot_model(Path(ROOT_PATH, "Models", model), show_shapes=True, show_layer_names=True, to_file="./output.png")
