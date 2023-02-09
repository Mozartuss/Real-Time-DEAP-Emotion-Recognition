import os

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def build_model(shape=(None, 1)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), input_shape=shape))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=32, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.LSTM(units=16))
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True, learning_rate=0.001),
                  loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    print(model.summary())
    return model


def build_model_LSTM_CNN(shape=(None, 1)):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.InputLayer(input_shape=shape))
    model.add(tf.keras.layers.Conv1D(128, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv1D(128, kernel_size=3, activation="relu"))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(4, activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True),
                  loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    print(model.summary())
    return model


def build_model_LSTM(shape=(None, 1)):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=256, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=64, return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=32))
    model.add(tf.keras.layers.Dense(units=16, activation="relu"))
    model.add(tf.keras.layers.Dense(units=4, activation="softmax"))
    model.compile(optimizer="adam", loss=tf.keras.losses.categorical_crossentropy, metrics=["accuracy"])
    print(model.summary())
    return model


def train_model(model, x_train, x_test, y_train, y_test, batch_size, epochs):
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
    return history, model
