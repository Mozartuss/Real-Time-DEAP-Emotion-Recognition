import sys
import time
import logging
from pathlib import Path
import tensorflow as tf

import numpy as np
from numpysocket import NumpySocket

from Utils.Constants import DEAP_ELECTRODES, RAW_DATA_PATH, ROOT_PATH
from Utils.DataHandler import LoadData

logger = logging.getLogger('EEG Data Sender')
logger.setLevel(logging.INFO)


def yield_x_seconds_raw_data(files_start: int = 0, files_end: int = 32, best_channels=None, seconds: int = 10):
    # Load Data and split into 10 second parts with 1280 datapoints (10*128Hz)
    if best_channels is None:
        # best_channels = EPOC_ELECTRODES
        best_channels = ['P7', 'FC5', 'T7', 'T8', 'F4', 'O1', 'AF3']
    best_channel_indexes = [DEAP_ELECTRODES.index(channel) for channel in best_channels]
    load_data = LoadData(RAW_DATA_PATH)
    labels_map = {(0, 0): "LALV", (0, 1): "LAHV", (1, 0): "HALV", (1, 1): "HAHV"}
    for filename, data in load_data.yield_raw_data(files_start, files_end):
        # loop over 0-39 trails
        for i in range(0, 40):
            # delete the leading 3 seconds baseline data
            trial_data = data["data"][i][best_channel_indexes, 384:]
            # Transform Arousal and Valence labels into 4 classes
            labels = np.array(data["labels"][i][:2] > 5).astype(int)
            labels = labels_map[tuple(labels)]
            for buffer in np.arange(0, trial_data.shape[1], seconds * 128):
                ten_sec_data = trial_data[:, buffer: buffer + seconds * 128]
                yield ten_sec_data, np.array(labels)


def convert_model(model_path: str):
    path = Path(ROOT_PATH, model_path)
    save_path = Path(ROOT_PATH, f"{model_path}.tflite")
    # check if model exists and if tflite model does not exist
    if not path.exists():
        if save_path.exists():
            logger.info(f"model in {model_path} does not exist but tflite model exists")
            return
        else:
            logger.warning(f"model in {model_path} does not exist and cannot be converted")
            return
    elif path.exists():
        if save_path.exists():
            logger.info(f"model in {model_path} exists and is already converted")
            return
        else:
            logger.info(f"model in {model_path} exists and tflite model does not exist")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(Path(ROOT_PATH, model_path)))
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            logger.info(f"model in {model_path} converted")
            with open(str(save_path), 'wb') as f:
                f.write(tflite_model)
            logger.info(f"model in {model_path} saved")
    else:
        logger.warning(f"model in {model_path} does not exist and cannot be converted")


def send_data():
    host = '<ip of the machin>'
    #host = "localhost"
    port = 12345

    s = NumpySocket()
    s.connect((host, port))
    interval = 4
    while True:
        for data, label in yield_x_seconds_raw_data(seconds=interval):
            send_tuple = np.array((data, label))
            logger.info("sending numpy array:")
            s.sendall(send_tuple)
            time.sleep(interval)


if __name__ == '__main__':
    # convert_model("Real_Time_Model_CNN_LSTM_BIG")
    send_data()
