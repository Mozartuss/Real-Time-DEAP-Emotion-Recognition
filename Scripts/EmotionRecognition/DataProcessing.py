import os
from pathlib import Path

import numpy as np
import pandas as pd

from Utils.Constants import RAW_DATA_PATH, DEAP_ELECTRODES, EPOC_ELECTRODES, PREPROCESSED_DATA_PATH
from Utils.DataHandler import LoadData
from Utils.Helper import bin_power_optimized


def yield_ten_second_data(files_start: int = 0, files_end: int = 32, best_channels=None):
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
            for buffer in np.arange(0, trial_data.shape[1], 1280):
                ten_sec_data = trial_data[:, buffer: buffer + 1280]
                window_channel_data = []
                for window in range(0, ten_sec_data.shape[1], 16):
                    channel_data = []
                    for channel in range(trial_data.shape[0]):
                        # Slice raw data over 2 sec, at interval of 0.125 sec
                        window_data = ten_sec_data[channel, window: window + 256]
                        x = bin_power_optimized(window_data, np.array([4, 8, 12, 16, 25, 45]), 128)
                        channel_data.append(x[0])
                    channel_data = np.concatenate(channel_data)
                    window_channel_data.append(channel_data)
                yield np.array(window_channel_data), np.array(labels)


def normalize_and_scale(vector):
    vector = np.array(vector)
    mean = np.mean(vector)
    standard_deviation = np.std(vector)
    normalized_vector = (vector - mean) / standard_deviation
    min_val = np.min(normalized_vector)
    max_val = np.max(normalized_vector)
    return (normalized_vector - min_val) / (max_val - min_val)


def generate_ten_sec_data(files_start: int = 0, files_end: int = 32, best_channels=None, overwrite=False):
    label_path = Path(PREPROCESSED_DATA_PATH, "realtime", "labels.npy")
    data_path = Path(PREPROCESSED_DATA_PATH, "realtime", "data.npy")
    if label_path.exists() and data_path.exists() and not overwrite:
        print("Loading data from path: ", os.path.dirname(data_path))
        try:
            labels = np.load(label_path, allow_pickle=True, fix_imports=True)
            data = np.load(data_path, allow_pickle=True, fix_imports=True)
            return data, labels
        except (FileNotFoundError, ValueError):
            print("File exists but is not a valid .npy file")
            print("Overwriting file")
            generate_ten_sec_data(files_start, files_end, best_channels, overwrite=True)
    else:
        print("Generating data")
        overall_data = []
        overall_labels = []
        for data, labels in yield_ten_second_data(files_start, files_end, best_channels):
            data = np.array([normalize_and_scale(channel) for channel in data])
            overall_data.append(data)
            overall_labels.append(labels.repeat(data.shape[0]))
        overall_data = np.concatenate(overall_data)
        overall_data = overall_data.reshape((overall_data.shape[0], overall_data.shape[1], 1))
        overall_labels = np.concatenate(overall_labels)
        df = pd.DataFrame({'labels': overall_labels})
        labels_one_hot = pd.get_dummies(df['labels'])
        labels_one_hot = np.array(labels_one_hot)
        if not os.path.exists(os.path.dirname(data_path)):
            os.makedirs(os.path.dirname(data_path))
        print("Saving data to path: ", os.path.dirname(data_path))
        np.save(data_path, overall_data, allow_pickle=True, fix_imports=True)
        np.save(label_path, labels_one_hot, allow_pickle=True, fix_imports=True)
        return overall_data, labels_one_hot
