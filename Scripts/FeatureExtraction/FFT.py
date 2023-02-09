from pathlib import Path

import numpy as np

from Utils.Constants import PREPROCESSED_DATA_PATH, DEAP_ELECTRODES
from Utils.Helper import delete_leading_zero, bin_power_optimized


def fft_processing(subject, filename, channels, band, window_size, step_size, sample_rate, overwrite):
    """
    Extract features from the given data using FFT.
    Save the features in numpy files for each participant in the PREPROCESSED_DATA_PATH folder.
    @param subject: participant data
    @param filename: filename of the participant
    @param channels: channels to be used for feature extraction i.e. EPOC_CHANNELS
    @param band: list of bands to be used i.e. [4, 8, 12, 16, 25, 45] for Theta(4-8), Alpha(8-12), LowerBeta(12-16), UpperBeta(16-25), Gamma(25-45)
    @param window_size: window size in data points
    @param step_size: step size in data points
    @param sample_rate: sample rate of the data
    @param overwrite: overwrite existing files
    @return: None
    """
    save_path = PREPROCESSED_DATA_PATH
    p_num = delete_leading_zero(filename.split(".")[0][1:])
    save_file_path = Path(save_path, f"Participant_{p_num}.npy")
    if not save_file_path.exists() or overwrite:
        meta = []
        # loop over 0-39 trails
        for i in range(0, 40):
            trial_data = subject["data"][i]
            # Arousal and Valence
            labels = subject["labels"][i][:2]
            start = 0

            while start + window_size < trial_data.shape[1]:
                meta_array = []
                meta_data = []  # meta vector for analysis
                for j in channels:
                    # Slice raw data over 2 sec, at interval of 0.125 sec
                    if j in DEAP_ELECTRODES:
                        index = DEAP_ELECTRODES.index(j)
                        x = trial_data[index][start: start + window_size]
                        # FFT over 2 sec of channel j, in seq of theta, alpha, low beta, high beta, gamma
                        y = bin_power_optimized(x, band, sample_rate)
                        meta_data.append(np.array(y[0]))

                meta_array.append(np.array(meta_data))
                # label_bin = np.array(labels >= 5).astype(int)
                meta_array.append(labels)

                meta.append(np.array(meta_array, dtype=object))
                start = start + step_size

        meta = np.array(meta)
        if not save_path.exists():
            save_path.mkdir(exist_ok=True)

        np.save(save_file_path, meta, allow_pickle=True, fix_imports=True)
