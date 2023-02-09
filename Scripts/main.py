from datetime import datetime
from pathlib import Path

import humanize
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import normalize
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from ChannelSelection.ChannelSelectionModel import training
from ChannelSelection.PrepareDataset import prepare_dataset
from EmotionRecognition.DataProcessing import generate_ten_sec_data
from EmotionRecognition.Models import train_model, build_model_LSTM_CNN
from EmotionRecognition.Utils import plot_confusion_matrix, plot_train_history
from FeatureExtraction.FFT import fft_processing
from FeatureSelection.MRMR import use_mrmr
from FeatureSelection.PCA import use_pca
from FeatureSelection.SwarmWrapper import use_swarm_based_fs
from Utils.Constants import RAW_DATA_PATH, EPOC_ELECTRODES, FsType, ClassifyType, save_channel_selection_path, \
    ROOT_PATH, PREPROCESSED_DATA_PATH
from Utils.DataHandler import LoadData

opts_config = {
    FsType.PSO: {'k': 14, 'N': 14, 'T': 14, 'c1': 1.49445, 'c2': 1.49445, 'w': 0.7298},
    FsType.BAT: {'k': 14, 'N': 14, 'T': 14, },
    FsType.CS: {'k': 14, 'N': 14, 'T': 14, 'Pa': 0.25},
    FsType.GWO: {'k': 14, 'N': 14, 'T': 14},
    FsType.TMGWO: {'k': 14, 'N': 14, 'T': 14, 'Mp': 0.5},
    FsType.SSA: {'k': 14, 'N': 14, 'T': 14},
    FsType.ISSA: {'k': 14, 'N': 14, 'T': 14, 'maxLt': 7},
}


def feature_extraction():
    load_data = LoadData(RAW_DATA_PATH)
    tqdm.write("Start Feature Extraction")
    start_time = datetime.now()
    for filename, data in load_data.yield_raw_data():
        fft_processing(subject=data,
                       filename=filename,
                       channels=EPOC_ELECTRODES,
                       band=[4, 8, 12, 16, 25, 45],
                       window_size=256,
                       step_size=16,
                       sample_rate=128,
                       overwrite=False)
    end_time = datetime.now()
    elapsed = end_time - start_time
    tqdm.write("Feature Extraction took: {} \n".format(humanize.precisedelta(elapsed, minimum_unit="seconds")))


def feature_selection(fs_type: FsType, components, classify_type: ClassifyType = None, n_channel: int = 14,
                      n_freq: int = 5, participant_list: [int] = range(1, 33)):
    tqdm.write("Start Feature Selection")
    start_time = datetime.now()
    if fs_type == FsType.PCA:
        use_pca(participant_list=participant_list, n_channel=n_channel, n_freq=n_freq, components=components)
    elif fs_type == FsType.MRMR:
        use_mrmr(participant_list=participant_list, n_channel=n_channel, n_freq=n_freq, components=components,
                 classify_type=classify_type)
    else:
        use_swarm_based_fs(natural_optimizer=fs_type, participant_list=participant_list, n_channel=n_channel,
                           classify_type=classify_type, components=components, n_freq=n_freq, opts=opts_config[fs_type])
    end_time = datetime.now()
    elapsed = end_time - start_time
    tqdm.write("Feature Selection took: {}\n".format(humanize.precisedelta(elapsed, minimum_unit="seconds")))


def pre_channel_selection(participant_list: [int] = range(1, 33)):
    """
    Pre Channel Selection for all participants and all classifiers and all feature selection methods
    Save the result in a csv file
    @param participant_list: list of participants (1 - 33)
    @return: None
    """

    feature_extraction()

    for optimizer in FsType:
        mean_acc = []
        for classify_type in ClassifyType:
            feature_selection(fs_type=optimizer, classify_type=classify_type, components=7,
                              participant_list=participant_list)
            acc_list = []
            for participant in participant_list:
                dataset = prepare_dataset(classify_type=classify_type, natural_optimizer=optimizer,
                                          participant=participant, test_size=0.25)
                acc = training(dataset=dataset, classify_type=classify_type, natural_optimizer=optimizer,
                               batch_size=128, epochs=48, participant=participant)
                acc_list.append(acc)
                tf.keras.backend.clear_session()
            if optimizer != FsType.PCA:
                file_path = Path(save_channel_selection_path(optimizer.value),
                                 f"{optimizer.value}_channels_{classify_type.value}.csv")
                df = pd.read_csv(file_path)
                df["accuracy"] = acc_list

                list_of_channels = []
                for channel in df['channels'].values:
                    obj = channel.replace("'", "").strip('][').split(', ')
                    list_of_channels.append(obj)
                df['channels'] = list_of_channels

                channel_accuracy = {}
                for index, row in df.iterrows():
                    for channel in row['channels']:
                        if channel not in channel_accuracy:
                            channel_accuracy[channel] = row['accuracy'] / 7
                        else:
                            channel_accuracy[channel] += row['accuracy'] / 7

                print(channel_accuracy)

                best_channels = [channel for channel, accuracy in
                                 sorted(channel_accuracy.items(), key=lambda item: item[1], reverse=True)[:7]]

                print(f"Best channels for {optimizer.value} {classify_type.value} are {best_channels}")
                final_channel_path = Path(ROOT_PATH, "final_channel_list.csv")
                if final_channel_path.exists():
                    final_channel_df = pd.read_csv(final_channel_path)
                    final_channel_df = final_channel_df.append(
                        {"optimizer": optimizer.value, 'classify_type': classify_type.value,
                         'channels': best_channels}, ignore_index=True)
                else:
                    final_channel_df = pd.DataFrame(
                        {"optimizer": optimizer.value, 'classify_type': classify_type.value,
                         'channels': [best_channels]})
                final_channel_df.to_csv(final_channel_path, index=False)
                df.to_csv(file_path, index=False)


def final_channel_selection():
    best_channel_list = pd.read_csv(Path(ROOT_PATH, "final_channel_list.csv"))
    best_channel_list["accuracy"] = None

    for index, row in best_channel_list.iterrows():
        classify_type = ClassifyType.Arousal if row["classify_type"] == "Arousal" else ClassifyType.Valence
        participant_list = range(1, 33)
        channel_selection = row["channels"].replace("'", "").strip('][').split(', ')
        overall_x = []
        overall_y = []
        for participant in participant_list:
            sub = np.load(str(Path(PREPROCESSED_DATA_PATH, f"Participant_{participant}.npy")), allow_pickle=True)
            data = []
            label = []
            for i in range(0, sub.shape[0]):
                data.append(np.array(sub[i][0]))
                label.append(np.array(sub[i][1]))
            data = np.array(data)
            label = np.array(label)
            epoc_indexes = [EPOC_ELECTRODES.index(channel) for channel in channel_selection]
            data = data[:, epoc_indexes, :]
            for i in range(0, data.shape[0]):
                overall_x.append(data[i].reshape(data.shape[1] * data.shape[2]))
                overall_y.append(label[i])

        x = np.array(normalize(np.array(overall_x)))
        x = StandardScaler().fit_transform(x).reshape(x.shape[0], x.shape[1], 1)
        y = np.ravel(np.array(overall_y)[:, [0 if classify_type == ClassifyType.Arousal else 1]])
        y *= 7 / y.max()
        y = tf.keras.utils.to_categorical(y)
        training_dataset = train_test_split(x, y, test_size=0.25, stratify=y)
        accuracy = training(dataset=training_dataset, classify_type=classify_type, batch_size=256, epochs=200,
                            natural_optimizer=FsType[row["optimizer"]], full_model=True)

        best_channel_list.at[index, "accuracy"] = round(accuracy * 100, 2)
        print("test")
    best_channel_list.to_csv(Path(ROOT_PATH, "final_channel_list.csv"), index=False)


def final_channel_selection_validation():
    participant_list: [int] = range(1, 33)

    best_channels = ['P7', 'FC5', 'T7', 'T8', 'F4', 'O1', 'AF3']

    overall_x = []
    overall_y = []
    epoc_indexes = [EPOC_ELECTRODES.index(channel) for channel in best_channels]
    for participant in participant_list:
        sub = np.load(str(Path(PREPROCESSED_DATA_PATH, f"Participant_{participant}.npy")), allow_pickle=True)
        data = []
        label = []
        for i in range(0, sub.shape[0]):
            data.append(np.array(sub[i][0]))
            label.append(np.array(sub[i][1]))
        data = np.array(data)
        label = np.array(label)
        data = data[:, epoc_indexes, :]
        for i in range(0, data.shape[0]):
            overall_x.append(data[i].reshape(data.shape[1] * data.shape[2]))
            overall_y.append(label[i])

    for classify_type in ClassifyType:
        x = np.array(normalize(np.array(overall_x)))
        x = StandardScaler().fit_transform(x).reshape(x.shape[0], x.shape[1], 1)
        y = np.ravel(np.array(overall_y)[:, [0 if classify_type == ClassifyType.Arousal else 1]])
        y *= 7 / y.max()
        y = tf.keras.utils.to_categorical(y)
        training_dataset = train_test_split(x, y, test_size=0.25, stratify=y)
        accuracy, history = training(dataset=training_dataset, classify_type=classify_type, batch_size=256, epochs=200,
                                     full_model=True)

        print(f"Accuracy for {classify_type.value} is {accuracy * 100}%")

        final = pd.DataFrame({"channels": [best_channels], "accuracy": accuracy})
        final.to_csv(Path(ROOT_PATH, f"best_final_channel_{classify_type.value}.csv"), index=False)

        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['val_loss'])
        plt.title(f"{classify_type.value}")
        plt.legend(["accuracy", "loss"])
        plt.ylabel('loss/accuracy')
        plt.xlabel('epoch')
        plt.savefig(Path(ROOT_PATH, f"Graph_{classify_type.value}.pdf"))
        plt.clf()


def train_final_model():
    model_name = "CNN_LSTM_BIG"
    overall_data, labels_one_hot = generate_ten_sec_data()
    # shuffle data
    np.random.seed(42)
    indices = np.arange(overall_data.shape[0])
    np.random.shuffle(indices)

    overall_data = overall_data[indices, :, :]
    labels_one_hot = labels_one_hot[indices]

    # split data into train, test, validation
    index_60 = int(overall_data.shape[0] * 0.6)
    index_80 = int(overall_data.shape[0] * 0.8)

    x_train = overall_data[:index_60, :]
    x_test = overall_data[index_60:index_80, :]
    x_val = overall_data[index_80:, :]

    y_train = labels_one_hot[:index_60]
    y_test = labels_one_hot[index_60:index_80]
    y_val = labels_one_hot[index_80:]

    model = build_model_LSTM_CNN((x_train.shape[1], 1))
    print("Training model with 60% of data {} and validating with 20% of data {}.".format(x_train.shape, x_test.shape))
    history, model = train_model(model, x_train, x_test, y_train, y_test, 256, 165)
    tf.keras.models.save_model(model, Path(ROOT_PATH, f"Real_Time_Model_{model_name}"))

    start_time = datetime.now()
    score = model.evaluate(x_val, y_val, verbose=1)
    end_time = datetime.now()
    elapsed = end_time - start_time
    print("Prediction took: {}\n".format(humanize.precisedelta(elapsed, minimum_unit="seconds")))
    print("With {} test shape! And {:.2%} Accuracy and {} loss.".format(x_val.shape, score[1], score[0]))

    plot_train_history(history, model_name, score[1], score[0])

    y_pred = model.predict(x_val)
    plot_confusion_matrix(y_pred, y_val, model_name)


if __name__ == "__main__":
    train_final_model()
