from pathlib import Path

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from tqdm.auto import tqdm

from Utils.Constants import final_dataset_path, PREPROCESSED_DATA_PATH, save_channel_selection_path, \
    ClassifyType, EPOC_ELECTRODES


def use_mrmr(participant_list: [int], n_channel: int, n_freq: int, components: int, classify_type: ClassifyType):
    tqdm.write(f"Run MRMR channel selection method with {classify_type.value} as classify type")

    final_dataset_path_mrmr = final_dataset_path("MRMR")

    backup = []
    for participant in participant_list:

        save_path_data = Path.joinpath(final_dataset_path_mrmr, "data_participant_{}.npy".format(participant))
        save_path_label = Path.joinpath(final_dataset_path_mrmr, "label_participant_{}.npy".format(participant))
        filename = f"Participant_{participant}.npy"
        sub = np.load(str(Path(PREPROCESSED_DATA_PATH, filename)), allow_pickle=True)
        data = []
        label = []
        for i in range(0, sub.shape[0]):
            data.append(np.array(sub[i][0]))
            label.append(np.array(sub[i][1]))
        data = np.array(data)
        label = np.array(label)

        x = data.transpose((1, 0, 2)).reshape(n_channel, -1).transpose((1, 0))
        if classify_type == ClassifyType.Arousal:
            y = np.repeat(label[:, 0], n_freq)
        else:
            y = np.repeat(label[:, 1], n_freq)

        x = pd.DataFrame(x)
        y = pd.Series(y)

        mrmr_x_idx = mrmr_classif(X=x, y=y, K=components)

        ch = [EPOC_ELECTRODES[i] for i in np.sort(mrmr_x_idx)]
        backup.append([participant, ch])

        data_new = x[mrmr_x_idx].to_numpy()

        z = []
        for i in data_new.transpose(1, 0):
            z.append(i.reshape(-1, n_freq))

        zx = np.array(z).transpose((1, 0, 2))

        final_data = zx.reshape(zx.shape[0], zx.shape[1] * zx.shape[2])
        np.save(str(save_path_data), np.array(final_data), allow_pickle=True, fix_imports=True)
        np.save(str(save_path_label), np.array(label), allow_pickle=True, fix_imports=True)

    pd.DataFrame(backup, columns=["participant", "channels"], index=None).to_csv(
        Path.joinpath(save_channel_selection_path("MRMR"), f"MRMR_channels_{classify_type.value}.csv"), index=False)


def use_mrmr_v2(participant_list: [int], n_channel: int, n_freq: int, components: int, classify_type: ClassifyType):
    tqdm.write(f"Run MRMR channel selection method with {classify_type.value} as classify type")

    final_dataset_path_mrmr = final_dataset_path("MRMR")

    save_path_data_training = Path.joinpath(final_dataset_path_mrmr, "data_training.npy")
    save_path_label_training = Path.joinpath(final_dataset_path_mrmr, "label_training.npy")
    save_path_data_testing = Path.joinpath(final_dataset_path_mrmr, "data_testing.npy")
    save_path_label_testing = Path.joinpath(final_dataset_path_mrmr, "label_testing.npy")

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    backup = []
    for participant in participant_list:
        filename = f"Participant_{participant}.npy"
        sub = np.load(str(Path(PREPROCESSED_DATA_PATH, filename)), allow_pickle=True)
        data = []
        label = []
        for i in range(0, sub.shape[0]):
            data.append(np.array(sub[i][0]))
            label.append(np.array(sub[i][1]))
        data = np.array(data)
        label = np.array(label)

        x = data.transpose((1, 0, 2)).reshape(n_channel, -1).transpose((1, 0))
        if classify_type == ClassifyType.Arousal:
            y = np.repeat(label[:, 0], n_freq)
        else:
            y = np.repeat(label[:, 1], n_freq)

        x = pd.DataFrame(x)
        y = pd.Series(y)

        mrmr_x_idx = mrmr_classif(X=x, y=y, K=components)

        backup.append([participant, mrmr_x_idx])

        data_new = x[mrmr_x_idx].to_numpy()

        z = []
        for i in data_new.transpose(1, 0):
            z.append(i.reshape(-1, n_freq))

        zx = np.array(z).transpose((1, 0, 2))

        for i in range(0, zx.shape[0]):
            x_test.append(zx[i].reshape(zx.shape[1] * zx.shape[2], ))
            y_test.append(label[i])
        print(f"Participant {participant} done")

    pd.DataFrame(backup, columns=["participant", "channels"], index=None).to_csv(
        Path.joinpath(save_channel_selection_path("MRMR"), f"mrmr_channels_{classify_type}.csv"), index=False)

    np.save(save_path_data_training, np.array(x_train), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_training, np.array(y_train), allow_pickle=True, fix_imports=True)

    np.save(save_path_data_testing, np.array(x_test), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_testing, np.array(y_test), allow_pickle=True, fix_imports=True)


if __name__ == '__main__':
    use_mrmr_v2(participant_list=range(1, 33), n_channel=14, n_freq=5, components=7,
                classify_type=ClassifyType.Arousal)
