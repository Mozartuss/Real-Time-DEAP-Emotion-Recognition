import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from Utils.Constants import FsType, ClassifyType, final_dataset_path, save_channel_selection_path, \
    PREPROCESSED_DATA_PATH, EPOC_ELECTRODES

from FeatureSelection.SwarmHelper import bat_optimizer, gwo_optimizer, tmgwo_optimizer, ssa_optimizer, \
    issa_optimizer, cs_optimizer, pso_optimizer


def use_swarm_based_fs(natural_optimizer: FsType, opts: dict, participant_list: [int], n_channel: int, n_freq: int,
                       classify_type: ClassifyType, components: int = 20, n_cores: int = multiprocessing.cpu_count()):
    tqdm.write(
        f"Run {natural_optimizer.value} channel selection method with {classify_type.value} and with {opts['N']} particles/solutions and with {opts['T']} max iterations on {n_cores} cores")

    backup = []

    dataset_path = final_dataset_path(natural_optimizer.value)

    pbar = tqdm(participant_list, leave=False, position=0)
    for participant in pbar:
        save_path_data = Path(dataset_path, f"data_participant_{participant}.npy")
        save_path_label = Path(dataset_path, f"label_participant_{participant}.npy")
        pbar.set_description(f"Participant {participant}/{len(participant_list)} select {components} channels")
        label, zx, p, ch = exec_feature_selection(participant=participant, opts=opts, components=components,
                                                  classify_type=classify_type, optimizer=natural_optimizer,
                                                  n_channel=n_channel, n_freq=n_freq)
        ch = list(ch)
        ch = [EPOC_ELECTRODES[i] for i in np.sort(ch)]
        backup.append([p, ch])
        data = zx.reshape(zx.shape[0], zx.shape[1] * zx.shape[2])
        np.save(str(save_path_data), np.array(data), allow_pickle=True, fix_imports=True)
        np.save(str(save_path_label), np.array(label), allow_pickle=True, fix_imports=True)

    pd.DataFrame(backup, columns=["participant", "channels"], index=None).sort_values("participant").to_csv(
        Path(save_channel_selection_path(natural_optimizer.value),
             f"{natural_optimizer.value}_channels_{classify_type.value}.csv"), index=False)


def exec_feature_selection(participant: [int], opts: dict, components: int, classify_type: ClassifyType,
                           optimizer: FsType, n_channel: int, n_freq: int):
    sub = np.load(str(Path(PREPROCESSED_DATA_PATH, f"Participant_{participant}.npy")), allow_pickle=True)
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
    # split data into train & validation (70 -- 30)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, stratify=y)

    scaler = StandardScaler()
    scaler2 = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler2.fit_transform(xtest)

    ytrain *= 7 / ytrain.max()
    ytrain = tf.keras.utils.to_categorical(ytrain)
    ytrain = np.asarray([ytrain_c.tolist().index(1) + 1 for ytrain_c in ytrain])

    ytest *= 7 / ytest.max()
    ytest = tf.keras.utils.to_categorical(ytest)
    ytest = np.asarray([ytest_c.tolist().index(1) + 1 for ytest_c in ytest])

    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

    opts["fold"] = fold
    pbar = tqdm(range(opts["T"]), leave=False, desc=optimizer.value, position=1)
    if optimizer in FsType:
        selected_channels = \
            getattr(sys.modules[__name__], "%s_optimizer" % optimizer.value.lower())(x, opts, components, pbar)
    else:
        raise Exception("Optimizer not found")
    pbar.close()

    x = pd.DataFrame(x)
    data_new = x[selected_channels].to_numpy()
    z = []
    for i in data_new.transpose(1, 0):
        z.append(i.reshape(-1, n_freq))
    zx = np.array(z).transpose((1, 0, 2))
    return label, zx, participant, selected_channels
