from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from Utils.Constants import final_dataset_path, PREPROCESSED_DATA_PATH


def use_pca(participant_list: [int], n_channel: int, n_freq: int, components: int):
    tqdm.write(f"Run PCA channel selection method")

    final_dataset_path_pca = final_dataset_path("PCA")

    for participant in participant_list:
        save_path_data = Path.joinpath(final_dataset_path_pca, f"data_participant_{participant}.npy")
        save_path_label = Path.joinpath(final_dataset_path_pca, f"label_participant_{participant}.npy")

        sub = np.load(str(Path(PREPROCESSED_DATA_PATH, f"Participant_{participant}.npy")), allow_pickle=True)
        data = []
        label = []
        for i in range(0, sub.shape[0]):
            data.append(np.array(sub[i][0]))
            label.append(np.array(sub[i][1]))
        data = np.array(data)
        label = np.array(label)

        x = data.transpose((1, 0, 2)).reshape(n_channel, -1).transpose((1, 0))

        standard_scaler = StandardScaler()
        x = standard_scaler.fit_transform(x)

        x = pd.DataFrame(x)

        pca = PCA(n_components=components)
        x_pca = pca.fit_transform(x)

        z = []
        for i in x_pca.transpose(1, 0):
            z.append(i.reshape(-1, n_freq))

        zx = np.array(z).transpose((1, 0, 2))

        final_data = zx.reshape(zx.shape[0], zx.shape[1] * zx.shape[2])
        np.save(str(save_path_data), np.array(final_data), allow_pickle=True, fix_imports=True)
        np.save(str(save_path_label), np.array(label), allow_pickle=True, fix_imports=True)
