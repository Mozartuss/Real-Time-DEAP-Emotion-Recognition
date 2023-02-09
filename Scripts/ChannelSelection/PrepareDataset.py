from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler

from Utils.Constants import FsType, ClassifyType, final_dataset_path


def prepare_dataset(classify_type: ClassifyType, natural_optimizer: FsType, participant: int, test_size: float = 0.25):
    """
    Prepare dataset for training and testing
    @param classify_type: Type of classification [enum ClassifyType]
    @param natural_optimizer: Type of channel selection algorithm [enum FsType]
    @param test_size: Size of test set [float]
    @return: x_train, x_test, y_train, y_test
    @rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """
    data_path = final_dataset_path(natural_optimizer.value)

    x = np.load(str(Path(data_path, f"data_participant_{participant}.npy")))
    y = np.load(str(Path(data_path, f"label_participant_{participant}.npy")))

    x = np.array(normalize(x))
    x = StandardScaler().fit_transform(x).reshape(x.shape[0], x.shape[1], 1)
    y = np.ravel(y[:, [0 if classify_type == ClassifyType.Arousal else 1]])
    y *= 7 / y.max()
    y = tf.keras.utils.to_categorical(y)

    return train_test_split(x, y, test_size=test_size, stratify=y)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = prepare_dataset(ClassifyType.Arousal, FsType.MRMR, 1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
