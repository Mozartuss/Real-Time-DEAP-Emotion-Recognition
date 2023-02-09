from enum import Enum
from pathlib import Path

from Utils.Helper import get_project_root

DEAP_ELECTRODES = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz",
                   "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
                   "PO4", "O2"]

EPOC_ELECTRODES = ["AF3", "F3", "F7", "FC5", "T7", "P7", "O1", "AF4", "F4", "F8", "FC6", "T8", "P8", "O2"]

FREQUENCIES = ["Theta", "Alpha", "LowerBeta", "UpperBeta", "Gamma"]


class FsType(Enum):
    MRMR = "MRMR"
    PCA = "PCA"
    PSO = "PSO"
    BAT = "BAT"
    CS = "CS"
    TMGWO = "TMGWO"
    GWO = "GWO"
    ISSA = "ISSA"
    SSA = "SSA"


class ClassifyType(Enum):
    Arousal = "Arousal"
    Valence = "Valence"


ROOT_PATH = get_project_root()

RAW_DATA_PATH = Path(ROOT_PATH, "Data", "RAW_DEAP_DATASET")
PREPROCESSED_DATA_PATH = Path(ROOT_PATH, "Data", "PREPROCESSED_DEAP_DATA")


def final_dataset_path(fs: str = ""):
    if fs == "":
        path = Path(ROOT_PATH, "Data", "FINAL_DEAP_DATASET")
    else:
        path = Path(ROOT_PATH, "Data", "FINAL_DEAP_DATASET_{}".format(fs))
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_channel_selection_path(fs: str = ""):
    path = Path(ROOT_PATH, "Data", "SAVED_{}_CHANNELS".format(fs))
    path.mkdir(parents=True, exist_ok=True)
    return path
