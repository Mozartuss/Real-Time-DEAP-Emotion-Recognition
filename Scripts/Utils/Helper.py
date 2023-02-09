import concurrent.futures
import pathlib
import re
from os.path import join, splitext, sep
from typing import List

import numpy as np
from tqdm.auto import tqdm


def natural_sort(in_list: List[str]) -> List[str]:
    """
    Sort the given list in the way that humans expect. 1,10,2,3,30 -> 1,2,3,10,30
    @param in_list: list of strings which have to be sorted like human do
    @return: sorted list of strings
    """

    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(in_list, key=alphanum_key)


def convert_path(path: str) -> str:
    """
    Convert path as string to os specific path.
    @param path: path as string
    @return: os specific path
    """
    if isinstance(path, pathlib.PurePath):
        return path
    else:
        path_sep = "/" if "/" in path else "\\"
        new_path = join(*path.split(path_sep))
        if splitext(path)[1] is not None and not (
                path.endswith("/") or path.endswith("\\")):
            new_path = new_path + sep
        if path.startswith(path_sep):
            return sep + new_path
        else:
            return new_path


def get_project_root() -> pathlib.Path:
    """Returns project root folder."""
    return pathlib.Path(__file__).parent.parent.parent


def delete_leading_zero(num):
    """
    Delete leading zeros from string
    @param num: number as string
    @return: number as string without leading zeros
    """
    if not num.startswith("0"):
        return num
    else:
        return delete_leading_zero(num[1:])


def bin_power_optimized(x_vector, band, fs):
    """
    Calculate power in each frequency band using FFT
    @param x_vector: signal vector
    @param band: frequency band [4, 8, 12, 16, 25, 45] for Theta(4-8), Alpha(8-12), LowerBeta(12-16), UpperBeta(16-25), Gamma(25-45)
    @param fs: sampling frequency
    @return: power in each frequency band and power ratio in each frequency band
    """
    c = np.abs(np.fft.fft(x_vector))
    indices = np.floor(band / fs * len(x_vector)).astype(int)
    power = np.zeros(len(band) - 1)
    for freq_index in range(len(band) - 1):
        power[freq_index] = np.sum(c[indices[freq_index]:indices[freq_index + 1]])
    power_ratio = power / np.sum(power)
    return power, power_ratio
