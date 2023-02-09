import os
import pickle
from os.path import exists, isdir
from pathlib import Path

from tqdm.auto import tqdm

from Utils.Helper import convert_path, natural_sort


class LoadData:

    def __init__(self, path):
        self.path = convert_path(path)
        self.__validate_path(path)

    @staticmethod
    def __validate_path(path: str) -> bool:
        try:
            if not isinstance(path, str) or not path:
                return False
            else:
                return exists(path)

        except TypeError:
            return False

    def yield_raw_data(self, num_files_start: int = 0, num_files_end: int = 32):
        if not isdir(self.path):
            os.mkdir(self.path)
        filenames = next(os.walk(self.path))[2]
        filenames = natural_sort(filenames)

        if len(filenames) >= 1:
            pbar = tqdm(filenames[num_files_start:num_files_end], position=0)
            for file in pbar:
                if file.endswith(".dat"):
                    pbar.set_description("Reading %s" % file)
                    with open(Path(self.path, file), 'rb') as f:
                        yield file, pickle.load(f, encoding="latin1")
        else:
            raise FileNotFoundError("No files found in %s" % self.path)
