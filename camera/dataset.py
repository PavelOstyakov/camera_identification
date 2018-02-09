from PIL import Image

import copy
import numpy as np
import os
import torch.utils.data as data

TTA_COUNT = 11


class CameraDataset(data.Dataset):
    def __init__(self, path, augmentator, expand_dataset=False):
        self._files = []
        self._labels = []
        with open(path) as f:
            for example in f.read().split("\n")[:-1]:
                if " " in example:
                    example = example.split(" ")
                    file_path = " ".join(example[:-1])
                    label = example[-1]
                    label = int(label)
                else:
                    file_path, label = example, example

                suffixes = ["JPG", "JPEG", "jpeg"]
                if not os.path.exists(file_path):
                    for suffix in suffixes:
                        pos = file_path.rfind(".")
                        file_path = file_path[:pos]
                        file_path += "." + suffix

                        if os.path.exists(file_path):
                            break

                self._files.append(file_path)
                self._labels.append(label)

        self._manip_labels = [-1] * len(self._files)

        if expand_dataset:
            files = self._files
            labels = self._labels

            self._files = []
            self._labels = []
            self._manip_labels = []
            for index in range(len(files)):
                self._files.extend([files[index]] * TTA_COUNT)
                self._labels.extend([labels[index]] * TTA_COUNT)
                self._manip_labels.extend(list(range(TTA_COUNT)))

        self._augmentator = augmentator

    def _preprocess(self, index):
        path = self._files[index]

        image = Image.open(path)
        image = np.array(image)

        return self._augmentator(image, self._manip_labels[index])

    def get_labels(self):
        return copy.deepcopy(self._labels)

    def __getitem__(self, index):
        image = self._preprocess(index)
        label = self._labels[index]
        return image, label

    def __len__(self):
        return len(self._files)
