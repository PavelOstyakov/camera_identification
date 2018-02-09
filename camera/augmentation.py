from PIL import Image
from io import BytesIO
from torchvision import transforms

import cv2
import numpy as np
import random

CROP_SIZE = 224
MANIP_PROBABILITY = 0.5


class Augmentator(object):
    def __init__(self, in_train_mode=True):
        self._in_train_mode = in_train_mode

        self._manip_list = []

        self._manip_list.append(lambda img: Augmentator._jpg_manip(img, 70))
        self._manip_list.append(lambda img: Augmentator._jpg_manip(img, 90))

        self._manip_list.append(lambda img: Augmentator._gamma_manip(img, 0.8))
        self._manip_list.append(lambda img: Augmentator._gamma_manip(img, 1.2))

        self._manip_list.append(lambda img: Augmentator._bicubic_manip(img, 0.5))
        self._manip_list.append(lambda img: Augmentator._bicubic_manip(img, 0.8))
        self._manip_list.append(lambda img: Augmentator._bicubic_manip(img, 1.5))
        self._manip_list.append(lambda img: Augmentator._bicubic_manip(img, 2.0))

        self._transform = Augmentator._get_transforms()

    @staticmethod
    def _get_transforms():
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def _jpg_manip(image, quality):
        buffer = BytesIO()
        image = Image.fromarray(image)
        image.save(buffer, format='jpeg', quality=quality)
        buffer.seek(0)
        result = Image.open(buffer)
        result = np.array(result)
        return result

    @staticmethod
    def _gamma_manip(image, gamma):
        result = np.uint8(cv2.pow(image / 255., gamma) * 255.)
        return result

    @staticmethod
    def _bicubic_manip(image, scale):
        result = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return result

    def __call__(self, image, aug_type):
        if 0 <= aug_type <= 7:
            manip_func = self._manip_list[aug_type]
            image = manip_func(image)
        elif aug_type == 8:
            image = np.rot90(image)
        elif aug_type == 9:
            image = np.rot90(image, 2)
        elif aug_type == 10:
            image = np.rot90(image, -1)

        if self._in_train_mode and random.random() < MANIP_PROBABILITY:
            manip_func = random.choice(self._manip_list)
            image = manip_func(image)

        image = self._transform(image)
        return image
