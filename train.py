from camera.augmentation import Augmentator
from camera.dataset import CameraDataset
from camera.model import CameraModel, Model
from camera.train_utils import train

from torch.utils.data import DataLoader

import argparse

PROCESS_COUNT = 32


def main():
    parser = argparse.ArgumentParser("Camera Kaggle competition")
    parser.add_argument("--train_files", required=True)
    parser.add_argument("--val_files", required=True)
    parser.add_argument("--pretrained_weights_path", required=True)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--model_save_path", required=True)

    args = parser.parse_args()

    train_augmentator = Augmentator(in_train_mode=True)
    test_augmentator = Augmentator(in_train_mode=False)

    train_dataset = CameraDataset(args.train_files, train_augmentator)
    val_dataset = CameraDataset(args.val_files, test_augmentator, expand_dataset=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, num_workers=PROCESS_COUNT)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=PROCESS_COUNT)

    model = Model(args.pretrained_weights_path)
    camera_model = CameraModel(model=model)

    train(camera_model, train_loader, val_loader, val_dataset.get_labels(), args.model_save_path)


if __name__ == "__main__":
    main()