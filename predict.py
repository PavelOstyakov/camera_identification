from camera.augmentation import Augmentator
from camera.dataset import CameraDataset
from camera.model import CameraModel
from camera.postprocessing import generate_submit
from camera.train_utils import predict_test

from torch.utils.data import DataLoader

import argparse

PROCESS_COUNT = 32


def main():
    parser = argparse.ArgumentParser("Camera Kaggle competition PREDICT")
    parser.add_argument("--test_files", required=True)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--submit_path", required=True)

    args = parser.parse_args()

    test_augmentator = Augmentator(in_train_mode=False)

    test_dataset = CameraDataset(args.test_files, test_augmentator, expand_dataset=True)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             drop_last=False, num_workers=PROCESS_COUNT)

    camera_model = CameraModel.load(args.model_path)
    predicts = predict_test(camera_model, test_loader)

    test_files = open(args.test_files).read().split("\n")[:-1]
    generate_submit(predicts, test_files, args.submit_path)

if __name__ == "__main__":
    main()
