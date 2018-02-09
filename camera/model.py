from .scheduler import ReduceLROnPlateau

from sklearn.metrics import accuracy_score, log_loss
from torch.autograd import Variable
from torchvision.models import resnet50


import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

NUM_CLASSES = 10


class SerializableModule(nn.Module):
    def __init__(self):
        super(SerializableModule, self).__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))


class Model(SerializableModule):
    def __init__(self, weights_path=None):
        super(Model, self).__init__()

        model = resnet50()
        if weights_path is not None:
            state_dict = torch.load(weights_path)
            model.load_state_dict(state_dict)

        num_features = model.fc.in_features
        model.fc = nn.Dropout(0.0)
        model.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

        self._model = model

    def forward(self, x):
        x = self._model(x)
        x = self.fc(x)
        return x


class CameraModel(object):
    def __init__(self, model=None):
        if model is None:
            model = Model()
        self._torch_single_model = model
        self._torch_model = nn.DataParallel(self._torch_single_model).cuda()

        self._optimizer = torch.optim.Adam(self._torch_model.parameters(), lr=0.0001)
        self._scheduler = ReduceLROnPlateau(self._optimizer, factor=0.5, patience=5,
                                            min_lr=1e-6, epsilon=1e-5, verbose=1, mode='min')
        self._optimizer.zero_grad()
        self._criterion = nn.CrossEntropyLoss()

    def scheduler_step(self, loss, epoch):
        self._scheduler.step(loss, epoch)

    def enable_train_mode(self):
        self._torch_model.train()

    def enable_predict_mode(self):
        self._torch_model.eval()

    def train_on_batch(self, X, y):
        X = X.cuda(async=True)
        y = y.cuda(async=True)
        X = Variable(X, requires_grad=False)
        y = Variable(y, requires_grad=False)

        y_pred = self._torch_model(X)

        loss = self._criterion(y_pred, y)
        loss.backward()

        self._optimizer.step()
        self._optimizer.zero_grad()

        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred.cpu().data.numpy()

    def predict_on_batch(self, X):
        X = X.cuda(async=True)
        X = Variable(X, requires_grad=False, volatile=True)
        y_pred = self._torch_model(X)
        y_pred = F.softmax(y_pred, dim=-1)
        return y_pred.cpu().data.numpy()

    def fit_generator(self, generator):
        self.enable_train_mode()
        mean_loss = None
        mean_accuracy = None
        start_time = time.time()
        for step_no, (X, y) in enumerate(generator):
            y_pred = self.train_on_batch(X, y)
            y = y.cpu().numpy()

            accuracy = accuracy_score(y, y_pred.argmax(axis=-1))
            loss = log_loss(y, y_pred, eps=1e-6, labels=list(range(10)))

            if mean_loss is None:
                mean_loss = loss

            if mean_accuracy is None:
                mean_accuracy = accuracy

            mean_loss = 0.9 * mean_loss + 0.1 * loss
            mean_accuracy = 0.9 * mean_accuracy + 0.1 * accuracy

            cur_time = time.time() - start_time
            print("[{3} s] Train step {0}. Loss {1}. Accuracy {2}".format(step_no, mean_loss, mean_accuracy, cur_time))

    def predict_generator(self, generator):
        self.enable_predict_mode()
        result = []
        start_time = time.time()
        for step_no, X in enumerate(generator):
            if isinstance(X, (tuple, list)):
                X = X[0]

            y_pred = self.predict_on_batch(X)
            result.append(y_pred)
            print("[{1} s] Predict step {0}".format(step_no, time.time() - start_time))

        return np.concatenate(result)

    def save(self, filename):
        self._torch_single_model.save(filename)

    @staticmethod
    def load(filename):
        model = Model()
        model.load(filename)
        return CameraModel(model=model)
