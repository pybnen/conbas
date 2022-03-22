from dataclasses import dataclass
from collections import deque
import numpy as np


class Logger(object):

    def __init__(self, writer):
        maxlen = 10

        self.writer = writer
        self.reset()

        self.recent_epoch_losses = deque(maxlen=maxlen)
        self.recent_epoch_accuracies = deque(maxlen=maxlen)

    def reset(self):
        self.batch_sizes = []

        self.losses = []
        self.epoch_loss = 0.
        self.mv_avg_epoch_loss = 0.

        self.accuracies = []
        self.epoch_accuracy = 0.
        self.mv_avg_epoch_accuracy = 0.

    def end_epoch(self):
        np_batch_size = np.array(self.batch_sizes)
        self.epoch_loss = np.sum(np.array(self.losses) * np_batch_size) / np_batch_size.sum()
        self.epoch_accuracy = np.sum(np.array(self.accuracies) * np_batch_size) / np_batch_size.sum()

        self.recent_epoch_losses.append(self.epoch_loss)
        self.recent_epoch_accuracies.append(self.epoch_accuracy)

        self.mv_avg_epoch_loss = np.mean(self.recent_epoch_losses)
        self.mv_avg_epoch_accuracy = np.mean(self.recent_epoch_accuracies)

    def to_str(self):
        s = f"loss {self.epoch_loss:.4f}/{self.mv_avg_epoch_loss:.4f}"
        s += f", accuracy {self.epoch_accuracy:.4f} / {self.mv_avg_epoch_accuracy:.4f}"
        return s

    def log(self, label, step):
        self.writer.add_scalar(f"{label}/avg_loss", self.mv_avg_epoch_loss, step)
        self.writer.add_scalar(f"{label}/epoch_loss", self.epoch_loss, step)

        self.writer.add_scalar(f"{label}/avg_accuracy", self.mv_avg_epoch_accuracy, step)
        self.writer.add_scalar(f"{label}/epoch_accuracy", self.epoch_accuracy, step)

    def add_step(self, loss: float, accuracy: float, batch_size: int):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.batch_sizes.append(batch_size)
