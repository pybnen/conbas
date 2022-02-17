from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import numpy as np


class Logger(object):

    def __init__(self):
        maxlen = 100

        # self.writer = SummaryWriter()
        self.reset()
        # self.batch_size = []
        # self.loss = []
        # self.batch_loss = 0.0

        self.recent_batch_losses = deque(maxlen=maxlen)
        self.recent_batch_accuracies = deque(maxlen=maxlen)

    # TODO its not really batch but epoch :(
    def reset(self):
        self.batch_sizes = []
        
        self.losses = []
        self.batch_loss = 0.
        self.mv_avg_batch_loss = 0.

        self.accuracies = []
        self.batch_accuracy = 0.
        self.mv_avg_batch_accuracy = 0.

    def end_batch(self):
        np_batch_size = np.array(self.batch_sizes)
        self.batch_loss = np.sum(np.array(self.losses) * np_batch_size) / np_batch_size.sum()
        self.batch_accuracy = np.sum(np.array(self.accuracies) * np_batch_size) / np_batch_size.sum()

        self.recent_batch_losses.append(self.batch_loss)
        self.recent_batch_accuracies.append(self.batch_accuracy)

        self.mv_avg_batch_loss = np.mean(self.recent_batch_losses)
        self.mv_avg_batch_accuracy = np.mean(self.recent_batch_accuracies)

    def to_str(self):
        s = f"loss {self.batch_loss:.4f} / {self.mv_avg_batch_loss:.4f} "
        s += f"accuracy {self.batch_accuracy:.4f} / {self.mv_avg_batch_accuracy:.4f}"
        return s

    def add_step(self, loss: float, accuracy: float, batch_size: int):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.batch_sizes.append(batch_size)
