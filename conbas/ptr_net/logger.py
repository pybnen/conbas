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
        self.recent_epoch_f1 = deque(maxlen=maxlen)
        self.recent_epoch_set_accuracies = deque(maxlen=maxlen)
        self.recent_epoch_precisions = deque(maxlen=maxlen)
        self.recent_epoch_recalls = deque(maxlen=maxlen)

    def reset(self):
        self.batch_sizes = []

        self.losses = []
        self.epoch_loss = 0.
        self.mv_avg_epoch_loss = 0.

        self.accuracies = []
        self.epoch_accuracy = 0.
        self.mv_avg_epoch_accuracy = 0.

        self.f1s = []
        self.epoch_f1 = 0.
        self.mv_avg_epoch_f1 = 0.

        self.set_accuracies = []
        self.epoch_set_accuracy = 0.
        self.mv_avg_epoch_set_accuracy = 0.

        self.precisions = []
        self.epoch_precision = 0.
        self.mv_avg_epoch_precision = 0.

        self.recalls = []
        self.epoch_recall = 0.
        self.mv_avg_epoch_recall = 0.

    def end_epoch(self):
        np_batch_size = np.array(self.batch_sizes)
        
        self.epoch_loss = np.sum(np.array(self.losses) * np_batch_size) / np_batch_size.sum()
        self.epoch_accuracy = np.sum(np.array(self.accuracies) * np_batch_size) / np_batch_size.sum()
        self.epoch_f1 = np.sum(np.array(self.f1s) * np_batch_size) / np_batch_size.sum()
        self.epoch_set_accuracy = np.sum(np.array(self.set_accuracies) * np_batch_size) / np_batch_size.sum()
        self.epoch_precision = np.sum(np.array(self.precisions) * np_batch_size) / np_batch_size.sum()
        self.epoch_recall = np.sum(np.array(self.recalls) * np_batch_size) / np_batch_size.sum()

        self.recent_epoch_losses.append(self.epoch_loss)
        self.recent_epoch_accuracies.append(self.epoch_accuracy)
        self.recent_epoch_f1.append(self.epoch_f1)
        self.recent_epoch_set_accuracies.append(self.epoch_set_accuracy)
        self.recent_epoch_precisions.append(self.epoch_precision)
        self.recent_epoch_recalls.append(self.epoch_recall)

        self.mv_avg_epoch_loss = np.mean(self.recent_epoch_losses)
        self.mv_avg_epoch_accuracy = np.mean(self.recent_epoch_accuracies)
        self.mv_avg_epoch_f1 = np.mean(self.recent_epoch_f1)
        self.mv_avg_epoch_set_accuracy = np.mean(self.recent_epoch_set_accuracies)
        self.mv_avg_epoch_precision = np.mean(self.recent_epoch_precisions)
        self.mv_avg_epoch_recall = np.mean(self.recent_epoch_recalls)

    def to_str(self):
        s = f"loss {self.epoch_loss:.4f}/{self.mv_avg_epoch_loss:.4f}"
        s += f", accuracy {self.epoch_accuracy:.4f} / {self.mv_avg_epoch_accuracy:.4f}"
        s += f", f1 {self.epoch_f1:.4f} / {self.mv_avg_epoch_f1:.4f}"
        s += f", set accuracy {self.epoch_set_accuracy:.4f} / {self.mv_avg_epoch_set_accuracy:.4f}"
        s += f", precision {self.epoch_precision:.4f} / {self.mv_avg_epoch_precision:.4f}"
        s += f", recall {self.epoch_recall:.4f} / {self.mv_avg_epoch_recall:.4f}"
        return s

    def log(self, label, step):
        self.writer.add_scalar(f"{label}_avg/loss", self.mv_avg_epoch_loss, step)
        self.writer.add_scalar(f"{label}_avg/accuracy", self.mv_avg_epoch_accuracy, step)
        self.writer.add_scalar(f"{label}_avg/f1", self.mv_avg_epoch_f1, step)
        self.writer.add_scalar(f"{label}_avg/set_accuracy", self.mv_avg_epoch_set_accuracy, step)
        self.writer.add_scalar(f"{label}_avg/precision", self.mv_avg_epoch_precision, step)
        self.writer.add_scalar(f"{label}_avg/recall", self.mv_avg_epoch_recall, step)

        self.writer.add_scalar(f"{label}_epoch/loss", self.epoch_loss, step)
        self.writer.add_scalar(f"{label}_epoch/accuracy", self.epoch_accuracy, step)
        self.writer.add_scalar(f"{label}_epoch/f1", self.epoch_f1, step)
        self.writer.add_scalar(f"{label}_epoch/set_accuracy", self.epoch_set_accuracy, step)
        self.writer.add_scalar(f"{label}_epoch/precision", self.epoch_precision, step)
        self.writer.add_scalar(f"{label}_epoch/recall", self.epoch_recall, step)

    def add_step(self,
                 loss: float,
                 accuracy: float,
                 f1: float,
                 set_accuracy: float,
                 precision: float,
                 recall: float,
                 batch_size: int):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.f1s.append(f1)
        self.set_accuracies.append(set_accuracy)
        self.precisions.append(precision)
        self.recalls.append(recall)
        
        self.batch_sizes.append(batch_size)
