import math
import torch


class Meter:

    def __init__(self):
        self.list = []

    def update(self, item):
        self.list.append(item)

    def avg(self):
        return torch.tensor(self.list).mean() if len(self.list) else None

    def confidence_interval(self):
        if len(self.list) == 0:
            return None
        std = torch.tensor(self.list).std()
        ci = std * 1.96 / math.sqrt(len(self.list))
        return ci

    def avg_and_confidence_interval(self):
        return self.avg(), self.confidence_interval()
