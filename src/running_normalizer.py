import numpy as np

class RunningNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.count = 0

    def update(self, values):
        values = values.detach()
        self.mean = values.mean().item()
        self.std = values.std(unbiased=False).item()
        self.count += 1

    def normalize(self, values):
        if self.std == 0:
            return values - self.mean
        return (values - self.mean) / self.std
