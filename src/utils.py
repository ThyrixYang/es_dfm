import numpy as np
from tensorflow.keras.optimizers import SGD, Adam, Adagrad
import re


def get_optimizer(name, params):
    if name == "SGD":
        if "momentum" not in params:
            params["momentum"] = 0.0
        if "nesterov" not in params:
            params["nesterov"] = False
        return SGD(learning_rate=params["lr"],
                   momentum=params["momentum"],
                   nesterov=params["nesterov"])
    elif name == "Adam":
        return Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999)


def parse_float_arg(input, prefix):
    p = re.compile(prefix+"_[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    if m is None:
        return None
    input = m.group()
    p = re.compile("[+-]?([0-9]*[.])?[0-9]+")
    m = p.search(input)
    return float(m.group())


class ScalarMovingAverage:

    def __init__(self, eps=1e-8):
        self.sum = 0
        self.count = eps

    def add(self, value, count):
        self.sum += value
        self.count += count
        return self

    def get(self):
        return self.sum / self.count
