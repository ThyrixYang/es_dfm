import copy
import os

import h5py
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import parse_float_arg

SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
# the input of neural network should be normalized


def get_data_df(params):
    df = pd.read_csv(params["data_path"], sep="\t", header=None)
    click_ts = df[df.columns[0]].to_numpy()
    pay_ts = df[df.columns[1]].fillna(-1).to_numpy()

    df = df[df.columns[2:]]
    for c in df.columns[8:]:
        df[c] = df[c].fillna("")
        df[c] = df[c].astype(str)
    for c in df.columns[:8]:
        df[c] = df[c].fillna(-1)
        df[c] = (df[c] - df[c].min())/(df[c].max() - df[c].min())
    df.columns = [str(i) for i in range(17)]
    df.reset_index(inplace=True)
    return df, click_ts, pay_ts


class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, sample_ts=None, labels=None, delay_label=None):
        self.x = features.copy(deep=True)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        self.delay_label = delay_label
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if labels is not None:
            self.labels = copy.deepcopy(labels)
        else:
            self.labels = (pay_ts > 0).astype(np.int32)

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def sub_hours(self, start_hour, end_hour):
        start_ts = start_hour*SECONDS_AN_HOUR
        end_ts = end_hour*SECONDS_AN_HOUR
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x.iloc[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask])

    def add_fake_neg(self):
        pos_mask = self.pay_ts > 0
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[pos_mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[pos_mask]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[pos_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones((np.sum(pos_mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def only_pos(self):
        pos_mask = self.pay_ts > 0
        print(np.mean(pos_mask))
        print(self.pay_ts[pos_mask].shape)
        return DataDF(self.x.iloc[pos_mask],
                      self.click_ts[pos_mask],
                      self.pay_ts[pos_mask],
                      self.sample_ts[pos_mask],
                      self.labels[pos_mask])

    def to_tn(self):
        mask = np.logical_or(self.pay_ts < 0, self.pay_ts -
                             self.click_ts > SECONDS_AN_HOUR)
        x = self.x.iloc[mask]
        sample_ts = self.sample_ts[mask]
        click_ts = self.click_ts[mask]
        pay_ts = self.pay_ts[mask]
        label = pay_ts < 0
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_dp(self):
        x = self.x
        sample_ts = self.sample_ts
        click_ts = self.click_ts
        pay_ts = self.pay_ts
        label = pay_ts - click_ts > SECONDS_AN_HOUR
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def add_esdfm_cut_fake_neg(self, cut_size):
        mask = self.pay_ts - self.click_ts > cut_size
        x = pd.concat(
            (self.x.copy(deep=True), self.x.iloc[mask].copy(deep=True)))
        sample_ts = np.concatenate(
            [self.click_ts+cut_size, self.pay_ts[mask]], axis=0)  # negative samples can only be used after cut_size hours
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[mask] = 0  # fake negatives
        # insert delayed positives
        labels = np.concatenate([labels, np.ones((np.sum(mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x.iloc[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def to_fsiw_1(self, cd, T):  # build pre-training dataset 1 of FSIW
        mask = np.logical_and(self.click_ts < T-cd, self.pay_ts > 0)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.click_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < T - cd] = 1
        # FSIW needs elapsed time information
        x.insert(x.shape[1], column="elapse", value=(
            T-click_ts-cd)/SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_0(self, cd, T):  # build pre-training dataset 0 of FSIW
        mask = np.logical_or(self.pay_ts >= T-cd, self.pay_ts < 0)
        mask = np.logical_and(self.click_ts < T-cd, mask)
        x = self.x.iloc[mask].copy(deep=True)
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.sample_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < 0] = 1
        x.insert(x.shape[1], column="elapse", value=(
            T-click_ts-cd)/SECONDS_FSIW_NORM)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        self.x.insert(self.x.shape[1], column="elapse", value=(
            cut_ts - self.click_ts)/SECONDS_FSIW_NORM)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      label)

    def to_dfm_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        delay = np.reshape(cut_ts - self.click_ts, (-1, 1))/SECONDS_DELAY_NORM
        labels = np.concatenate([np.reshape(label, (-1, 1)), delay], axis=1)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      labels)

    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(self.x.iloc[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.sample_ts[idx],
                      self.labels[idx])


def get_criteo_dataset_stream(params):
    name = params["dataset"]
    print("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        print("cache_path {}".format(cache_path))
        print("\nloading from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_stream = data["train"]
        test_stream = data["test"]
    else:
        print("\ncan't load from cache, building dataset")
        df, click_ts, pay_ts = get_data_df(params)
        if name == "last_30_train_test_oracle":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif name == "last_30_train_test_fsiw":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                cut_ts = (tr+1)*SECONDS_AN_HOUR
                train_hour = train_data.sub_hours(
                    tr, tr+1).to_fsiw_tune(cut_ts)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif name == "last_30_train_test_dfm":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(30, 60)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                cut_ts = (tr+1)*SECONDS_AN_HOUR
                train_hour = train_data.sub_hours(tr, tr+1).to_dfm_tune(cut_ts)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif "last_30_train_test_esdfm" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            print("cut_hour {}".format(cut_hour))
            cut_sec = cut_hour*SECONDS_AN_HOUR
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(0, 60).add_esdfm_cut_fake_neg(cut_sec)
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        elif name == "last_30_train_test_fnw":
            data = DataDF(df, click_ts, pay_ts)
            train_data = data.sub_days(0, 60).add_fake_neg()
            test_data = data.sub_days(30, 60)
            train_stream = []
            test_stream = []
            for tr in range(30*24, 59*24+23):
                train_hour = train_data.sub_hours(tr, tr+1)
                train_stream.append({"x": train_hour.x,
                                     "click_ts": train_hour.click_ts,
                                     "pay_ts": train_hour.pay_ts,
                                     "sample_ts": train_hour.sample_ts,
                                     "labels": train_hour.labels})
            for tr in range(30*24+1, 60*24):
                test_hour = test_data.sub_hours(tr, tr+1)
                test_stream.append({"x": test_hour.x,
                                    "click_ts": test_hour.click_ts,
                                    "pay_ts": test_hour.pay_ts,
                                    "sample_ts": test_hour.sample_ts,
                                    "labels": test_hour.labels})
        else:
            raise NotImplementedError("{} data does not exist".format(name))
        if params["data_cache_path"] != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_stream, "test": test_stream}, f)
    return train_stream, test_stream


def get_criteo_dataset(params):
    name = params["dataset"]
    print("loading datasest {}".format(name))
    cache_path = os.path.join(
        params["data_cache_path"], "{}.pkl".format(name))
    if params["data_cache_path"] != "None" and os.path.isfile(cache_path):
        print("cache_path {}".format(cache_path))
        print("\nloading from dataset cache")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        train_data = data["train"]
        test_data = data["test"]
    else:
        print("\nbuilding dataset")
        df, click_ts, pay_ts = get_data_df(params)
        data = DataDF(df, click_ts, pay_ts)
        if name == "baseline_prtrain":
            train_data = data.sub_days(0, 30).shuffle()
            mask = train_data.pay_ts < 0
            train_data.pay_ts[mask] = 30 * \
                SECONDS_A_DAY + train_data.click_ts[mask]
            test_data = data.sub_days(30, 60)
        elif name == "dfm_prtrain":
            train_data = data.sub_days(0, 30).shuffle()
            train_data.pay_ts[train_data.pay_ts < 0] = SECONDS_A_DAY*30
            delay = np.reshape(train_data.pay_ts -
                               train_data.click_ts, (-1, 1))/SECONDS_DELAY_NORM
            train_data.labels = np.reshape(train_data.labels, (-1, 1))
            train_data.labels = np.concatenate(
                [train_data.labels, delay], axis=1)
            test_data = data.sub_days(30, 60)
        elif "tn_dp_pretrain" in name:
            cut_hour = parse_float_arg(name, "cut_hour")
            cut_sec = int(SECONDS_AN_HOUR*cut_hour)
            train_data = data.sub_days(0, 30).shuffle()
            train_label_tn = np.reshape(train_data.pay_ts < 0, (-1, 1))
            train_label_dp = np.reshape(
                train_data.pay_ts - train_data.click_ts > cut_sec, (-1, 1))
            train_label = np.reshape(train_data.pay_ts > 0, (-1, 1))
            train_data.labels = np.concatenate(
                [train_label_tn, train_label_dp, train_label], axis=1)
            test_data = data.sub_days(30, 60)
            test_label_tn = np.reshape(test_data.pay_ts < 0, (-1, 1))
            test_label_dp = np.reshape(
                test_data.pay_ts - test_data.click_ts > cut_sec, (-1, 1))
            test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
            test_data.labels = np.concatenate(
                [test_label_tn, test_label_dp, test_label], axis=1)
        elif "fsiw1" in name:
            cd = parse_float_arg(name, "cd")
            print("cd {}".format(cd))
            train_data = data.sub_days(0, 30).shuffle()
            test_data = data.sub_days(30, 60)
            train_data = train_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=30*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_1(
                cd=cd*SECONDS_A_DAY, T=60*SECONDS_A_DAY)
        elif "fsiw0" in name:
            cd = parse_float_arg(name, "cd")
            train_data = data.sub_days(0, 30).shuffle()
            test_data = data.sub_days(30, 60)
            train_data = train_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=30*SECONDS_A_DAY)
            test_data = test_data.to_fsiw_0(
                cd=cd*SECONDS_A_DAY, T=60*SECONDS_A_DAY)
        else:
            raise NotImplementedError("{} dataset does not exist".format(name))
        if params["data_cache_path"] != "None":
            with open(cache_path, "wb") as f:
                pickle.dump({"train": train_data, "test": test_data}, f)
    return {
        "train": {
            "x": train_data.x,
            "click_ts": train_data.click_ts,
            "pay_ts": train_data.pay_ts,
            "sample_ts": train_data.sample_ts,
            "labels": train_data.labels,
        },
        "test": {
            "x": test_data.x,
            "click_ts": test_data.click_ts,
            "pay_ts": test_data.pay_ts,
            "sample_ts": test_data.sample_ts,
            "labels": test_data.labels,
        }
    }
