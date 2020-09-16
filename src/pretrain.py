from models import get_model
from loss import get_loss_fn
from utils import get_optimizer
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from collections import defaultdict
from data import get_criteo_dataset
from utils import ScalarMovingAverage

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(
    physical_devices[0], enable=True
)
if len(physical_devices) > 1:
    tf.config.experimental.set_memory_growth(
        physical_devices[1], enable=True
    )


def optim_step(model, x, targets, optimizer, loss_fn, params):
    with tf.GradientTape() as g:
        outputs = model(x, training=True)
        reg_loss = tf.add_n(model.losses)
        loss_dict = loss_fn(targets, outputs, params)
        loss = loss_dict["loss"] + reg_loss

    trainable_variables = model.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))


def test(model, test_data, params):
    all_logits = []
    all_probs = []
    all_labels = []
    for step, (batch_x, batch_y) in enumerate(tqdm(test_data), 1):
        logits = model(batch_x, training=False)["logits"]
        all_logits.append(logits.numpy())
        all_labels.append(batch_y.numpy())
        all_probs.append(tf.math.sigmoid(logits))
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 1))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, 1))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 1))
    llloss = cal_llloss_with_logits(all_labels, all_logits)
    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)
    batch_size = all_logits.shape[0]
    return auc


def train(model, optimizer, train_data, params):
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y = batch[1]
        targets = {"label": batch_y}
        optim_step(model, batch_x, targets, optimizer,
                   get_loss_fn(params["loss"]), params)


def run(params):
    dataset = get_criteo_dataset(params)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_data = tf.data.Dataset.from_tensor_slices(
        (dict(train_dataset["x"]), train_dataset["labels"]))
    train_data = train_data.batch(params["batch_size"]).prefetch(1)
    test_data = tf.data.Dataset.from_tensor_slices(
        (dict(test_dataset["x"]), test_dataset["labels"]))
    test_data = test_data.batch(params["batch_size"]).prefetch(1)
    model = get_model(params["model"], params)
    optimizer = get_optimizer(params["optimizer"], params)
    best_acc = 0
    for ep in range(params["epoch"]):
        train(model, optimizer, train_data, params)
        model.save_weights(params["model_ckpt_path"], save_format="tf")