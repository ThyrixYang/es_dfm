from models import get_model
from loss import get_loss_fn
from utils import get_optimizer, ScalarMovingAverage
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc, cal_llloss_with_prob
from data import get_criteo_dataset_stream
from tqdm import tqdm
import numpy as np
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import layers, Model
import tensorflow as tf
from collections import defaultdict
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


def test(model, test_data, params):
    all_logits = []
    all_probs = []
    all_labels = []
    for step, (batch_x, batch_y) in enumerate(tqdm(test_data), 1):
        logits = model.predict(batch_x)
        all_logits.append(logits.numpy())
        all_labels.append(batch_y.numpy())
        all_probs.append(tf.sigmoid(logits))
    all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1,))
    all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1,))
    all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1,))
    if params["method"] == "FNC":
        all_probs = all_probs / (1-all_probs+1e-8)
        llloss = cal_llloss_with_prob(all_labels, all_probs)
    else:
        llloss = cal_llloss_with_logits(all_labels, all_logits)
    batch_size = all_logits.shape[0]
    pred = all_probs >= 0.5
    auc = cal_auc(all_labels, all_probs)
    prauc = cal_prauc(all_labels, all_probs)
    return auc, prauc, llloss


def train(models, optimizer, train_data, params):
    if params["loss"] == "none_loss":
        return
    loss_fn = get_loss_fn(params["loss"])
    for step, batch in enumerate(tqdm(train_data), 1):
        batch_x = batch[0]
        batch_y = batch[1]
        targets = {"label": batch_y}

        with tf.GradientTape() as g:
            outputs = models["model"](batch_x, training=True)
            if params["method"] == "FSIW":
                logits0 = models["fsiw0"](batch_x, training=False)["logits"]
                logits1 = models["fsiw1"](batch_x, training=False)["logits"]
                outputs = {
                    "logits": outputs["logits"],
                    "logits0": logits0,
                    "logits1": logits1
                }
            elif params["method"] == "ES-DFM":
                logitsx = models["esdfm"](batch_x, training=False)
                outputs = {
                    "logits": outputs["logits"],
                    "tn_logits": logitsx["tn_logits"],
                    "dp_logits": logitsx["dp_logits"]
                }
            reg_loss = tf.add_n(models["model"].losses)
            loss_dict = loss_fn(targets, outputs, params)
            loss = loss_dict["loss"] + reg_loss

        trainable_variables = models["model"].trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))


def stream_run(params):
    train_stream, test_stream = get_criteo_dataset_stream(params)
    if params["method"] == "DFM":
        model = get_model("MLP_EXP_DELAY", params)
        model.load_weights(params["pretrain_dfm_model_ckpt_path"])
    else:
        model = get_model("MLP_SIG", params)
        model.load_weights(params["pretrain_baseline_model_ckpt_path"])
    models = {"model": model}
    if params["method"] == "FSIW":
        fsiw0_model = get_model("MLP_FSIW", params)
        fsiw0_model.load_weights(params["pretrain_fsiw0_model_ckpt_path"])
        fsiw1_model = get_model("MLP_FSIW", params)
        fsiw1_model.load_weights(params["pretrain_fsiw1_model_ckpt_path"])
        models["fsiw0"] = fsiw0_model
        models["fsiw1"] = fsiw1_model
    elif params["method"] == "ES-DFM":
        esdfm_model = get_model("MLP_tn_dp", params)
        esdfm_model.load_weights(params["pretrain_esdfm_model_ckpt_path"])
        models["esdfm"] = esdfm_model
    elif params["method"] == "DFM":
        dfm_model = get_model("MLP_EXP_DELAY", params)
        dfm_model.load_weights(params["pretrain_dfm_model_ckpt_path"])
        models["model"] = dfm_model

    optimizer = get_optimizer(params["optimizer"], params)

    auc_ma = ScalarMovingAverage()
    nll_ma = ScalarMovingAverage()
    prauc_ma = ScalarMovingAverage()

    for ep, (train_dataset, test_dataset) in enumerate(zip(train_stream, test_stream)):
        train_data = tf.data.Dataset.from_tensor_slices(
            (dict(train_dataset["x"]), train_dataset["labels"]))
        train_data = train_data.batch(params["batch_size"]).prefetch(1)
        train(models, optimizer, train_data, params)

        test_batch_size = test_dataset["x"].shape[0]
        test_data = tf.data.Dataset.from_tensor_slices(
            (dict(test_dataset["x"]), test_dataset["labels"]))
        test_data = test_data.batch(params["batch_size"]).prefetch(1)
        auc, prauc, llloss = test(model, test_data, params)
        print("epoch {}, auc {}, prauc {}, llloss {}".format(
            ep, auc, prauc, llloss))
        auc_ma.add(auc*test_batch_size, test_batch_size)
        nll_ma.add(llloss*test_batch_size, test_batch_size)
        prauc_ma.add(prauc*test_batch_size, test_batch_size)
        print("epoch {}, auc_ma {}, prauc_ma {}, llloss_ma {}".format(
            ep, auc_ma.get(), prauc_ma.get(), nll_ma.get()))
