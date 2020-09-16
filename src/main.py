import argparse
import os
import pathlib
from copy import deepcopy

import tensorflow as tf
import numpy as np

from pretrain import run
from stream_train_test import stream_run


def run_params(args):
    params = deepcopy(vars(args))
    params["model"] = "MLP_SIG"
    params["optimizer"] = "Adam"
    if args.data_cache_path != "None":
        pathlib.Path(args.data_cache_path).mkdir(parents=True, exist_ok=True)
    if args.mode == "pretrain":
        if args.method == "Pretrain":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "baseline_prtrain"
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "dfm_prtrain"
            params["model"] = "MLP_EXP_DELAY"
        elif args.method == "FSIW":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = args.fsiw_pretraining_type+"_cd_"+str(args.CD)
            params["model"] = "MLP_FSIW"
        elif args.method == "ES-DFM":
            params["loss"] = "tn_dp_pretraining_loss"
            params["dataset"] = "tn_dp_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_tn_dp"
        else:
            raise ValueError(
                "{} method do not need pretraining other than Pretrain".format(args.method))
    else:
        if args.method == "Pretrain":
            params["loss"] = "none_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "Oracle":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "DFM":
            params["loss"] = "delayed_feedback_loss"
            params["dataset"] = "last_30_train_test_dfm"
        elif args.method == "FSIW":
            params["loss"] = "fsiw_loss"
            params["dataset"] = "last_30_train_test_fsiw"
        elif args.method == "ES-DFM":
            params["loss"] = "esdfm_loss"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(args.C)
        elif args.method == "Vanilla":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + \
                str(1)
        elif args.method == "FNW":
            params["loss"] = "fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNC":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw"
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="delayed feedback method",
                        choices=["FSIW",
                                 "DFM",
                                 "ES-DFM",
                                 "FNW",
                                 "FNC",
                                 "Pretrain",
                                 "Oracle",
                                 "Vanilla"],
                        type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["pretrain", "stream"], help="training mode", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--CD", type=int, default=7,
                        help="counterfactual deadline in FSIW")
    parser.add_argument("--C", type=int, default=0.25,
                        help="elapsed time in ES-DFM")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_path", type=str, required=True,
                        help="path of the data.txt in criteo dataset, e.g. /home/xxx/data.txt")
    parser.add_argument("--data_cache_path", type=str, default="None")
    parser.add_argument("--model_ckpt_path", type=str,
                        help="path to save pretrained model")
    parser.add_argument("--pretrain_fsiw0_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw0 model,  \
                        necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_fsiw1_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw1 model,  \
                        necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_baseline_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained baseline model(Pretrain),  \
                        necessary for the streaming evaluation of \
                            FSIW, ES-DFM, FNW, FNC, Pretrain, Oracle, Vanilla method")
    parser.add_argument("--pretrain_dfm_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained DFM model,  \
                        necessary for the streaming evaluation of \
                            DFM method")
    parser.add_argument("--pretrain_esdfm_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained ES-DFM model,  \
                        necessary for the streaming evaluation of \
                        ES-DFM method")
    parser.add_argument("--fsiw_pretraining_type", choices=["fsiw0", "fsiw1"], type=str, default="None",
                        help="FSIW needs two pretrained weighting model")
    parser.add_argument("--batch_size", type=int,
                        default=1024)
    parser.add_argument("--epoch", type=int, default=5,
                        help="training epoch of pretraining")
    parser.add_argument("--l2_reg", type=float, default=1e-6,
                        help="l2 regularizer strength")

    args = parser.parse_args()
    params = run_params(args)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    print("params {}".format(params))
    if args.mode == "pretrain":
        run(params)
    else:
        stream_run(params)
