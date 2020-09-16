import tensorflow as tf
from tensorflow.python.ops import array_ops, math_ops


def stable_log1pex(x):
    return -tf.minimum(x, 0) + tf.math.log(1+tf.math.exp(-tf.abs(x)))


def fake_negative_weighted_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    p_no_grad = tf.sigmoid(tf.stop_gradient(x))
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = tf.reduce_mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}


def cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    x = tf.reshape(x, (-1,))
    z = tf.cast(z, tf.float32)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=z, logits=x))
    return {"loss": loss}


def exp_delay_loss(targets, outputs, params=None):
    z = tf.reshape(tf.cast(targets["label"][:, 0], tf.float32), (-1, 1))
    x = outputs["logits"]
    lamb = tf.math.softplus(outputs["log_lamb"])
    log_lamb = tf.math.log(lamb)
    d = tf.reshape(tf.cast(targets["label"][:, 1], tf.float32), (-1, 1))
    e = d
    p = tf.nn.sigmoid(x)
    pos_loss = -(-stable_log1pex(x) + log_lamb - lamb*d)
    neg_loss = -tf.math.log(1 - p + p*tf.math.exp(-lamb*e))
    return {"loss": tf.reduce_mean(pos_loss*z + neg_loss*(1-z))}


def delay_tn_dp_loss(targets, outputs, params=None):
    tn = tf.cast(outputs["tn_logits"], tf.float32)
    dp = tf.cast(outputs["dp_logits"], tf.float32)
    z = tf.cast(targets["label"], tf.float32)
    tn_label = tf.reshape(z[:, 0], (-1, 1))
    dp_label = tf.reshape(z[:, 1], (-1, 1))
    pos_label = tf.reshape(z[:, 2], (-1, 1))
    tn_mask = (1-pos_label)+dp_label
    tn_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tn_label, logits=tn)*tn_mask)\
        / tf.reduce_sum(tn_mask)
    dp_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=dp_label, logits=dp))
    loss = tn_loss + dp_loss
    return {
        "loss": loss,
        "tn_loss": tn_loss,
        "dp_loss": dp_loss
    }


def fsiw_loss(targets, outputs, params=None):
    x = outputs["logits"]
    logits0 = tf.stop_gradient(tf.cast(outputs["logits0"], tf.float32))
    logits1 = tf.stop_gradient(tf.cast(outputs["logits1"], tf.float32))
    prob0 = tf.sigmoid(logits0)
    prob1 = tf.sigmoid(logits1)
    z = tf.reshape(tf.cast(targets["label"], tf.float32), (-1, 1))

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1/(prob1+1e-8)
    neg_weight = prob0

    clf_loss = tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {
        "loss": loss,
    }


def delay_tn_importance_weight_loss(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"]
    z = targets["label"]
    z = tf.reshape(tf.cast(z, tf.float32), (-1, 1))
    prob = tf.stop_gradient(tf.math.sigmoid(x))
    dist_prob = tf.math.sigmoid(tn_logits)
    dp_prob = tf.math.sigmoid(dp_logits)

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)

    pos_weight = 1+dp_prob
    neg_weight = (1+dp_prob)*dist_prob
    neg_weight = tf.stop_gradient(neg_weight)
    pos_weight = tf.stop_gradient(pos_weight)

    clf_loss = tf.reduce_mean(
        pos_loss*pos_weight*z + neg_loss*neg_weight*(1-z))
    loss = clf_loss
    return {"loss": loss,
            "clf_loss": clf_loss}


def get_loss_fn(name):
    if name == "cross_entropy_loss":
        return cross_entropy_loss
    elif name == "fake_negative_weighted_loss":
        return fake_negative_weighted_loss
    elif name == "delayed_feedback_loss":
        return exp_delay_loss
    elif name == "tn_dp_pretraining_loss":
        return delay_tn_dp_loss
    elif name == "fsiw_loss":
        return fsiw_loss
    elif name == "esdfm_loss":
        return delay_tn_importance_weight_loss
    else:
        raise NotImplementedError("{} loss does not implemented".format(name))
