from sklearn import metrics
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(np.clip(-x, a_min=-1e50, a_max=1e20)))

def cal_auc(label, pos_prob):
    fpr, tpr, thresholds = metrics.roc_curve(label, pos_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc

def stable_log1pex(x):
    return -np.minimum(x, 0) + np.log(1+np.exp(-np.abs(x)))

def cal_llloss_with_logits(label, logits):
    ll = -np.mean(label*(-stable_log1pex(logits)) + (1-label)*(-logits - stable_log1pex(logits)))
    return ll

def prob_clip(x):
    return np.clip(x, a_min=1e-20, a_max=1)

def cal_llloss_with_neg_log_prob(label, neg_log_prob):
    ll = -np.mean((1-label)*neg_log_prob + label*(np.log(prob_clip(1 - prob_clip(np.exp(neg_log_prob))))))
    return ll

def cal_llloss_with_prob(label, prob):
    ll = -np.mean(label*np.log(prob_clip(prob)) + (1-label)*(np.log(prob_clip(1-prob))))
    return ll

def cal_prauc(label, pos_prob):
    precision, recall, thresholds = metrics.precision_recall_curve(label, pos_prob)
    area = metrics.auc(recall, precision)
    return area

def cal_acc(label, prob):
    label = np.reshape(label, (-1,))
    prob = np.reshape(label, (-1,))
    prob_acc = np.mean(label*prob)
    return prob_acc

def stable_softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)