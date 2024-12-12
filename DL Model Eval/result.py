import numpy as np
import pandas as pd
import os
import sklearn.metrics as metrics

DIR = os.path.dirname(__file__)


def print_aprf1_on_full(y_label, y_pred):
    tp = len(y_label[(y_label == 1) & (y_pred == 1)])
    fp = len(y_label[(y_label == 0) & (y_pred == 1)])
    tn = len(y_label[(y_label == 0) & (y_pred == 0)])
    fn = len(y_label[(y_label == 1) & (y_pred == 0)])
    a = (tp + tn) / (tp + fp + tn + fn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f1 = 2 * p * r / (p + r)
    print(f"|{tn}|{fp}|")
    print(f"|{fn}|{tp}|")
    print(f"A = {a*100:.2f}")
    print(f"P = {p*100:.2f}")
    print(f"R = {r*100:.2f}")
    print(f"F1 = {f1*100:.2f}")
    false_alarm_rate = fp / (tp + fp)
    print(f"False Alarm Rate = {false_alarm_rate*100:.2f}")
    failed = (y_pred == -1).sum()
    if failed:
        print(f"failed: {failed}")


def print_aprf1(y_label, y_pred):
    print_aprf1_on_full(y_label, y_pred)


ps_all = pd.read_parquet(DIR + "/input-data.parquet")
y_label = ps_all.label.values

print(len(ps_all))

linevul = np.load("linevul.npy")
print(metrics.confusion_matrix(y_label, linevul))
print_aprf1(y_label, linevul)


vulberta_cnn = np.load("vulberta_cnn.npy")
print(metrics.confusion_matrix(y_label, vulberta_cnn))
print_aprf1(y_label, vulberta_cnn)


vulberta_mlp = np.load("vulberta_mlp.npy")
print(metrics.confusion_matrix(y_label, vulberta_mlp))
print_aprf1(y_label, vulberta_mlp)


devign = np.load("devign.npy")
print(metrics.confusion_matrix(y_label, devign))
print_aprf1(y_label, devign)


ivdetect = np.load("ivdetect.npy")
print(metrics.confusion_matrix(y_label, ivdetect))
print_aprf1(y_label, ivdetect)

