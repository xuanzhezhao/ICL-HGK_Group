import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import pickle
import serial
import re
import random
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def load_data():
    df_B = pd.read_table('sign/b.txt', header=None, sep=',')
    B = np.array(df_B)
    df_C = pd.read_table('sign/c.txt', header=None, sep=',')
    C = np.array(df_C)
    df_D = pd.read_table('sign/d.txt', header=None, sep=',')
    D = np.array(df_D)
    df_F = pd.read_table('sign/f.txt', header=None, sep=',')
    F = np.array(df_F)
    df_H = pd.read_table('sign/h.txt', header=None, sep=',')
    df_G = pd.read_table('sign/g.txt', header=None, sep=',')
    G = np.array(df_G)
    H = np.array(df_H)
    df_I = pd.read_table('sign/i.txt', header=None, sep=',')
    I = np.array(df_I)
    df_K = pd.read_table('sign/k.txt', header=None, sep=',')
    K = np.array(df_K)
    df_L = pd.read_table('sign/l.txt', header=None, sep=',')
    L = np.array(df_L)
    df_P = pd.read_table('sign/p.txt', header=None, sep=',')
    P = np.array(df_P)
    df_V = pd.read_table('sign/v.txt', header=None, sep=',')
    V = np.array(df_V)
    df_O = pd.read_table('sign/o.txt', header=None, sep=',')
    O = np.array(df_O)
    df_Q = pd.read_table('sign/q.txt', header=None, sep=',')
    Q = np.array(df_Q)
    df_R = pd.read_table('sign/r.txt', header=None, sep=',')
    R = np.array(df_R)
    df_W = pd.read_table('sign/w.txt', header=None, sep=',')
    W = np.array(df_W)

    df = df_B.append(df_C)
    df = df.append(df_D)
    df = df.append(df_F)
    df = df.append(df_G)
    df = df.append(df_H)
    df = df.append(df_I)
    df = df.append(df_K)
    df = df.append(df_L)

    df = df.append(df_O)
    df = df.append(df_P)
    df = df.append(df_Q)
    df = df.append(df_R)
    df = df.append(df_V)
    df = df.append(df_W)

    df = df.drop(df.columns[-1], axis=1)
    data = pd.DataFrame(df).to_numpy()
    print(type(data))
    print(data.shape)

    class_b = [0 for i in range(len(B))]
    class_c = [1 for i in range(len(C))]
    class_d = [2 for i in range(len(D))]
    class_f = [3 for i in range(len(F))]
    class_g = [4 for i in range(len(G))]
    class_h = [5 for i in range(len(H))]
    class_i = [6 for i in range(len(I))]
    class_k = [7 for i in range(len(K))]
    class_l = [8 for i in range(len(L))]
    class_o = [9 for i in range(len(O))]
    class_p = [10 for i in range(len(P))]
    class_q = [11 for i in range(len(Q))]
    class_r = [12 for i in range(len(R))]
    class_v = [13 for i in range(len(V))]
    class_w = [14 for i in range(len(W))]

    y_label = np.append(class_b, class_c)
    y_label = np.append(y_label, class_d)
    y_label = np.append(y_label, class_f)
    y_label = np.append(y_label, class_g)
    y_label = np.append(y_label, class_h)
    y_label = np.append(y_label, class_i)
    y_label = np.append(y_label, class_k)
    y_label = np.append(y_label, class_l)

    y_label = np.append(y_label, class_o)
    y_label = np.append(y_label, class_p)
    y_label = np.append(y_label, class_q)
    y_label = np.append(y_label, class_r)

    y_label = np.append(y_label, class_v)
    y_label = np.append(y_label, class_w)
    y_label = y_label.reshape(654, 1)
    print(type(y_label))
    print(y_label.shape)
    return data, y_label

def normalize_data(data):
    normalized_data=data
    max=np.max(abs(data),axis=0)
    for i in range (0,len(data)):
        normalized_data[i,:]=normalized_data[i,:]/max
    return normalized_data

"""
if __name__=="__main__":
    data, label=load_data()
    norm_data=normalize_data(data)
    print(data.shape)
"""