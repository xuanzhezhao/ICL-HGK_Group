import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import pickle
import serial
import re
import random
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,plot_precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df_B = pd.read_table('sign/b.txt',header=None,sep=',')
B = np.array(df_B)
df_C = pd.read_table('sign/c.txt',header=None,sep=',')
C = np.array(df_C)
df_D = pd.read_table('sign/d.txt',header=None,sep=',')
D = np.array(df_D)
df_F = pd.read_table('sign/f.txt',header=None,sep=',')
F = np.array(df_F)
df_H = pd.read_table('sign/h.txt',header=None,sep=',')
df_G= pd.read_table('sign/g.txt',header=None,sep=',')
G = np.array(df_G)
H = np.array(df_H)
df_I = pd.read_table('sign/i.txt',header=None,sep=',')
I = np.array(df_I)
df_K = pd.read_table('sign/k.txt',header=None,sep=',')
K = np.array(df_K)
df_L = pd.read_table('sign/l.txt',header=None,sep=',')
L = np.array(df_L)
df_P= pd.read_table('sign/p.txt',header=None,sep=',')
P = np.array(df_P)
df_V = pd.read_table('sign/v.txt',header=None,sep=',')
V = np.array(df_V)
df_O = pd.read_table('sign/o.txt',header=None,sep=',')
O = np.array(df_O)
df_Q = pd.read_table('sign/q.txt',header=None,sep=',')
Q = np.array(df_Q)
df_R = pd.read_table('sign/r.txt',header=None,sep=',')
R = np.array(df_R)
df_W = pd.read_table('sign/w.txt',header=None,sep=',')
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


df = df.drop(df.columns[-1],axis=1)
print(df)
class_b = [1 for i in range(len(B))]
class_c = [2 for i in range(len(C))]
class_d = [3 for i in range(len(D))]
class_f = [5 for i in range(len(F))]
class_g = [6 for i in range(len(G))]
class_h = [7 for i in range(len(H))]
class_i = [8 for i in range(len(I))]
class_k = [10 for i in range(len(K))]
class_l = [11 for i in range(len(L))]
class_o = [14 for i in range(len(O))]
class_p = [15 for i in range(len(P))]
class_q = [16 for i in range(len(Q))]
class_r = [17 for i in range(len(R))]
class_v = [21 for i in range(len(V))]
class_w = [22 for i in range(len(W))]

y_label = np.append(class_b,class_c)
y_label = np.append(y_label,class_d)
y_label = np.append(y_label,class_f)
y_label = np.append(y_label,class_g)
y_label = np.append(y_label,class_h)
y_label = np.append(y_label,class_i)
y_label = np.append(y_label,class_k)
y_label = np.append(y_label,class_l)

y_label = np.append(y_label,class_o)
y_label = np.append(y_label,class_p)
y_label = np.append(y_label,class_q)
y_label = np.append(y_label,class_r)

y_label = np.append(y_label,class_v)
y_label = np.append(y_label,class_w)
y_label = y_label.reshape(654)


'''
i=0
j=0
k=0
while (j<len(A_Z)):
    while (i<j+100):
        A_Z[i] = np.array(A_Z[i])
        A_Z[i] = np.insert(A_Z[i],0,k)
        i=i+1
    j=j+100
    k=k+1
'''
#A_Z = np.array(A_Z)

#print(A_Z)

X = np.array(df)
y = np.array(y_label)

X_train, X_test,y_train, y_test =model_selection.train_test_split(X,y,test_size=0.2)



clf =svm.SVC(gamma='scale',
             C=1.5, decision_function_shape='ovr',
             kernel='rbf',probability=True)

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)
with open('letter26', 'wb') as f:
    pickle.dump(clf, f)
'''
class_num =26
result = clf.predict(X_test)
confusion_m=confusion_matrix(y_test,result)
#confusion_m_table=prettytable.PrettyTable()
#for i in range(class_num):
#    confusion_m_table.add_row(confusion_m[i,:])
print("The confusion matrix: \n")
#print(confusion_m_table)
from sklearn.metrics import classification_report
target_names = ['class A', 'class B', 'class C', 'class D', 'class E', 'class F', 'class G', 'class H'
                , 'class I', 'class J', 'class K', 'class L', 'class M', 'class N', 'class O', 'class P'
                , 'class Q', 'class R', 'class S', 'class T', 'class U', 'class V', 'class W', 'class X'
                , 'class Y', 'class Z']
print(classification_report(y_test, result, target_names=target_names))
def plot_confusion_matrix(confusion_mat):
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(class_num)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
plot_confusion_matrix(confusion_m)

import scikitplot as skplt
plot = skplt.metrics.plot_confusion_matrix(y_test, result, normalize=True)
plt.show()
'''