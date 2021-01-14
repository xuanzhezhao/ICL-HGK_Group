import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
import prettytable
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,plot_precision_recall_curve
import matplotlib.pyplot as plt
#归一化处理
from sklearn.metrics import accuracy_score
def normalize(data):
    mu=np.mean(data,axis=0)
    sigma=np.std(data,axis=0)
    return (data-mu)/sigma


"""Load the dataset"""
print("Loading the dataset...")
df=pd.read_table('data_60/A/A.txt',header=None,delimiter=',')
data_A=normalize(df.values)
df=pd.read_table('data_60/B/B.txt',header=None,delimiter=',')
data_B=normalize(df.values)
df=pd.read_table('data_60/C/C.txt',header=None,delimiter=',')
data_C=normalize(df.values)
df=pd.read_table('data_60/D/D.txt',header=None,delimiter=',')
data_D=normalize(df.values)
df=pd.read_table('data_60/E/E.txt',header=None,delimiter=',')
data_E=normalize(df.values)
df=pd.read_table('data_60/F/F.txt',header=None,delimiter=',')
data_F=normalize(df.values)
df=pd.read_table('data_60/G/G.txt',header=None,delimiter=',')
data_G=normalize(df.values)
df=pd.read_table('data_60/H/H.txt',header=None,delimiter=',')
data_H=normalize(df.values)
df=pd.read_table('data_60/I/I.txt',header=None,delimiter=',')
data_I=normalize(df.values)
df=pd.read_table('data_60/J/J.txt',header=None,delimiter=',')
data_J=normalize(df.values)
df=pd.read_table('data_60/K/K.txt',header=None,delimiter=',')
data_K=normalize(df.values)
df=pd.read_table('data_60/L/L.txt',header=None,delimiter=',')
data_L=normalize(df.values)
df=pd.read_table('data_60/M/M.txt',header=None,delimiter=',')
data_M=normalize(df.values)
df=pd.read_table('data_60/N/N.txt',header=None,delimiter=',')
data_N=normalize(df.values)
df=pd.read_table('data_60/O/O.txt',header=None,delimiter=',')
data_O=normalize(df.values)
df=pd.read_table('data_60/P/P.txt',header=None,delimiter=',')
data_P=normalize(df.values)
df=pd.read_table('data_60/Q/Q.txt',header=None,delimiter=',')
data_Q=normalize(df.values)
df=pd.read_table('data_60/R/R.txt',header=None,delimiter=',')
data_R=normalize(df.values)
df=pd.read_table('data_60/S/S.txt',header=None,delimiter=',')
data_S=normalize(df.values)
df=pd.read_table('data_60/T/T.txt',header=None,delimiter=',')
data_T=normalize(df.values)
df=pd.read_table('data_60/U/U.txt',header=None,delimiter=',')
data_U=normalize(df.values)
df=pd.read_table('data_60/V/V.txt',header=None,delimiter=',')
data_V=normalize(df.values)
df=pd.read_table('data_60/W/W.txt',header=None,delimiter=',')
data_W=normalize(df.values)
df=pd.read_table('data_60/X/X.txt',header=None,delimiter=',')
data_X=normalize(df.values)
df=pd.read_table('data_60/Y/Y.txt',header=None,delimiter=',')
data_Y=normalize(df.values)
df=pd.read_table('data_60/Z/Z.txt',header=None,delimiter=',')
data_Z=normalize(df.values)

row=100 #row size of data, samples per class
col=360 #col size of data, dimension of sample(=sample frequency*6)
A=np.hstack((np.reshape(data_A,[row,col]),np.ones(row).reshape(row,1)))
B=np.hstack((np.reshape(data_B,[row,col]),np.ones(row).reshape(row,1)+1))
C=np.hstack((np.reshape(data_C,[row,col]),np.ones(row).reshape(row,1)+2))
D=np.hstack((np.reshape(data_D,[row,col]),np.ones(row).reshape(row,1)+3))
E=np.hstack((np.reshape(data_E,[row,col]),np.ones(row).reshape(row,1)+4))
F=np.hstack((np.reshape(data_F,[row,col]),np.ones(row).reshape(row,1)+5))
G=np.hstack((np.reshape(data_G,[row,col]),np.ones(row).reshape(row,1)+6))
H=np.hstack((np.reshape(data_H,[row,col]),np.ones(row).reshape(row,1)+7))
I=np.hstack((np.reshape(data_I,[row,col]),np.ones(row).reshape(row,1)+8))
J=np.hstack((np.reshape(data_J,[row,col]),np.ones(row).reshape(row,1)+9))
K=np.hstack((np.reshape(data_K,[row,col]),np.ones(row).reshape(row,1)+10))
L=np.hstack((np.reshape(data_L,[row,col]),np.ones(row).reshape(row,1)+11))
M=np.hstack((np.reshape(data_M,[row,col]),np.ones(row).reshape(row,1)+12))
N=np.hstack((np.reshape(data_N,[row,col]),np.ones(row).reshape(row,1)+13))
O=np.hstack((np.reshape(data_O,[row,col]),np.ones(row).reshape(row,1)+14))
P=np.hstack((np.reshape(data_P,[row,col]),np.ones(row).reshape(row,1)+15))
Q=np.hstack((np.reshape(data_Q,[row,col]),np.ones(row).reshape(row,1)+16))
R=np.hstack((np.reshape(data_R,[row,col]),np.ones(row).reshape(row,1)+17))
S=np.hstack((np.reshape(data_S,[row,col]),np.ones(row).reshape(row,1)+18))
T=np.hstack((np.reshape(data_T,[row,col]),np.ones(row).reshape(row,1)+19))
U=np.hstack((np.reshape(data_U,[row,col]),np.ones(row).reshape(row,1)+20))
V=np.hstack((np.reshape(data_V,[row,col]),np.ones(row).reshape(row,1)+21))
W=np.hstack((np.reshape(data_W,[row,col]),np.ones(row).reshape(row,1)+22))
X=np.hstack((np.reshape(data_X,[row,col]),np.ones(row).reshape(row,1)+23))
Y=np.hstack((np.reshape(data_Y,[row,col]),np.ones(row).reshape(row,1)+24))
Z=np.hstack((np.reshape(data_Z,[row,col]),np.ones(row).reshape(row,1)+25))
print("Finished, data has been successfully loaded!")



"""Split of training data and test data"""
#训练集分割
total=100 #total samples for each class
train_num=70 #trainig samples for each class
class_num=26
raw_data=np.vstack((A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z))
#raw_data=np.vstack((A,B,C,D,G))
index = [i for i in range(len(raw_data))]
np.random.shuffle(index) # shuffle the index of train_x
raw_data=raw_data[index,:]

X=raw_data[:,:-1]
y=raw_data[:,-1]
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#print(y_test)



# 获取支持向量机模型与训练
predictor = svm.SVC(gamma="scale", C=1.5, decision_function_shape='ovr', kernel='rbf')
predictor.fit(X_train, y_train)
print("The parameters of model: \n",predictor)



# 预测结果
result = predictor.predict(X_test)
"""
print("predict result:")
print(result)
print("ground truth:")
print(y_test)
"""
print("accuracy:",accuracy_score(y_test,result))



#混淆矩阵
confusion_m=confusion_matrix(y_test,result)
confusion_m_table=prettytable.PrettyTable()
for i in range(class_num):
    confusion_m_table.add_row(confusion_m[i,:])
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
#skplt.metrics.plot_roc(y_test, result)
#plt.show()


#保存模型
with open('svm.pickle', 'wb') as fw:
    pickle.dump(predictor, fw)



