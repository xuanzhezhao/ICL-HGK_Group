import pandas as pd
import numpy as np
from sklearn import svm
import pickle
import matplotlib.pyplot as plt

#归一化处理
from sklearn.metrics import accuracy_score


def normalize(data):
    mu=np.mean(data,axis=0)
    sigma=np.std(data,axis=0)
    return (data-mu)/sigma


df=pd.read_table('data/A/A.txt',header=None,delimiter=',')
data_A=normalize(df.values)
#print(type(data_A),data_A.shape)
df=pd.read_table('data/M/M.txt',header=None,delimiter=',')
data_M=normalize(df.values)
#print(type(data_M),data_M.shape)
df=pd.read_table('data/K/K.txt',header=None,delimiter=',')
data_K=normalize(df.values)
#print(type(data_A),data_A.shape)
df=pd.read_table('data/W/W.txt',header=None,delimiter=',')
data_W=normalize(df.values)


a=np.hstack((np.reshape(data_A,[150,900]),np.ones(150).reshape(150,1)))
m=np.hstack((np.reshape(data_M,[150,900]),np.ones(150).reshape(150,1)+1))
k=np.hstack((np.reshape(data_K,[150,900]),np.ones(150).reshape(150,1)+2))
w=np.hstack((np.reshape(data_W,[150,900]),np.ones(150).reshape(150,1)+3))

train_num=100

train_x=np.vstack((a[0:train_num,:],m[0:train_num,:],k[0:train_num,:],w[0:train_num,:]))
test_x=np.vstack((a[train_num:,:],m[train_num:,:],k[train_num:,:],w[train_num:,:]))


index = [i for i in range(len(train_x))] # test_data为测试数据
np.random.shuffle(index) # 打乱索引
train_x=train_x[index,:]
index = [i for i in range(len(test_x))]
np.random.shuffle(index)
test_x=test_x[index,:]


train_y=np.array(train_x[:,-1],int)
train_x=np.delete(train_x,-1,axis=1)
test_y=np.array(test_x[:,-1],int)
test_x=np.delete(test_x,-1,axis=1)

#print(train_y)
# 获取一个支持向量机模型
predictor = svm.SVC(gamma='scale', C=1.5, decision_function_shape='ovr', kernel='rbf')
# 把数据丢进去
predictor.fit(train_x, train_y)
# 预测结果
result = predictor.predict(test_x)
# 准确率估计
accurancy = np.sum(np.equal(result, test_y)) / (600-4*train_num)
print(test_y)
print(result)
print(accurancy)

with open('svm.pickle', 'wb') as fw:
    pickle.dump(predictor, fw)

"""
with open('svm.pickle', 'rb') as fr:
    new_svm = pickle.load(fr)
    print("predicted value:")
    #print(test_x.shape)
    example=test_x[3]
    pred_result=new_svm.predict(example.reshape(1,900))

    print (pred_result)
    #print(type(pred_result))
    print("ground true value:")
    print(test_y[3])
"""
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
w = predictor.coef_
b = predictor.intercept_
ax = plt.subplot(111, projection='3d')
x = np.arange(0, 1, 0.01)
y = np.arange(0, 1, 0.11)
x, y = np.meshgrid(x, y)
z = (w[0, 0] * x + w[0, 1] * y + b) / (-w[0, 2])
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1)
"""



"""
scores = []
for m in range(1,len(train_x)):#循环10-80
    predictor.fit(train_x[:m],train_y[:m])
    y_train_predict = predictor.predict(train_x[:m])
    y_val_predict = predictor.predict(test_x)
    scores.append(accuracy_score(y_train_predict,train_y[:m]))
    print("Iteration:",m)
    print(scores[m-1])

plt.plot(range(1,len(train_x)),scores,c='green', alpha=0.6)
"""