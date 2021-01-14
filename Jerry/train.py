#文件是以github上最开始的4类数据作为输入
#14/01/2021 获取到26类手写数据，此文件待更新

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
import joblib


# 标签，数据命名格式为 标签.txt
targets = ['A', 'K', 'M', 'W']
X = []
y = []

# 数据读入，参数K指定K行数据为一个样本
K = 150
for t in targets:
    label = targets.index(t)
    with open(os.path.join('data', t+'.txt'), 'r') as f:
        cnt = 0
        item = []
        for l in f:
            if cnt % K == 0:
                if item:
                    X.append(item)
                    y.append(label)
                item = []
            try:
                item.extend([float(i) for i in l.split(',')])
                cnt += 1
            except:
                pass
        # 最后一个样本
        if cnt % K == 0:
            if item:
                X.append(item)
                y.append(label)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
joblib.dump((X_train, y_train), 'train.pkl')
joblib.dump((X_test, y_test), 'test.pkl')
print('Dimension of train data:', X_train.shape, y_train.shape)
print('Dimension of test data:', X_test.shape, y_test.shape)

print('training')
model = svm.SVC(verbose=True)
model.fit(X_train, y_train)

# 保存模型
print('\nsave model')
joblib.dump(model, 'model.pkl')
