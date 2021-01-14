#此文件待更新成实时获取串口数据
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import svm
import joblib


# 加载数据和模型
X_train, y_train = joblib.load('train.pkl')
X_test, y_test = joblib.load('test.pkl')
model = joblib.load('model.pkl')

# 预测
print('predicting')
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

# 结果分析
conf_mat_train = confusion_matrix(y_train, pred_train)
print('train_confusion_matrix')
print(conf_mat_train)

conf_mat_test = confusion_matrix(y_test, pred_test)
print('test_confusion_matrix')
print(conf_mat_test)
