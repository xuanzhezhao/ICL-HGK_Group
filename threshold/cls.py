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

pickle_in = open('save_letter.pickle', 'rb')
Xy = pickle.load(pickle_in)

X=Xy[0]
y=Xy[1]

X_train, X_test,y_train, y_test =model_selection.train_test_split(X,y,test_size=0.2)



clf =svm.SVC(gamma='scale',
             C=1, decision_function_shape='ovr',
             kernel='rbf',probability=True)

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)

with open('letter26', 'wb') as f1:
    pickle.dump(clf, f1)