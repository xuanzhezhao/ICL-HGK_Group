import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import pickle
import serial
import re

#n,h,b,k,x,d

ser = serial.Serial('COM8', 9600)

pickle_in = open('letter26', 'rb')
clf = pickle.load(pickle_in)

s = ''

line = ser.readline()
#print(line)
while line:
    line = line.strip()
    line = line.decode('utf-8')
    line = line.split(',')
    numbers=[]
    #print(len(line))
    for x in line:
        if x != '':
            numbers.append(float(x))
    example_measure = np.array([numbers])
    #print(example_measure.shape)
    example_measure = example_measure.reshape(len(example_measure), -1)
    prediction = clf.predict(example_measure)

    if int(prediction) == 1:
        s = s + 'b'
        print(s)
    if int(prediction) == 2:
        s = s + 'c'
        print(s)
    if int(prediction) == 3:
        s = s + 'd'
        print(s)

    if int(prediction) == 5:
        s = s + 'f'
        print(s)
    if int(prediction) == 6:
        s = s + 'g'
        print(s)
    if int(prediction) == 7:
        s = s + 'h'
        print(s)
    if int(prediction) == 8:
        s = s + 'i'
        print(s)

    if int(prediction) == 10:
        s = s + 'k'
        print(s)
    if int(prediction) == 11:
        s = s + 'l'
        print(s)

    if int(prediction) == 14:
        s = s + 'o'
        print(s)
    if int(prediction) == 15:
        s = s + 'p'
        print(s)
    if int(prediction) == 16:
        s = s + 'q'
        print(s)
    if int(prediction) == 17:
        s = s + 'r'
        print(s)


    if int(prediction) == 21:
        s = s + 'v'
        print(s)
    if int(prediction) == 22:
        s = s + 'w'
        print(s)


    line = ser.readline()

