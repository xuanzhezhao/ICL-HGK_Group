
import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import pickle
import serial
import re
"""
df = pd.read_excel('letter.xlsx')

X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

X_train, X_test,y_train, y_test =model_selection.train_test_split(X,y,test_size=0.2)

clf =svm.SVC(gamma='scale',
             C=1, decision_function_shape='ovr',
             kernel='rbf')

clf.fit(X_train,y_train)

accuracy = clf.score(X_test,y_test)

print(accuracy)
with open('letter26', 'wb') as f:
    pickle.dump(clf, f)
"""

ser = serial.Serial('COM8', 9600)

pickle_in = open('letter26', 'rb')
clf = pickle.load(pickle_in)


line = ser.readline()
while line:
    line = line.strip()
    line = line.decode('utf-8')
    line = line.split(',')
    numbers=[]
    for x in line:
        if x != '':
            numbers.append(float(x))
    example_measure = np.array([numbers])
    print(example_measure)
    example_measure = example_measure.reshape(len(example_measure), -1)
    prediction = clf.predict(example_measure)
    if int(prediction) == 1:
        print('a')
    if int(prediction) == 2:
        print('b')
    if int(prediction) == 3:
        print('c')
    if int(prediction) == 4:
        print('d')
    if int(prediction) == 5:
        print('e')
    if int(prediction) == 6:
        print('f')
    if int(prediction) == 7:
        print('g')
    if int(prediction) == 8:
        print('h')
    if int(prediction) == 9:
        print('i')
    if int(prediction) == 10:
        print('j')
    if int(prediction) == 11:
        print('k')
    if int(prediction) == 12:
        print('l')
    if int(prediction) == 13:
        print('m')
    line = ser.readline()

"""
example_measure = np.array([[-0.38,1.55,-0.10,34.12,-73.61,86.24,-0.38,1.55,-0.10,12.45,-17.70,-253.91,-0.38,1.55,-0.10,-9.83,134.34,-220.09,-0.38,1.55,-0.10,94.30,158.87,74.95,-0.38,1.55,-0.10,69.15,50.42,244.87,-0.38,1.55,-0.10,91.74,-38.51,269.65,-0.38,1.55,-0.10,91.86,-167.60,-15.08,-0.38,1.55,-0.10,29.17,-112.18,-253.05,-0.38,1.55,-0.10,-18.55,48.52,-294.80,-0.38,1.55,-0.10,32.59,83.19,-54.93,-0.38,1.55,-0.10,62.81,13.73,162.23,-0.38,1.55,-0.10,36.13,1.65,218.32,-0.38,1.55,-0.10,17.70,-63.84,187.74,-0.38,1.55,-0.10,35.46,-150.51,45.47,-0.38,1.55,-0.10,33.26,-71.23,-31.92,-0.38,1.55,-0.10,15.50,-6.10,-26.55,],
                            [-0.52,1.21,0.28,50.90,31.49,208.31,-0.52,1.21,0.28,12.51,34.00,89.23,-0.52,1.21,0.28,-17.03,-4.58,-154.54,-0.52,1.21,0.28,-37.23,-38.21,-209.78,-0.52,1.21,0.28,-29.11,-106.45,-159.79,-0.52,1.21,0.28,0.92,-145.75,4.09,-0.52,1.21,0.28,-28.26,-148.86,111.08,-0.52,1.21,0.28,-22.34,-63.96,115.23,-0.52,1.21,0.28,-4.82,37.54,115.11,-0.52,1.21,0.28,-60.85,96.80,63.72,-0.52,1.21,0.28,-86.12,118.84,-26.86,-0.52,1.21,0.28,-67.38,114.50,-84.35,-0.52,1.21,0.28,-12.82,54.20,-87.83,-0.52,1.21,0.28,-7.26,15.01,-33.39,-0.52,1.21,0.28,6.16,4.39,-4.03,-0.52,1.21,0.28,1.65,0.12,6.41,],
                            [0.16,1.55,0.39,14.40,-82.21,88.99,0.16,1.55,0.39,-12.15,29.48,-174.07,0.16,1.55,0.39,9.70,160.40,-116.15,0.16,1.55,0.39,27.95,205.81,43.70,0.16,1.55,0.39,26.31,112.30,130.49,0.16,1.55,0.39,81.91,56.58,176.94,0.16,1.55,0.39,60.24,-0.55,134.52,0.16,1.55,0.39,85.27,-76.84,118.71,0.16,1.55,0.39,111.82,-143.68,16.24,0.16,1.55,0.39,107.91,-170.90,-126.04,0.16,1.55,0.39,27.28,-57.31,-255.13,0.16,1.55,0.39,20.57,28.99,-42.66,0.16,1.55,0.39,-0.98,17.88,6.96,0.16,1.55,0.39,5.86,13.06,2.20,0.16,1.55,0.39,18.62,14.95,-2.01,0.16,1.55,0.39,15.14,5.92,0.49,],
                            [-0.44,1.45,-0.31,50.66,-108.34,107.30,-0.44,1.45,-0.31,17.33,-8.48,-237.49,-0.44,1.45,-0.31,15.38,168.70,-228.64,-0.44,1.45,-0.31,97.29,182.25,160.52,-0.44,1.45,-0.31,133.67,-4.39,421.63,-0.44,1.45,-0.31,104.31,-179.38,99.55,-0.44,1.45,-0.31,59.94,-160.58,-426.45,-0.44,1.45,-0.31,24.35,74.52,-445.25,-0.44,1.45,-0.31,46.81,70.50,-43.15,-0.44,1.45,-0.31,36.19,4.70,254.52,-0.44,1.45,-0.31,99.85,7.93,378.72,-0.44,1.45,-0.31,53.65,-126.40,266.60,-0.44,1.45,-0.31,60.49,-183.23,-37.05,-0.44,1.45,-0.31,53.89,-2.99,-37.35,-0.44,1.45,-0.31,25.76,12.45,-18.98,-0.44,1.45,-0.31,15.44,0.43,-7.57,],
                            [-0.25,0.72,-1.04,12.08,-112.73,55.05,-0.25,0.72,-1.04,53.22,-247.31,38.76,-0.25,0.72,-1.04,27.40,-138.61,-94.54,-0.25,0.72,-1.04,18.92,38.39,-133.67,-0.25,0.72,-1.04,22.71,130.37,-55.60,-0.25,0.72,-1.04,3.11,172.00,32.04,-0.25,0.72,-1.04,14.83,145.26,116.21,-0.25,0.72,-1.04,41.44,70.01,153.63,-0.25,0.72,-1.04,52.43,-23.01,113.46,-0.25,0.72,-1.04,26.31,-114.44,116.21,-0.25,0.72,-1.04,89.29,-163.45,31.07,-0.25,0.72,-1.04,75.01,-122.25,-34.73,-0.25,0.72,-1.04,41.26,-39.06,-49.38,-0.25,0.72,-1.04,12.45,8.30,-21.12,-0.25,0.72,-1.04,10.07,9.52,-3.72,-0.25,0.72,-1.04,9.64,4.58,-3.11,],
                            [0.13,2.09,0.34,24.78,-118.10,91.67,0.13,2.09,0.34,-24.17,48.28,-259.46,0.13,2.09,0.34,26.49,204.35,-173.89,0.13,2.09,0.34,6.35,134.70,243.53,0.13,2.09,0.34,66.53,-112.12,349.37,0.13,2.09,0.34,15.20,-25.70,299.38,0.13,2.09,0.34,-49.13,133.67,120.06,0.13,2.09,0.34,-160.16,151.86,-138.61,0.13,2.09,0.34,-137.27,57.43,-317.50,0.13,2.09,0.34,-85.08,-119.14,-159.91,0.13,2.09,0.34,-3.72,-306.88,-16.66,0.13,2.09,0.34,5.49,-205.93,7.51,0.13,2.09,0.34,8.30,-36.74,-12.82,0.13,2.09,0.34,-4.15,4.94,3.05,0.13,2.09,0.34,12.21,11.66,0.73,0.13,2.09,0.34,3.72,0.31,3.05,],
                            [0.51,1.32,-0.19,-58.11,-94.60,54.69,0.51,1.32,-0.19,-66.16,-8.85,-139.65,0.51,1.32,-0.19,13.00,163.02,-133.73,0.51,1.32,-0.19,98.63,184.20,215.88,0.51,1.32,-0.19,135.80,-39.55,340.70,0.51,1.32,-0.19,90.39,-224.24,-17.70,0.51,1.32,-0.19,1.28,-125.24,-442.08,0.51,1.32,-0.19,-22.64,70.37,-193.54,0.51,1.32,-0.19,10.38,8.61,282.41,0.51,1.32,-0.19,-18.19,31.43,323.91,0.51,1.32,-0.19,-50.96,174.99,186.46,0.51,1.32,-0.19,-118.90,160.95,35.10,0.51,1.32,-0.19,-134.28,146.00,-165.34,0.51,1.32,-0.19,-72.33,-95.64,-290.16,0.51,1.32,-0.19,-8.67,-126.59,-113.71,0.51,1.32,-0.19,-5.80,-45.78,9.52,],
                            [-0.53,1.11,0.50,10.62,-10.93,230.96,-0.53,1.11,0.50,24.11,16.17,124.15,-0.53,1.11,0.50,-0.67,29.42,35.95,-0.53,1.11,0.50,7.26,-1.71,-142.76,-0.53,1.11,0.50,-33.33,-36.87,-169.07,-0.53,1.11,0.50,-2.08,-63.84,-105.29,-0.53,1.11,0.50,-23.01,-91.25,25.94,-0.53,1.11,0.50,-14.10,-78.19,167.18,-0.53,1.11,0.50,-46.57,-34.67,148.56,-0.53,1.11,0.50,-0.73,0.24,21.55,-0.53,1.11,0.50,1.89,6.65,-1.77,-0.53,1.11,0.50,-0.37,2.87,-23.32,-0.53,1.11,0.50,3.97,8.97,-4.82,-0.53,1.11,0.50,3.36,4.76,3.48,-0.53,1.11,0.50,3.11,-0.43,0.49,-0.53,1.11,0.50,3.72,0.67,2.14,],
                            [0.30,-0.60,-1.22,-65.92,-87.95,105.53,0.30,-0.60,-1.22,-58.17,-146.12,187.01,0.30,-0.60,-1.22,16.24,159.85,-70.86,0.30,-0.60,-1.22,12.21,95.58,10.44,0.30,-0.60,-1.22,-4.82,-256.04,21.48,0.30,-0.60,-1.22,49.68,16.05,28.44,0.30,-0.60,-1.22,51.21,223.88,178.04,0.30,-0.60,-1.22,9.64,144.17,185.97,0.30,-0.60,-1.22,49.62,46.20,150.57,0.30,-0.60,-1.22,91.00,-141.78,14.77,0.30,-0.60,-1.22,62.68,-332.40,-218.32,0.30,-0.60,-1.22,50.29,39.00,-28.02,0.30,-0.60,-1.22,8.79,8.61,-12.94,0.30,-0.60,-1.22,7.69,8.48,0.43,0.30,-0.60,-1.22,14.71,7.51,7.63,0.30,-0.60,-1.22,20.51,1.83,4.52,],
                            [-0.20,1.77,0.31,-15.38,-145.81,276.98,-0.20,1.77,0.31,15.44,20.87,-78.55,-0.20,1.77,0.31,-1.04,52.61,-109.56,-0.20,1.77,0.31,-5.37,4.21,26.98,-0.20,1.77,0.31,-5.43,-33.94,142.64,-0.20,1.77,0.31,2.08,9.70,121.09,-0.20,1.77,0.31,11.78,2.38,153.75,-0.20,1.77,0.31,-5.43,23.32,78.43,-0.20,1.77,0.31,-24.96,84.66,39.73,-0.20,1.77,0.31,-50.48,101.75,-30.21,-0.20,1.77,0.31,-32.59,88.32,-130.98,-0.20,1.77,0.31,1.95,-35.52,-162.29,-0.20,1.77,0.31,6.23,-79.90,-62.87,-0.20,1.77,0.31,0.92,-24.17,17.70,-0.20,1.77,0.31,0.67,-2.20,15.81,-0.20,1.77,0.31,4.27,0.24,8.24,],
                            [-1.00,0.83,0.24,23.13,-1.46,370.24,-1.00,0.83,0.24,-1.89,78.61,156.62,-1.00,0.83,0.24,-15.99,8.79,-298.65,-1.00,0.83,0.24,-50.72,-82.52,-279.11,-1.00,0.83,0.24,-20.02,-184.27,-48.22,-1.00,0.83,0.24,-32.41,-176.39,234.37,-1.00,0.83,0.24,-11.47,151.86,154.24,-1.00,0.83,0.24,-62.07,206.60,-68.42,-1.00,0.83,0.24,-15.44,42.11,-94.73,-1.00,0.83,0.24,18.01,-89.78,92.22,-1.00,0.83,0.24,48.34,-173.03,147.34,-1.00,0.83,0.24,90.58,-227.05,223.45,-1.00,0.83,0.24,68.36,-80.02,38.82,-1.00,0.83,0.24,33.14,-7.14,-12.76,-1.00,0.83,0.24,3.60,12.21,-14.34,-1.00,0.83,0.24,-1.34,7.63,-5.55,],
                            [-0.47,1.03,0.53,-35.34,79.16,227.48,-0.47,1.03,0.53,12.63,104.13,124.94,-0.47,1.03,0.53,3.97,40.77,137.45,-0.47,1.03,0.53,9.95,-128.60,129.52,-0.47,1.03,0.53,141.72,-184.51,-91.86,-0.47,1.03,0.53,57.37,-67.20,-104.37,-0.47,1.03,0.53,11.78,-2.75,-21.91,-0.47,1.03,0.53,3.36,-8.67,7.93,-0.47,1.03,0.53,11.29,-0.85,7.75,-0.47,1.03,0.53,24.60,2.26,14.04,-0.47,1.03,0.53,27.95,2.93,7.39,-0.47,1.03,0.53,14.83,-1.65,0.85,-0.47,1.03,0.53,4.15,-3.54,-2.99,-0.47,1.03,0.53,2.81,-0.85,-0.79,-0.47,1.03,0.53,2.93,0.98,0.12,-0.47,1.03,0.53,6.10,-2.20,0.73,],
                            [-0.57,1.17,-0.31,9.16,9.52,26.86,-0.57,1.17,-0.31,21.91,-97.66,-271.97,-0.57,1.17,-0.31,-25.94,-138.12,-261.41,-0.57,1.17,-0.31,-5.68,-80.51,62.32,-0.57,1.17,-0.31,-39.18,-9.95,356.14,-0.57,1.17,-0.31,-6.53,66.04,306.09,-0.57,1.17,-0.31,15.20,-33.14,-163.15,-0.57,1.17,-0.31,52.67,-44.13,-378.91,-0.57,1.17,-0.31,48.71,-63.84,-210.51,-0.57,1.17,-0.31,-48.46,-77.58,59.08,-0.57,1.17,-0.31,-49.50,-68.85,294.31,-0.57,1.17,-0.31,-31.62,-0.06,335.88,-0.57,1.17,-0.31,-32.47,27.04,171.69,-0.57,1.17,-0.31,0.12,6.96,13.79,-0.57,1.17,-0.31,9.22,-5.98,-16.42,-0.57,1.17,-0.31,3.23,-2.08,-7.57,]])
"""




