import pandas as pd
import numpy as np
import pickle
import serial
import time

#归一化处理
def normalize(data):
    mu=np.mean(data,axis=0)
    sigma=np.std(data,axis=0)
    return (data-mu)/sigma

Classification_libary=['A','B','C','D','E','F','G','H','I',
                       'J','K','L','M','N','O','P','Q','R',
                       'S','T','U','V','W','X','Y','Z'] #26分类字母库
# 设置端口变量和值
serialPosrt = "COM5"
# 设置波特率变量和值
baudRate = 115200
#设置超时时间,单位为s
timeout = 0.5
# 接受串口数据
ser = serial.Serial(serialPosrt, baudRate)
# 循环获取数据(条件始终为真)
size_col=60
test_example = np.zeros(6*size_col).reshape(size_col, 6)
print("Loading SVM model...")
with open('svm.pickle', 'rb') as fr:
    new_svm = pickle.load(fr)
print("Start to record!")

while 1:
    str = ser.readline()
    sample_time = 0
    if len(str) > 5:
        while  sample_time < size_col:
            str = np.array(
                eval(
                    str.strip()))  # strip() is used to remove irrelevant syntax '\r\n' and eval() is used to remove the ''
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # set data precise for 0.001
            test_example[sample_time, :] = str
            sample_time = sample_time + 1
            # 把数据打印到终端显示
            str = ser.readline()
        test_example = normalize(test_example)  # 实时采集到大小为6x60的一组样本数据，并归一化
        #print(test_example)
        print('Data collected. Start classification...')
        print("predicted value:")
        pred_result = new_svm.predict(test_example.reshape(1, 6 * size_col))  # 预测结果
        # print(pred_result)
        print(Classification_libary[int(pred_result[0]) - 1])  # 显示结果
        print('Recognized!')
        time.sleep(0.5)



