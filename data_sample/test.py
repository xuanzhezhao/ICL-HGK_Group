import pandas as pd
import numpy as np
import pickle
import serial

#归一化处理
def normalize(data):
    mu=np.mean(data,axis=0)
    sigma=np.std(data,axis=0)
    return (data-mu)/sigma

Classification_libary=['A','M','K','W'] #4分类字母库
# 设置端口变量和值
serialPosrt = "COM5"
# 设置波特率变量和值
baudRate = 9600
# 设置超时时间,单位为s
timeout = 0.5
# 接受串口数据
ser = serial.Serial(serialPosrt, baudRate)
# 循环获取数据(条件始终为真)
test_example = np.zeros(900).reshape(150, 6)


while 1:
    # str = ser.readline()
    str = ser.readline()
    sample_time = 0
    flag=0
    while len(str) > 7 and sample_time < 150:
        str = np.array(
            eval(str.strip()))  # strip() is used to remove irrelevant syntax '\r\n' and eval() is used to remove the ''
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})  # set data precise for 0.001
        test_example[sample_time, :] = str
        sample_time = sample_time + 1
        # 把数据打印到终端显示
        # print('ok')
        str = ser.readline()
    test_example=normalize(test_example)#实时采集到大小为6x150的一组样本数据，并归一化
    print(test_example)
    print('Data collected. Start classification...')
    flag=1    #根据flag值执行判断，结束样本数据采集
    if flag==1:
        with open('svm.pickle', 'rb') as fr:
            new_svm = pickle.load(fr)  #加载训练好的SVM模型：svm.pickle文件

            print("predicted value:")
            pred_result = new_svm.predict(test_example.reshape(1, 900)) #预测结果
            print(pred_result)
            print(Classification_libary[pred_result[0]-1])  #显示结果
            print('Recognized!')




