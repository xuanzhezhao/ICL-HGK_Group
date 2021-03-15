
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.layers import Input, UpSampling2D, concatenate
from keras.models import Model
from keras.utils import to_categorical
import keras
import data_process
data, label = data_process.load_data()
data=data_process.normalize_data(data)
#data processing

num_classes = 26
label_onehot=to_categorical(label,num_classes)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.4)
y_train=to_categorical(y_train,num_classes)
y_test=to_categorical(y_test,num_classes)

inputs = Input(shape=(20,))

"""
dense1 = Dense(64, activation='relu')(inputs[:,0:5])
dense2 = Dense(64, activation='relu')(inputs[:,5:10])
dense3 = Dense(64, activation='relu')(inputs[:,10:15])
dense4 = Dense(64, activation='relu')(inputs[:,15:20])
"""

#y = keras.layers.Lambda(lambda inputs: x * 1)(x)
IMU1 = keras.layers.Lambda(lambda inputs: inputs[:,:5]*1, input_shape=[20])(inputs)
IMU2 = keras.layers.Lambda(lambda inputs: inputs[:,5:10]*1, input_shape=[20])(inputs)
IMU3 = keras.layers.Lambda(lambda inputs: inputs[:,10:15]*1, input_shape=[20])(inputs)
IMU4 = keras.layers.Lambda(lambda inputs: inputs[:,15:20]*1, input_shape=[20])(inputs)


dense1 = Dense(units=32, activation='relu')(IMU1)
dense2 = Dense(units=32, activation='relu')(IMU2)
dense3 = Dense(units=32, activation='relu')(IMU3)
dense4 = Dense(units=32, activation='relu')(IMU4)

merge1=concatenate([dense1,dense2,dense3,dense4])
#merge1 = concatenate([drop4,up6], axis = 3)
#merge1=Lambda(add_op(dense1,dense2,dense3,dense4), output_shape=(64,), mask=None, arguments=None)
#merge1=keras.layers.Add()([dense1,dense2,dense3,dense4])
merge2=Dense(units=128, activation='relu')(merge1)
merge3=Dense(units=64, activation='relu')(merge2)
#drop1 = Dropout(0.2)(merge3)
#merge3=Dense(64, activation='relu')(drop1)
output=Dense(units=26, activation='softmax')(merge3)
model = Model(inputs = inputs, outputs = output)
model.summary()

# initiate Adam optimizer
#opt = keras.optimizers.Adam(lr=0.001, decay=1e-5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train , batch_size=64,epochs=50, validation_split=0.3,verbose=1)

#Validation score
score = model.evaluate(x_test, y_test, verbose=1)
print('Validation loss:', score[0])
print('Validation accuracy:', score[1])

y1_train_loss=history.history['loss']
y1_val_loss=history.history['val_loss']
y1_train_acc=history.history['categorical_accuracy']
y1_val_acc=history.history['val_categorical_accuracy']
x1=history.epoch

x = np.arange(0,20) #numpy.linspace(开始，终值(含终值))，个数)
plt.title('Performance of SLRNet')
#plt.plot(x,y)
#常见线的属性有：color,label,linewidth,linestyle,marker等
#plt.plot(x1, y1_train_loss, color='r', label='SLR_Net train loss',linestyle='-',linewidth=1.6)
#plt.plot(x1, y1_val_loss, 'b', label='SLR_Net val loss ',linestyle='-',linewidth=1.6)#'b'指：color='blue'

plt.plot(x1, y1_train_acc, color='r', label='SLR_Net train acc',linestyle='--',linewidth=1.6)
plt.plot(x1, y1_val_acc, 'b', label='SLR_Net val acc ',linestyle='--',linewidth=1.6)#'b'指：color='blue'

plt.legend(fontsize=10)  #显示上面的label
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.axis([0, 50, 0.2, 1.1])#设置坐标范围axis([xmin,xmax,ymin,ymax])
#plt.ylim(-1,1)#仅设置y轴坐标范围
plt.show()