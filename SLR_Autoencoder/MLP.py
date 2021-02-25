from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from keras.layers import Input, UpSampling2D, concatenate
from keras.models import Model
from keras.utils import to_categorical
import keras
import data_process
data, label = data_process.load_data()
data=data_process.normalize_data(data)
#data processing

num_classes = 15
label_onehot=to_categorical(label,15)
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
output=Dense(units=15, activation='softmax')(merge3)
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