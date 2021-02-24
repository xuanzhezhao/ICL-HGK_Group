import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import data_process

#Load sign language dataset
data, label = data_process.load_data()
data=data_process.normalize_data(data)
#data processing


x_train, x_test, y_train, y_test = train_test_split(data, label, test_size = 0.3)
num_classes = 15
#y_train=to_categorical(y_train,num_classes)
#y_test=to_categorical(y_test,num_classes)


#print(x_train.shape)
#print(y_train.shape)
print(type(y_test))
#print(y_test)
#Autoencoder
model = Sequential()
model.add(Dense(3, name='representation', input_shape=(20,)))
model.add(Activation('relu'))
model.add(Dense(20))
model.add(Activation('relu'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mse'])
print(model.summary())
epochs = 200
validation_split = 0.2
history = model.fit(data, data, batch_size=128,
          epochs=epochs, validation_split=validation_split)

def predict_representation(model, data, layer_name='representation'):
  ## We form a new model. Instead of doing \psi\phi(x), we only take \phi(x)
  ## To do so, we use the layer name
  intermediate_layer_model = Model(inputs=model.input,
                                   outputs=model.get_layer(layer_name).output)
  representation = intermediate_layer_model.predict(data)
  representation = representation.reshape(representation.shape[0], -1)
  return representation

representation = predict_representation(model, x_train)

def plot_representation_label(representation, labels, plot3d=1):
    ## Function used to plot the representation vectors and assign different
    ## colors to the different classes

    # First create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    # In case representation dimension is 3, we can plot in a 3d projection too
    if plot3d:
        ax = fig.add_subplot(111, projection='3d')

    # Check number of labels to separate by colors
    #n_labels = labels.max() + 1
    #n_labels=n_labels.astype('int32')
    n_labels=10
    # Color map, and give different colors to every label
    cm = plt.get_cmap('gist_rainbow')
    ax.set_prop_cycle(color=[cm(1. * i / (n_labels)) for i in range(n_labels)])
    # Loop is to plot different color for each label
    for l in range(n_labels):
        # Only select indices for corresponding label
        index = labels == l
        ind=index.reshape((len(index,)))
        print(ind.shape)
        if plot3d:
            ax.scatter(representation[ind, 0], representation[ind, 1],
                       representation[ind, 2], label=str(l))
        else:
            ax.scatter(representation[ind, 0], representation[ind, 1], label=str(l))
    ax.legend()
    plt.title('Features in the representation space with corresponding label')
    plt.show()
    return fig, ax


plot_representation_label(representation, y_train)

