import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.datasets
from keras.src.datasets import cifar10

(x_train, y_train),(x_test,y_test) = cifar10.load_data()

x_train = x_train.astype("float32")/255.0 #divisão para normalização
x_test = x_test.astype("float32")/255.0

def my_model_functional():
    inputs = keras.Input(shape=(32,32,3))
    x=layers.Conv2D(32,3)(inputs) #3 é o tamanho do kernek que corresponte a largura e altura do filro da convolução
    x=layers.BatchNormalization()(x) #Envia para a normalização antes de enviar para a função de ativação
    x=keras.activations.relu(x)
    x=layers.MaxPooling2D()(x)
    x=layers.Conv2D(64,3,padding='same')(x)
    x=layers.BatchNormalization()(x)
    x=keras.activations.relu(x)
    x=layers.Conv2D(128,3)(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x=layers.Flatten()(x) #O Flaten serve para ter um "match" da saída da convolution com a entrada da fully-connected

    #setting the fully-connected
    x=layers.Dense(64,activation='relu')(x)
    outputs=layers.Dense(10)(x)
    model=keras.Model(inputs=inputs, outputs=outputs)
    return model

#Chamada da rede Functional
model=my_model_functional()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=2)
model.evaluate(x_test,y_test,batch_size=64,verbose=2)