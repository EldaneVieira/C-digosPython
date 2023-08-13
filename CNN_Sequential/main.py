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

model=keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),#Dimensão 32x32 e 3 canais RGB. Não foi feito um redimensionamento unindo as dimensões
        layers.Conv2D(32,3,padding='valid',activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,3,activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu'),
        layers.Flatten(),
        #setting the fully-connected
        layers.Dense(64,activation='relu'),
        layers.Dense(10)
    ]
)

print(model.summary)


model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),
    metrics=["accuracy"]
)

model.fit(x_train,y_train,batch_size=64,epochs=10,verbose=2)
model.evaluate(x_test,y_test,batch_size=64,verbose=2)