import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf 

import matplotlib.pyplot as plt

#loading data 

x=np.load("x_final_training.npy")
y=np.load("y_final_training.npy")
x_test=np.load("x_final_test.npy")
y_test=np.load("y_final_test.npy")


X=x[...,np.newaxis]


print("X shape ",x.shape)
print("Y shape ",y.shape)

#Train-test split
X_train=X
X_test=x_test
y_train=y
y_test=y_test

input=X_train.shape[1:]

model=tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=input))
model.add(tf.keras.layers.MaxPooling2D((2,2)))

model.add(tf.keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2,2)))


model.add(tf.keras.layers.BatchNormalization())


model.add(tf.keras.layers.Reshape((-1,64)))


model.add(tf.keras.layers.GRU(64,return_sequences=False))

model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

#compile 
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()



history=model.fit(
    X_train,y_train,
    epochs=35,
    batch_size=32,
    validation_data=(X_test,y_test)
)


loss,acc=model.evaluate(X_test,y_test)
print("test accuracy" ,acc)
print("loss",loss)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(['Model Accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["Train","validation"])
plt.show()


model.save("illness_detection_model.h5")


print("Model saved ")
