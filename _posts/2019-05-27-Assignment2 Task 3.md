---
title: "Assginment 2 Task 3 Part 2 "
date: 2019-05-27
tags: [CNN, Deep learning]
excerpt: ""
mathjax: "true"
---

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import os
import gzip
import glob
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from time import time

```


```python
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,BatchNormalization,Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import TensorBoard
import keras.backend as K
from keras import backend as K
```

    Using TensorFlow backend.



```python
tf.__version__
```




    '1.12.0'




```python
num_classes = 96
img_size_flat = 28*28*3
img_shape_full = (28,28,3)
TotalSize = 96000
```


```python
x = []
y_class = []
p = 0
for i in glob.glob("classes1/*"):
    for j in glob.glob(i+'/*.jpeg'):
        im = cv2.imread(j)
        resize_img = cv2.resize(im  , (28 , 28))
        x.append(resize_img)
        temp = []
        temp.append(os.path.basename(i))
        temp.append(p)
        y_class.append(temp)
    p+=1
x = np.array(x)
y_class = np.array(y_class)
```


```python
x_train, x_test, y_train_cls, y_test_cls = train_test_split(x, y_class, test_size=0.40, random_state=42)
train_size = y_train_cls.shape[0]
test_size = y_test_cls.shape[0]
print(train_size,test_size)
```

    57600 38400



```python
b = np.zeros((train_size, num_classes))
b[np.arange(train_size), y_train_cls[:,1].astype(int)] = 1
y_train = b
```


```python
b = np.zeros((test_size, num_classes))
b[np.arange(test_size), y_test_cls[:,1].astype(int)] = 1
y_test = b
```


```python
imgplot = plt.imshow(x_train[3])
plt.show()
print(y_train[3])
```


![png](/images/Task12_Line_files/Task12_Line_8_0.png)


    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]



```python
def Def_Model1():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv1'))
    model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv3'))
    model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv4'))
    
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))


    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def Def_Model2():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))

    model.add(Conv2D(kernel_size=7, strides=1, filters=16,
                     activation='relu', name='layer_conv1'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))


    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def Def_Model3():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))

    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='valid',
                     activation='relu', name='layer_conv1'))

    model.add(BatchNormalization())

    #model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='valid',
                     activation='relu', name='layer_conv2'))

    model.add(BatchNormalization())


    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
def Def_Model4():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))

    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv1'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv2'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    
    model.add(Conv2D(kernel_size=7, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv3'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    model.add(Conv2D(kernel_size=7, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv4'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))
   
   
   

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def Def_Model5():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))

    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv1'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
 

    
    
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv2'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))
    
    
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv3'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=2, strides=2))
   

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def Def_Model6():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))

    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='valid',
                     activation='relu', name='layer_conv1'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    
    
    model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='valid',
                     activation='relu', name='layer_conv2'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())

    
    
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv3'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    
    model.add(Conv2D(kernel_size=3, strides=1, filters=64, padding='same',
                     activation='relu', name='layer_conv4'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())   
   

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def Def_Model7():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,3,)))

    model.add(Conv2D(kernel_size=5, strides=1, filters=128, padding='valid',
                     activation='relu', name='layer_conv1'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    
    
    model.add(Conv2D(kernel_size=3, strides=1, filters=128, padding='valid',
                     activation='relu', name='layer_conv2'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())

    
    
    model.add(Conv2D(kernel_size=5, strides=1, filters=64, padding='same',
                     activation='relu', name='layer_conv3'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    
    model.add(Conv2D(kernel_size=3, strides=1, filters=32, padding='same',
                     activation='relu', name='layer_conv4'))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())   
   

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model
```


```python
model1 = Def_Model1()
model1.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 28, 28, 32)        2432      
    _________________________________________________________________
    layer_conv3 (Conv2D)         (None, 28, 28, 32)        9248      
    _________________________________________________________________
    layer_conv4 (Conv2D)         (None, 28, 28, 32)        9248      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 32)        128       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 6272)              0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              6423552   
    _________________________________________________________________
    dense_1 (Dense)              (None, 96)                98400     
    =================================================================
    Total params: 6,543,008
    Trainable params: 6,542,944
    Non-trainable params: 64
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model1.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model1line/{}'.format(time()))
model1.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 26s 451us/step - loss: 1.2985 - acc: 0.6910 - val_loss: 0.3740 - val_acc: 0.8839
    Epoch 2/3
    57600/57600 [==============================] - 25s 435us/step - loss: 0.1119 - acc: 0.9656 - val_loss: 0.1111 - val_acc: 0.9655
    Epoch 3/3
    57600/57600 [==============================] - 25s 436us/step - loss: 0.0906 - acc: 0.9751 - val_loss: 0.1551 - val_acc: 0.9539





    <tensorflow.python.keras.callbacks.History at 0x7efd1ef6ff98>




```python
result = model1.evaluate(x=x_test,y=y_test)
for name, value in zip(model1.metrics_names, result):
    print(name, value)
model1.save("Task12_line1.h5")
```

    38400/38400 [==============================] - 5s 137us/step
    loss 0.1551327172468882
    acc 0.9538802083333333



```python
model2 = Def_Model2()
model2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 22, 22, 16)        2368      
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 22, 22, 16)        64        
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 11, 11, 16)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1936)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1024)              1983488   
    _________________________________________________________________
    dense_3 (Dense)              (None, 96)                98400     
    =================================================================
    Total params: 2,084,320
    Trainable params: 2,084,288
    Non-trainable params: 32
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model2.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model2line/{}'.format(time()))
model2.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 8s 132us/step - loss: 1.2989 - acc: 0.6456 - val_loss: 0.3583 - val_acc: 0.8669
    Epoch 2/3
    57600/57600 [==============================] - 7s 127us/step - loss: 0.1637 - acc: 0.9440 - val_loss: 0.1246 - val_acc: 0.9548
    Epoch 3/3
    57600/57600 [==============================] - 8s 140us/step - loss: 0.0908 - acc: 0.9703 - val_loss: 0.1051 - val_acc: 0.9661





    <tensorflow.python.keras.callbacks.History at 0x7efd00229f98>




```python
result = model2.evaluate(x=x_test,y=y_test)
for name, value in zip(model2.metrics_names, result):
    print(name, value)
model2.save("Task12_line2.h5")
```

    38400/38400 [==============================] - 2s 46us/step
    loss 0.10507126471512795
    acc 0.9661197916666666



```python
model3 = Def_Model3()
model3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 24, 24, 32)        2432      
    _________________________________________________________________
    batch_normalization (BatchNo (None, 24, 24, 32)        128       
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 20, 20, 32)        25632     
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 20, 20, 32)        128       
    _________________________________________________________________
    flatten (Flatten)            (None, 12800)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              13108224  
    _________________________________________________________________
    dense_1 (Dense)              (None, 96)                98400     
    =================================================================
    Total params: 13,234,944
    Trainable params: 13,234,816
    Non-trainable params: 128
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model3.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model3line/{}'.format(time()))
model3.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 30s 516us/step - loss: 2.0063 - acc: 0.6508 - val_loss: 0.3616 - val_acc: 0.9158
    Epoch 2/3
    57600/57600 [==============================] - 28s 485us/step - loss: 0.3884 - acc: 0.9242 - val_loss: 0.5225 - val_acc: 0.8898
    Epoch 3/3
    57600/57600 [==============================] - 28s 480us/step - loss: 0.3010 - acc: 0.9514 - val_loss: 0.2604 - val_acc: 0.9588





    <tensorflow.python.keras.callbacks.History at 0x7f2582a01550>




```python
result = model3.evaluate(x=x_test,y=y_test)
for name, value in zip(model3.metrics_names, result):
    print(name, value)
model3.save("Task12_line3.h5")
```

    38400/38400 [==============================] - 5s 129us/step
    loss 0.26036784167789545
    acc 0.9587760416666666



```python
model4 = Def_Model4()
model4.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 28, 28, 32)        2432      
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 14, 14, 32)        25632     
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    layer_conv3 (Conv2D)         (None, 7, 7, 32)          50208     
    _________________________________________________________________
    batch_normalization_6 (Batch (None, 7, 7, 32)          128       
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 3, 3, 32)          0         
    _________________________________________________________________
    layer_conv4 (Conv2D)         (None, 3, 3, 32)          50208     
    _________________________________________________________________
    batch_normalization_7 (Batch (None, 3, 3, 32)          128       
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 1, 1, 32)          0         
    _________________________________________________________________
    flatten_3 (Flatten)          (None, 32)                0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1024)              33792     
    _________________________________________________________________
    dense_7 (Dense)              (None, 96)                98400     
    =================================================================
    Total params: 261,184
    Trainable params: 260,928
    Non-trainable params: 256
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model4.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model4line/{}'.format(time()))
model4.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 14s 240us/step - loss: 0.6133 - acc: 0.8295 - val_loss: 0.0820 - val_acc: 0.9713
    Epoch 2/3
    57600/57600 [==============================] - 13s 225us/step - loss: 0.0406 - acc: 0.9884 - val_loss: 0.0160 - val_acc: 0.9952
    Epoch 3/3
    57600/57600 [==============================] - 13s 228us/step - loss: 0.0377 - acc: 0.9876 - val_loss: 0.0851 - val_acc: 0.9755





    <tensorflow.python.keras.callbacks.History at 0x7efcf83a9160>




```python
result = model4.evaluate(x=x_test,y=y_test)
for name, value in zip(model4.metrics_names, result):
    print(name, value)
model4.save("Task12_line4.h5")
```

    38400/38400 [==============================] - 4s 91us/step
    loss 0.08513712100522146
    acc 0.9754947916666666



```python
model5 = Def_Model5()
model5.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 28, 28, 32)        2432      
    _________________________________________________________________
    dropout (Dropout)            (None, 28, 28, 32)        0         
    _________________________________________________________________
    batch_normalization_8 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 28, 28, 32)        25632     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 28, 28, 32)        0         
    _________________________________________________________________
    batch_normalization_9 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 14, 14, 32)        0         
    _________________________________________________________________
    layer_conv3 (Conv2D)         (None, 14, 14, 32)        25632     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    batch_normalization_10 (Batc (None, 14, 14, 32)        128       
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 1568)              0         
    _________________________________________________________________
    dense_8 (Dense)              (None, 1024)              1606656   
    _________________________________________________________________
    dense_9 (Dense)              (None, 96)                98400     
    =================================================================
    Total params: 1,759,136
    Trainable params: 1,758,944
    Non-trainable params: 192
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model5.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model5line/{}'.format(time()))
model5.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 26s 443us/step - loss: 0.9788 - acc: 0.7114 - val_loss: 1.8732 - val_acc: 0.4359
    Epoch 2/3
    57600/57600 [==============================] - 25s 436us/step - loss: 0.1037 - acc: 0.9651 - val_loss: 1.2285 - val_acc: 0.6473
    Epoch 3/3
    57600/57600 [==============================] - 26s 447us/step - loss: 0.0607 - acc: 0.9798 - val_loss: 1.0814 - val_acc: 0.7184





    <tensorflow.python.keras.callbacks.History at 0x7efca9743a20>




```python
result = model5.evaluate(x=x_test,y=y_test)
for name, value in zip(model5.metrics_names, result):
    print(name, value)
model5.save("Task12_line5.h5")
```

    38400/38400 [==============================] - 5s 125us/step
    loss 1.081372225607435
    acc 0.7184114583333333



```python
model6 = Def_Model6()
model6.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 24, 24, 32)        2432      
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 24, 24, 32)        0         
    _________________________________________________________________
    batch_normalization_11 (Batc (None, 24, 24, 32)        128       
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 22, 22, 32)        9248      
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 22, 22, 32)        0         
    _________________________________________________________________
    batch_normalization_12 (Batc (None, 22, 22, 32)        128       
    _________________________________________________________________
    layer_conv3 (Conv2D)         (None, 22, 22, 32)        25632     
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 22, 22, 32)        0         
    _________________________________________________________________
    batch_normalization_13 (Batc (None, 22, 22, 32)        128       
    _________________________________________________________________
    layer_conv4 (Conv2D)         (None, 22, 22, 64)        18496     
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 22, 22, 64)        0         
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 22, 22, 64)        256       
    _________________________________________________________________
    flatten_5 (Flatten)          (None, 30976)             0         
    _________________________________________________________________
    dense_10 (Dense)             (None, 1024)              31720448  
    _________________________________________________________________
    dense_11 (Dense)             (None, 96)                98400     
    =================================================================
    Total params: 31,875,296
    Trainable params: 31,874,976
    Non-trainable params: 320
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model6.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model6line/{}'.format(time()))
model6.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 71s 1ms/step - loss: 1.2924 - acc: 0.6273 - val_loss: 0.4313 - val_acc: 0.8542
    Epoch 2/3
    57600/57600 [==============================] - 68s 1ms/step - loss: 0.2652 - acc: 0.9073 - val_loss: 0.2182 - val_acc: 0.9248
    Epoch 3/3
    57600/57600 [==============================] - 69s 1ms/step - loss: 0.2123 - acc: 0.9322 - val_loss: 0.2132 - val_acc: 0.9285





    <tensorflow.python.keras.callbacks.History at 0x7efca88a1ba8>




```python
result = model6.evaluate(x=x_test,y=y_test)
for name, value in zip(model6.metrics_names, result):
    print(name, value)
model6.save("Task12_line6.h5")
```

    38400/38400 [==============================] - 10s 253us/step
    loss 0.21324227306390336
    acc 0.928515625



```python
model7 = Def_Model7()
model7.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 24, 24, 128)       9728      
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 24, 24, 128)       0         
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 24, 24, 128)       512       
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 22, 22, 128)       147584    
    _________________________________________________________________
    dropout_8 (Dropout)          (None, 22, 22, 128)       0         
    _________________________________________________________________
    batch_normalization_16 (Batc (None, 22, 22, 128)       512       
    _________________________________________________________________
    layer_conv3 (Conv2D)         (None, 22, 22, 64)        204864    
    _________________________________________________________________
    dropout_9 (Dropout)          (None, 22, 22, 64)        0         
    _________________________________________________________________
    batch_normalization_17 (Batc (None, 22, 22, 64)        256       
    _________________________________________________________________
    layer_conv4 (Conv2D)         (None, 22, 22, 32)        18464     
    _________________________________________________________________
    dropout_10 (Dropout)         (None, 22, 22, 32)        0         
    _________________________________________________________________
    batch_normalization_18 (Batc (None, 22, 22, 32)        128       
    _________________________________________________________________
    flatten_6 (Flatten)          (None, 15488)             0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 1024)              15860736  
    _________________________________________________________________
    dense_13 (Dense)             (None, 96)                98400     
    =================================================================
    Total params: 16,341,184
    Trainable params: 16,340,480
    Non-trainable params: 704
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-3)
model7.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model7line/{}'.format(time()))
model7.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 57600 samples, validate on 38400 samples
    Epoch 1/3
    57600/57600 [==============================] - 94s 2ms/step - loss: 1.1558 - acc: 0.6605 - val_loss: 1.7994 - val_acc: 0.6307
    Epoch 2/3
    57600/57600 [==============================] - 91s 2ms/step - loss: 0.2044 - acc: 0.9299 - val_loss: 0.1693 - val_acc: 0.9454
    Epoch 3/3
    57600/57600 [==============================] - 93s 2ms/step - loss: 0.1657 - acc: 0.9484 - val_loss: 0.8836 - val_acc: 0.7898





    <tensorflow.python.keras.callbacks.History at 0x7efc7d29ed30>




```python
result = model7.evaluate(x=x_test,y=y_test)
for name, value in zip(model7.metrics_names, result):
    print(name, value)
model7.save("Task12_line1.h5")
```

    38400/38400 [==============================] - 16s 415us/step
    loss 0.883619467181464
    acc 0.789765625



```python
def print_confusion_matrix(cm):
    recall=[]
    precision=[]
    recall_val = 0
    for i in range(len(cm)):
        num = cm[i][i]
        row_sum=cm[i].sum()
        recall_val = (1.0*num/row_sum)
        recall.append(recall_val);
        precision_val = (1.0*cm[i][i]/cm[:,i].sum())
        precision.append(precision_val)
    f = []
    f.append(np.array(recall))
    f.append(np.array(precision))
    f_score=[]
    for i in range(len(recall)):
        val = 2.0 * recall[i] * precision[i]
        val /= (precision[i]+recall[i])
        f_score.append(val)
    f.append(np.array(f_score))
    print(np.array(f))
    plt.figure(figsize=(40,40), dpi=200)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig('3.png')
```


```python
# Predicting the Test set results
y_pred = model1.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[1.         0.98469388 1.         0.99228792 0.95652174 0.98687664
      0.99270073 0.91168831 1.         1.         0.98449612 0.97727273
      0.96428571 0.99741602 0.98714653 0.9921671  1.         1.
      0.82352941 0.7357513  0.69458128 0.91911765 0.99480519 1.
      0.98511166 0.99244332 0.83028721 0.99228792 0.8694517  0.96610169
      0.96287129 0.99507389 1.         0.9562212  1.         1.
      1.         1.         0.94847775 0.995      1.         1.
      1.         1.         1.         0.95011876 0.81055156 0.99748744
      0.703125   0.95226131 0.78337531 0.91022444 1.         1.
      0.99294118 0.99202128 0.6870229  1.         1.         0.97074468
      0.995      0.95771144 0.64       0.99240506 0.99255583 1.
      0.95771144 0.93908629 0.96836983 1.         0.99457995 0.89260143
      0.94074074 1.         1.         0.95734597 0.99012346 0.96675192
      1.         0.94763092 0.97323601 1.         0.93866667 0.9973822
      1.         0.87468031 0.815      0.98232323 1.         0.94559585
      0.98321343 0.94358974 1.         0.87848101 0.95026178 1.        ]
     [1.         0.87727273 1.         0.97721519 1.         1.
      1.         0.99152542 0.99278846 1.         0.98704663 1.
      0.984375   0.75984252 0.9721519  1.         0.99749373 0.99491094
      0.94769231 0.98954704 0.96575342 0.99734043 0.95989975 0.94964029
      0.94075829 0.96805897 0.6989011  1.         1.         0.98275862
      1.         0.97584541 0.98611111 1.         0.78326996 1.
      0.95720721 0.88741722 1.         0.90249433 0.96766169 1.
      0.94089835 1.         1.         0.95923261 0.99120235 1.
      0.95744681 1.         0.99361022 0.86084906 1.         0.9882904
      1.         0.9973262  1.         1.         1.         1.
      1.         1.         0.86349206 0.96551724 1.         0.87363834
      0.9625     0.85846868 0.83613445 1.         1.         1.
      0.97692308 1.         0.9921875  0.99507389 0.92824074 0.97674419
      0.99515738 0.97938144 1.         0.94835681 0.57799672 0.94306931
      0.99496222 0.82808717 0.93142857 1.         0.89240506 0.95052083
      0.9512761  0.78969957 1.         0.98860399 1.         1.        ]
     [1.         0.92788462 1.         0.98469388 0.97777778 0.99339498
      0.996337   0.94993234 0.99638118 1.         0.98576973 0.98850575
      0.9742268  0.86256983 0.97959184 0.99606815 0.99874529 0.99744898
      0.88125894 0.84398217 0.80802292 0.95663265 0.97704082 0.97416974
      0.96242424 0.9800995  0.75894988 0.99612903 0.9301676  0.97435897
      0.98108449 0.98536585 0.99300699 0.97762073 0.87846482 1.
      0.97813579 0.94035088 0.97355769 0.94649227 0.98356511 1.
      0.96954933 1.         1.         0.95465394 0.89182058 0.99874214
      0.81081081 0.97554698 0.87605634 0.88484848 1.         0.99411072
      0.99645809 0.99466667 0.81447964 1.         1.         0.9851552
      0.99749373 0.97839898 0.73513514 0.97877653 0.99626401 0.93255814
      0.96009975 0.8969697  0.89740699 1.         0.99728261 0.94325347
      0.95849057 1.         0.99607843 0.97584541 0.95818399 0.97172237
      0.99757282 0.96324461 0.9864365  0.97349398 0.71544715 0.96946565
      0.99747475 0.85074627 0.86933333 0.9910828  0.94314381 0.94805195
      0.96698113 0.85981308 1.         0.93029491 0.97449664 1.        ]]



![png](/images/Task12_Line_files/Task12_Line_32_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model2.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[1.         0.96683673 1.         0.98200514 1.         1.
      0.98540146 0.98441558 1.         1.         1.         0.99747475
      0.92602041 0.80620155 0.99485861 0.98172324 1.         1.
      0.94919786 0.94559585 0.8546798  0.9877451  0.98961039 1.
      1.         0.98488665 0.95561358 1.         1.         1.
      1.         0.96551724 0.98356808 0.97004608 0.96359223 1.
      1.         0.82587065 1.         1.         0.93573265 1.
      1.         1.         1.         0.97149644 0.96402878 0.9798995
      0.91666667 1.         0.97481108 0.95511222 1.         1.
      0.99529412 0.9787234  1.         1.         0.88516746 0.98404255
      0.9975     1.         0.85882353 0.97974684 1.         0.98254364
      0.97512438 0.97208122 0.94647202 0.99061033 1.         0.98329356
      1.         1.         1.         0.992891   0.99506173 0.35294118
      0.99270073 1.         0.99513382 0.93564356 0.77066667 0.9921466
      0.98481013 0.98209719 0.9225     0.84343434 0.97399527 0.9119171
      0.99280576 0.97948718 0.9925     0.94683544 0.92931937 1.        ]
     [0.99270073 0.97179487 0.98325359 0.9870801  0.91569087 1.
      0.99022005 0.8536036  1.         1.         0.94160584 0.9875
      0.98108108 0.975      0.99485861 0.98429319 1.         1.
      0.58580858 0.96560847 0.98300283 0.97815534 0.98449612 0.99748111
      0.98533007 0.9354067  0.90147783 1.         0.96962025 0.99758454
      0.92448513 1.         0.98820755 1.         1.         0.97831325
      1.         0.97647059 0.9816092  0.97323601 1.         1.
      0.99749373 1.         1.         1.         0.95035461 0.97256858
      0.95392954 1.         0.96029777 0.77217742 1.         0.98598131
      1.         1.         0.9751861  0.99238579 0.99730458 1.
      0.9975     1.         0.97333333 0.95085995 1.         0.99494949
      0.98989899 0.98966408 1.         1.         1.         0.91150442
      1.         1.         1.         1.         1.         1.
      0.95327103 0.9804401  1.         1.         0.81869688 1.
      0.9239905  0.82580645 0.86013986 1.         0.99516908 0.97237569
      0.99759036 0.93170732 0.90432802 0.94444444 0.94164456 1.        ]
     [0.996337   0.96930946 0.99155609 0.98453608 0.95599022 1.
      0.98780488 0.91435464 1.         1.         0.96992481 0.99246231
      0.95275591 0.88260255 0.99485861 0.98300654 1.         1.
      0.7244898  0.95549738 0.914361   0.98292683 0.98704663 0.99873897
      0.99261084 0.9595092  0.92775665 1.         0.98457584 0.99879081
      0.960761   0.98245614 0.98588235 0.98479532 0.98145859 0.98903776
      1.         0.89487871 0.99071926 0.9864365  0.96679947 1.
      0.99874529 1.         1.         0.98554217 0.95714286 0.97622028
      0.93492696 1.         0.9675     0.85395764 1.         0.99294118
      0.99764151 0.98924731 0.98743719 0.99617834 0.93789607 0.9919571
      0.9975     1.         0.9125     0.96508728 1.         0.98870765
      0.98245614 0.98079385 0.9725     0.99528302 1.         0.94603904
      1.         1.         1.         0.99643282 0.99752475 0.52173913
      0.97258641 0.99012346 0.99756098 0.96675192 0.79395604 0.99605782
      0.95343137 0.89719626 0.89022919 0.91506849 0.98446834 0.94117647
      0.99519231 0.955      0.94636472 0.94563843 0.93544137 1.        ]]



![png](/images/Task12_Line_files/Task12_Line_33_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model3.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    /home/dord/.local/lib/python3.6/site-packages/ipykernel_launcher.py:10: RuntimeWarning: invalid value encountered in double_scalars
      # Remove the CWD from sys.path while we load stuff.


    [[0.96813725 0.98214286 0.96107056 0.96915167 0.99232737 1.
      1.         1.         1.         1.         1.         0.97222222
      0.83928571 1.         1.         0.93211488 1.         1.
      0.82352941 0.88082902 0.9729064  0.89215686 0.97402597 1.
      0.96774194 0.71284635 0.97389034 1.         0.97911227 1.
      0.92326733 0.84236453 0.95305164 0.99539171 1.         0.94334975
      0.99764706 1.         1.         0.         1.         1.
      1.         0.91885442 1.         0.99287411 1.         0.98743719
      0.9296875  0.98492462 0.93198992 0.99750623 1.         0.98815166
      0.98352941 0.8537234  1.         0.91815857 1.         1.
      0.8775     0.         0.99058824 1.         1.         1.
      0.97512438 1.         0.99756691 0.97183099 1.         0.91646778
      1.         1.         0.99212598 0.61374408 0.94074074 0.93350384
      0.36982968 0.9925187  1.         1.         0.96533333 0.9921466
      0.97468354 0.95140665 0.9775     0.88888889 0.         0.98704663
      0.99280576 0.87692308 0.605      0.95443038 0.9947644  0.99      ]
     [1.         0.97222222 0.97772277 0.90191388 1.         1.
      1.         0.74468085 1.         1.         1.         0.99483204
      1.         0.89583333 0.94188862 0.96226415 1.         0.99744898
      0.91394659 0.75892857 0.99246231 0.98913043 0.99469496 0.9924812
      0.98236776 0.99298246 0.91646192 1.         0.99734043 1.
      0.93483709 1.         1.         0.96644295 1.         0.88657407
      1.         0.98771499 1.                nan 0.58939394 0.92982456
      1.         1.         1.         1.         0.99522673 1.
      0.98347107 1.         1.         0.96618357 0.9468599  0.98349057
      0.98584906 0.55344828 1.         1.         1.         1.
      0.99715909        nan 0.9376392  1.         1.         0.57947977
      0.68411867 1.         0.98557692 0.98337292 1.         0.9974026
      1.         1.         0.984375   0.96282528 1.         1.
      0.72380952 1.         1.         0.43208556 0.99178082 0.88139535
      1.         0.83595506 1.         0.6875            nan 0.97193878
      0.69579832 0.82808717 0.89962825 0.98177083 0.83333333 1.        ]
     [0.98381071 0.97715736 0.96932515 0.93432466 0.99614891 1.
      1.         0.85365854 1.         1.         1.         0.98339719
      0.91262136 0.94505495 0.97007481 0.9469496  1.         0.99872286
      0.86638537 0.81534772 0.98258706 0.93814433 0.98425197 0.99622642
      0.975      0.82991202 0.9443038  1.         0.98814229 1.
      0.92901619 0.9144385  0.97596154 0.98070375 1.         0.91408115
      0.99882214 0.99381953 1.                nan 0.74165872 0.96363636
      1.         0.95771144 1.         0.99642431 0.99760766 0.99367889
      0.95582329 0.99240506 0.96479791 0.98159509 0.97270471 0.9858156
      0.98468787 0.67154812 1.         0.95733333 1.         1.
      0.93351064        nan 0.96338673 1.         1.         0.73376029
      0.80410256 1.         0.99153567 0.97756789 1.         0.95522388
      1.         1.         0.98823529 0.74963821 0.96946565 0.96560847
      0.48953301 0.99624531 1.         0.6034354  0.97837838 0.93349754
      0.98717949 0.88995215 0.988622   0.7753304         nan 0.97943445
      0.81818182 0.85180573 0.72346786 0.96790757 0.90692124 0.99497487]]



![png](/images/Task12_Line_files/Task12_Line_34_2.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model4.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.99019608 0.8622449  1.         0.97943445 1.         1.
      1.         1.         1.         1.         1.         1.
      0.05867347 0.94315245 1.         1.         1.         1.
      0.97058824 0.98445596 0.95320197 0.94362745 1.         1.
      1.         1.         0.97911227 1.         1.         1.
      1.         1.         1.         1.         1.         1.
      1.         1.         1.         1.         1.         1.
      0.94723618 1.         1.         0.94774347 1.         1.
      0.99739583 0.99497487 0.27959698 0.9925187  1.         1.
      0.99529412 1.         1.         1.         1.         1.
      1.         1.         0.92705882 1.         1.         1.
      0.99253731 1.         1.         0.96478873 1.         0.99761337
      1.         1.         0.99737533 1.         1.         1.
      1.         1.         1.         0.97772277 1.         1.
      1.         0.96675192 1.         1.         1.         1.
      1.         1.         1.         1.         0.98167539 1.        ]
     [1.         1.         1.         1.         1.         1.
      1.         1.         1.         1.         1.         0.96350365
      1.         1.         1.         1.         1.         1.
      0.96031746 1.         0.87755102 0.97964377 1.         1.
      1.         0.96829268 0.99734043 1.         1.         1.
      1.         1.         1.         1.         1.         1.
      1.         1.         1.         0.93023256 1.         1.
      1.         1.         1.         1.         1.         1.
      0.50930851 1.         0.96521739 0.9925187  1.         0.99528302
      1.         1.         1.         1.         1.         1.
      1.         1.         0.57101449 1.         1.         1.
      1.         0.97044335 1.         1.         1.         0.9952381
      1.         1.         1.         1.         1.         0.97263682
      1.         1.         1.         1.         0.91019417 0.99220779
      1.         0.945      0.95465394 1.         1.         0.94146341
      0.99760766 1.         1.         1.         1.         1.        ]
     [0.99507389 0.9260274  1.         0.98961039 1.         1.
      1.         1.         1.         1.         1.         0.98141264
      0.11084337 0.97074468 1.         1.         1.         1.
      0.96542553 0.9921671  0.91381346 0.96129838 1.         1.
      1.         0.98389095 0.98814229 1.         1.         1.
      1.         1.         1.         1.         1.         1.
      1.         1.         1.         0.96385542 1.         1.
      0.97290323 1.         1.         0.97317073 1.         1.
      0.67429577 0.99748111 0.43359375 0.9925187  1.         0.99763593
      0.99764151 1.         1.         1.         1.         1.
      1.         1.         0.70672646 1.         1.         1.
      0.99625468 0.985      1.         0.98207885 1.         0.99642431
      1.         1.         0.99868594 1.         1.         0.98612863
      1.         1.         1.         0.98873592 0.95298602 0.99608866
      1.         0.95575221 0.97680098 1.         1.         0.96984925
      0.9988024  1.         1.         1.         0.99075297 1.        ]]



![png](/images/Task12_Line_files/Task12_Line_35_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model5.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    /home/dord/.local/lib/python3.6/site-packages/ipykernel_launcher.py:18: RuntimeWarning: invalid value encountered in double_scalars


    [[0.91176471 0.00255102 0.99270073 1.         0.81074169 0.97637795
      0.59854015 0.29350649 1.         0.43291139 0.99483204 0.0530303
      0.79591837 0.01550388 1.         0.84073107 0.98994975 0.95140665
      0.21122995 0.88341969 0.16256158 1.         1.         1.
      0.82630273 1.         0.02349869 1.         0.92689295 1.
      1.         0.98522167 0.         0.79032258 0.30097087 1.
      0.73647059 0.97014925 0.99297424 1.         0.99485861 1.
      1.         0.93556086 1.         0.98574822 0.46522782 0.33165829
      0.92447917 1.         0.02518892 0.12468828 1.         0.1563981
      0.91058824 0.57180851 1.         0.99744246 0.99521531 1.
      0.975      1.         0.07058824 0.14177215 1.         0.90274314
      0.91044776 0.88071066 0.02189781 0.8685446  0.95663957 0.30548926
      1.         1.         0.8503937  0.96445498 0.41728395 0.30690537
      0.89294404 1.         1.         0.90594059 0.28266667 0.65445026
      0.28101266 0.67007673 0.045      0.64646465 0.9929078  0.39896373
      0.32134293 0.38205128 1.         0.05063291 0.95811518 1.        ]
     [0.80869565 0.01315789 0.44884488 0.7627451  0.68763557 1.
      0.9389313  1.         0.97867299 1.         0.82264957 0.1875
      0.43636364 1.         1.         0.44475138 1.         1.
      1.         0.69591837 0.72527473 0.62672811 0.835141   1.
      1.         0.55369596 0.0505618  1.         0.89873418 0.31867284
      0.66556837 0.98039216 0.         0.41982864 1.         0.90625
      1.         1.         0.46799117 1.         1.         1.
      1.         0.99745547 1.         0.56309362 0.67361111 1.
      0.5035461  0.84322034 1.         0.98039216 0.83760684 0.27848101
      0.54661017 0.92672414 0.35597826 1.         1.         0.98429319
      0.57863501 1.         0.96774194 0.5        0.98292683 1.
      0.99456522 0.9747191  1.         0.49333333 0.9943662  0.19814241
      1.         0.9538835  1.         0.99268293 1.         1.
      0.74291498 1.         0.99515738 1.         1.         1.
      0.44758065 0.99242424 1.         0.56140351 0.66037736 0.15876289
      1.         0.54379562 0.66006601 0.08474576 0.57366771 1.        ]
     [0.85714286 0.0042735  0.61818182 0.86540601 0.74413146 0.98804781
      0.73105498 0.45381526 0.98922156 0.60424028 0.9005848  0.08267717
      0.56368564 0.03053435 1.         0.58175248 0.99494949 0.9750983
      0.34878587 0.77853881 0.26559356 0.77053824 0.91016548 1.
      0.9048913  0.71274686 0.03208556 1.         0.9125964  0.48332358
      0.7992087  0.98280098        nan 0.54836131 0.46268657 0.95081967
      0.84823848 0.98484848 0.63615904 1.         0.99742268 1.
      1.         0.96551724 1.         0.71675302 0.55035461 0.49811321
      0.65197429 0.91494253 0.04914005 0.22123894 0.91162791 0.20030349
      0.6831421  0.70723684 0.5250501  0.99871959 0.99760192 0.99208443
      0.72625698 1.         0.13157895 0.2209073  0.99138991 0.94888598
      0.95064935 0.92533333 0.04285714 0.6292517  0.97513812 0.24037559
      1.         0.97639752 0.91914894 0.97836538 0.58885017 0.46966732
      0.81104972 1.         0.99757282 0.95064935 0.44074844 0.79113924
      0.34525661 0.8        0.0861244  0.60093897 0.79320113 0.22713864
      0.48638838 0.44879518 0.79522863 0.06339144 0.71764706 1.        ]]



![png](/images/Task12_Line_files/Task12_Line_36_2.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model6.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[1.         0.96683673 1.         0.85861183 0.99488491 1.
      1.         0.7038961  1.         0.99746835 1.         0.74494949
      0.93877551 0.84754522 0.96915167 0.9843342  0.98241206 0.98209719
      0.90641711 0.98704663 0.83497537 0.71813725 1.         0.95454545
      1.         0.85390428 0.15665796 1.         1.         0.98547215
      0.98019802 0.93349754 0.97183099 0.96774194 1.         0.98768473
      0.99529412 0.84825871 1.         1.         1.         1.
      1.         1.         1.         1.         0.93045564 1.
      0.9296875  1.         0.67758186 0.42643392 1.         0.79383886
      0.96941176 0.98670213 0.99491094 1.         1.         1.
      1.         1.         0.94588235 0.97974684 0.99751861 0.69077307
      0.9278607  0.86294416 0.93187348 0.57981221 1.         0.88305489
      1.         1.         0.94750656 0.95260664 1.         0.98721228
      0.99026764 1.         1.         1.         0.45866667 0.90575916
      0.95443038 0.99232737 0.5225     1.         0.98817967 0.98445596
      0.97841727 0.86923077 1.         0.99746835 1.         1.        ]
     [1.         0.84598214 0.95359629 0.73085339 0.95110024 1.
      1.         0.82121212 1.         1.         0.88154897 0.96405229
      0.92929293 0.99696049 1.         0.76938776 1.         1.
      0.98546512 0.86986301 0.63602251 0.99659864 0.95533499 1.
      1.         1.         0.52173913 1.         0.99222798 0.99754902
      0.99748111 1.         1.         0.98360656 0.76579926 0.98284314
      0.93584071 1.         0.995338   1.         1.         1.
      1.         0.99054374 1.         1.         1.         1.
      0.85611511 1.         0.92123288 0.47107438 1.         1.
      0.78625954 0.98408488 1.         1.         0.95652174 1.
      1.         1.         0.97336562 1.         1.         0.9822695
      1.         0.99415205 0.99739583 1.         1.         0.47254151
      1.         1.         0.95755968 0.94145199 0.97122302 0.75097276
      0.98786408 0.99257426 1.         1.         1.         1.
      0.83039648 0.86222222 0.476082   0.90410959 0.78571429 0.87759815
      0.89082969 0.99705882 1.         0.90160183 0.93627451 0.98039216]
     [1.         0.90238095 0.97624703 0.78959811 0.9725     1.
      1.         0.75804196 1.         0.99873257 0.937046   0.84045584
      0.93401015 0.91620112 0.9843342  0.86368843 0.99112801 0.99096774
      0.94428969 0.92475728 0.72204473 0.83475783 0.97715736 0.97674419
      1.         0.92119565 0.24096386 1.         0.99609883 0.99147381
      0.98876404 0.9656051  0.98571429 0.97560976 0.86736842 0.98525799
      0.96465222 0.9179004  0.99766355 1.         1.         1.
      1.         0.99524941 1.         1.         0.96397516 1.
      0.89138577 1.         0.7808418  0.44764398 1.         0.88507266
      0.8682824  0.98539177 0.99744898 1.         0.97777778 1.
      1.         1.         0.95942721 0.98976982 0.99875776 0.81112738
      0.96258065 0.92391304 0.96352201 0.73402675 1.         0.6156406
      1.         1.         0.9525066  0.94699647 0.98540146 0.85303867
      0.9890644  0.99627329 1.         1.         0.62888483 0.95054945
      0.88810365 0.92271106 0.49821216 0.94964029 0.87539267 0.92796093
      0.93257143 0.92876712 1.         0.94711538 0.96708861 0.99009901]]



![png](/images/Task12_Line_files/Task12_Line_37_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model7.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    /home/dord/.local/lib/python3.6/site-packages/ipykernel_launcher.py:18: RuntimeWarning: invalid value encountered in double_scalars


    [[0.9877451  0.83163265 0.44768856 0.62210797 0.14322251 0.98687664
      0.9756691  1.         0.92009685 0.96708861 0.9870801  0.97727273
      0.85714286 1.         1.         0.54830287 0.99748744 0.95907928
      0.86631016 0.40414508 0.80541872 0.63235294 0.         1.
      0.5235732  0.58186398 0.90600522 0.86118252 0.58746736 0.99031477
      0.56188119 0.45073892 0.41079812 0.16820276 1.         0.20689655
      1.         1.         0.92740047 1.         1.         0.97169811
      0.90703518 1.         0.83248731 0.93824228 0.82973621 0.92211055
      0.859375   0.98241206 0.99244332 0.42394015 0.96938776 0.79146919
      0.20470588 0.83244681 0.92875318 0.98976982 1.         0.85638298
      0.74       1.         0.80941176 1.         0.86848635 0.97755611
      0.31343284 0.97715736 0.9270073  0.56807512 1.         0.54892601
      0.60740741 0.98727735 1.         0.08056872 0.97530864 0.97186701
      0.38442822 0.89276808 0.99756691 1.         0.99466667 0.87696335
      0.61772152 0.60613811 0.8625     0.77272727 0.9929078  0.83678756
      0.92805755 0.46923077 0.645      0.68860759 0.72251309 1.        ]
     [1.         0.81094527 1.         1.         1.         0.88470588
      0.88520971 1.         1.         0.99738903 1.         0.37283237
      0.82151589 0.41568206 0.70471014 0.72916667 1.         0.92137592
      0.77697842 0.70909091 1.         1.         0.         0.76007678
      0.98598131 1.         0.61524823 0.98529412 1.         1.
      0.40248227 0.83944954 0.97765363 0.25347222 0.99757869 0.6
      0.94026549 1.         1.         0.98280098 0.92619048 0.99038462
      0.97043011 1.         1.         1.         0.52424242 0.97606383
      0.96491228 1.         0.76953125 0.77981651 0.99737533 0.53100159
      0.87878788 0.48527132 1.         0.99742268 0.88559322 1.
      0.996633   0.98529412 0.98567335 0.9875     0.96685083 1.
      0.22183099 0.98214286 0.70686456 0.63020833 0.93654822 0.73248408
      0.86925795 0.97487437 0.96946565 0.07375271 0.99246231 0.89411765
      0.44886364 0.65808824 1.         0.6196319  0.5846395  0.49925484
      1.         0.74528302 0.82932692 0.73205742 0.9976247  0.69017094
      0.61234177 0.46683673 0.59310345 0.56548857 0.99638989 1.        ]
     [0.99383477 0.82115869 0.61848739 0.76703645 0.25055928 0.93300248
      0.92824074 1.         0.95838588 0.98200514 0.99349805 0.53974895
      0.83895131 0.58725341 0.82678002 0.62593145 0.99874214 0.93984962
      0.81921618 0.51485149 0.89222374 0.77477477        nan 0.86368593
      0.68395462 0.73566879 0.73284055 0.91906722 0.74013158 0.99513382
      0.46900826 0.58653846 0.5785124  0.20221607 0.99878788 0.30769231
      0.96921323 1.         0.96233293 0.9913259  0.96168109 0.98095238
      0.93766234 1.         0.90858726 0.96813725 0.64252553 0.94832041
      0.90909091 0.99112801 0.86688669 0.54927302 0.98318241 0.63558516
      0.33206107 0.61312439 0.96306069 0.99358151 0.93932584 0.9226361
      0.84935438 0.99259259 0.88888889 0.99371069 0.91503268 0.98865069
      0.25979381 0.97964377 0.80210526 0.59753086 0.9672346  0.62755798
      0.71511628 0.98103666 0.98449612 0.07701019 0.98381071 0.93137255
      0.41415465 0.75767196 0.99878197 0.76515152 0.73642646 0.6362773
      0.76369327 0.66854725 0.84558824 0.75184275 0.99526066 0.75644028
      0.73784557 0.46803069 0.61796407 0.62100457 0.83763278 1.        ]]



![png](/images/Task12_Line_files/Task12_Line_38_2.png)



    <Figure size 432x288 with 0 Axes>



```python
weights, biases = model3.layers[0].get_weights()
```


```python
weights.shape
```




    (5, 5, 3, 32)




```python
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
```


```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(deprocess_image(weights[:,:,:,filter_index]))
    filter_index += 1
```


![png](/images/Task12_Line_files/Task12_Line_42_0.png)



```python
weights, biases = model3.layers[2].get_weights()
```


```python
weights.shape
```




    (5, 5, 32, 32)




```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(deprocess_image(weights[:,:,0:3,filter_index]))
    filter_index += 1
```


![png](/images/Task12_Line_files/Task12_Line_45_0.png)



```python
layer_outputs = [layer.output for layer in model3.layers[:12]] 
# Extracts the outputs of the top 12 layers

activation_model = models.Model(inputs=model3.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
```


```python
imgplot = plt.imshow(x_test[6].reshape(28,28,3))
plt.show()
print(y_test[6])
```


![png](/images/Task12_Line_files/Task12_Line_47_0.png)


    [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]



```python
activations = activation_model.predict(x_test[6].reshape(1,28,28,3))
# Returns a list of five Numpy arrays: one array per layer activation
```


```python
first_layer_activation = activations[0]
print(first_layer_activation.shape)
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```

    (1, 24, 24, 32)



![png](/images/Task12_Line_files/Task12_Line_49_1.png)



```python
first_layer_activation = activations[1]
print(first_layer_activation.shape)
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index],)
    filter_index += 1
```

    (1, 24, 24, 32)



![png](/images/Task12_Line_files/Task12_Line_50_1.png)



```python
first_layer_activation = activations[2]
print(first_layer_activation.shape)
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```

    (1, 20, 20, 32)



![png](/images/Task12_Line_files/Task12_Line_51_1.png)



```python
print(model3.output)
flower_output = model3.output[1]
```

    Tensor("dense_1/Softmax:0", shape=(?, 96), dtype=float32)



```python
last_conv_layer = model3.get_layer('layer_conv2')
grads = K.gradients(flower_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model3.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#2048 is the number of filters/channels in 'mixed10' layer
for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
print(heatmap.shape)
heatmap=heatmap.reshape((11,11))
plt.imshow(heatmap)
```
