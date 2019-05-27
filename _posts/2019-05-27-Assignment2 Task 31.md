---
title: "Assginment 2 Task 3 Part 1 "
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
import random
from keras.preprocessing import image
```


```python
from tensorflow.python.keras import models
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D,Dropout
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,BatchNormalization
from tensorflow.python.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.datasets import mnist
import keras.backend as K
from keras import backend as K
```

    Using TensorFlow backend.



```python
tf.__version__
```




    '1.12.0'




```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```


```python
x = np.concatenate((x_train,x_test))
y_class = np.concatenate((y_train,y_test))
```


```python
num_classes = 10
img_size_flat = 28*28
img_shape_full = (28,28)
TotalSize = 70000
```


```python
x_train, x_test, y_train_cls, y_test_cls = train_test_split(x.reshape(TotalSize,28,28,1), y_class, test_size=0.40, random_state=42)
train_size = y_train_cls.shape[0]
test_size = y_test_cls.shape[0]
print(train_size,test_size)
```

    42000 28000



```python
b = np.zeros((train_size, num_classes))
b[np.arange(train_size), y_train_cls] = 1
y_train = b
```


```python
b = np.zeros((test_size, num_classes))
b[np.arange(test_size), y_test_cls] = 1
y_test = b
```


```python
x_train[0].shape
```




    (28, 28, 1)




```python
imgplot = plt.imshow(x_train[0].reshape(28,28))
plt.show()
print(y_train[5])
```


![png](/images/Task12_MNIST_files/Task12_MNIST_10_0.png)


    [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]



```python
def Def_Model1():
    model = Sequential()

    model.add(InputLayer(input_shape=(28,28,1,)))
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

    model.add(InputLayer(input_shape=(28,28,1,)))

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

    model.add(InputLayer(input_shape=(28,28,1,)))

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

    model.add(InputLayer(input_shape=(28,28,1,)))

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

    model.add(InputLayer(input_shape=(28,28,1,)))

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

    model.add(InputLayer(input_shape=(28,28,1,)))

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

    model.add(InputLayer(input_shape=(28,28,1,)))

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
    layer_conv1 (Conv2D)         (None, 28, 28, 32)        832       
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
    dense_1 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 6,453,258
    Trainable params: 6,453,194
    Non-trainable params: 64
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model1.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model1minist/{}'.format(time()))
model1.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3
    42000/42000 [==============================] - 29s 696us/step - loss: 0.1726 - acc: 0.9467 - val_loss: 0.0767 - val_acc: 0.9767
    Epoch 2/3
    42000/42000 [==============================] - 17s 404us/step - loss: 0.0394 - acc: 0.9884 - val_loss: 0.0700 - val_acc: 0.9796
    Epoch 3/3
    42000/42000 [==============================] - 17s 404us/step - loss: 0.0171 - acc: 0.9950 - val_loss: 0.0486 - val_acc: 0.9854





    <tensorflow.python.keras.callbacks.History at 0x7f6e14bb7b70>




```python
result = model1.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model1.metrics_names, result):
    print(name, value)
model1.save("Task12_minist1.h5")   
```

    28000/28000 [==============================] - 3s 124us/step
    loss 0.048638307643405695
    acc 0.9853571428571428



```python
model2 = Def_Model2()
model2.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 22, 22, 16)        800       
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 22, 22, 16)        64        
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 11, 11, 16)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 1936)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1024)              1983488   
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 1,994,602
    Trainable params: 1,994,570
    Non-trainable params: 32
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model2.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model2minist/{}'.format(time()))
model2.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3
    42000/42000 [==============================] - 5s 111us/step - loss: 0.2014 - acc: 0.9379 - val_loss: 0.0893 - val_acc: 0.9747
    Epoch 2/3
    42000/42000 [==============================] - 4s 103us/step - loss: 0.0581 - acc: 0.9832 - val_loss: 0.0672 - val_acc: 0.9794
    Epoch 3/3
    42000/42000 [==============================] - 4s 103us/step - loss: 0.0344 - acc: 0.9905 - val_loss: 0.0568 - val_acc: 0.9827





    <tensorflow.python.keras.callbacks.History at 0x7f6e143c4e10>




```python
result = model2.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model2.metrics_names, result):
    print(name, value)
model2.save("Task12_minist2.h5")   
```

    28000/28000 [==============================] - 1s 39us/step
    loss 0.05677692559354806
    acc 0.9826785714285714



```python
model3 = Def_Model3()
model3.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 24, 24, 32)        832       
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 24, 24, 32)        128       
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 20, 20, 32)        25632     
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 20, 20, 32)        128       
    _________________________________________________________________
    flatten_2 (Flatten)          (None, 12800)             0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1024)              13108224  
    _________________________________________________________________
    dense_5 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 13,145,194
    Trainable params: 13,145,066
    Non-trainable params: 128
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model3.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model3minist/{}'.format(time()))
model3.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3
    42000/42000 [==============================] - 20s 465us/step - loss: 0.1483 - acc: 0.9549 - val_loss: 0.0697 - val_acc: 0.9787
    Epoch 2/3
    42000/42000 [==============================] - 20s 466us/step - loss: 0.0175 - acc: 0.9948 - val_loss: 0.0695 - val_acc: 0.9792
    Epoch 3/3
    42000/42000 [==============================] - 19s 444us/step - loss: 0.0053 - acc: 0.9988 - val_loss: 0.0582 - val_acc: 0.9836





    <tensorflow.python.keras.callbacks.History at 0x7f6dd067df98>




```python
result = model3.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model3.metrics_names, result):
    print(name, value)
model3.save("Task12_minist3.h5")   
```

    28000/28000 [==============================] - 3s 122us/step
    loss 0.058238421496308417
    acc 0.9836428571428572



```python
model4 = Def_Model4()
model4.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 28, 28, 32)        832       
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
    dense_7 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 171,434
    Trainable params: 171,178
    Non-trainable params: 256
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model4.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model4minist/{}'.format(time()))
model4.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3
    42000/42000 [==============================] - 10s 239us/step - loss: 0.4484 - acc: 0.8862 - val_loss: 0.1480 - val_acc: 0.9701
    Epoch 2/3
    42000/42000 [==============================] - 9s 212us/step - loss: 0.0696 - acc: 0.9795 - val_loss: 0.0694 - val_acc: 0.9785
    Epoch 3/3
    42000/42000 [==============================] - 9s 212us/step - loss: 0.0397 - acc: 0.9889 - val_loss: 0.0567 - val_acc: 0.9820





    <tensorflow.python.keras.callbacks.History at 0x7f6d9a5cba20>




```python
result = model4.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model4.metrics_names, result):
    print(name, value)
model4.save("Task12_minist4.h5")   
```

    28000/28000 [==============================] - 2s 86us/step
    loss 0.0566943829646334
    acc 0.9819642857142857



```python
model5 = Def_Model5()
model5.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 28, 28, 32)        832       
    _________________________________________________________________
    dropout (Dropout)            (None, 28, 28, 32)        0         
    _________________________________________________________________
    batch_normalization (BatchNo (None, 28, 28, 32)        128       
    _________________________________________________________________
    layer_conv2 (Conv2D)         (None, 28, 28, 32)        25632     
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 28, 28, 32)        0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 28, 28, 32)        128       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
    _________________________________________________________________
    layer_conv3 (Conv2D)         (None, 14, 14, 32)        25632     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 14, 14, 32)        0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 14, 14, 32)        128       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1568)              0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              1606656   
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                10250     
    =================================================================
    Total params: 1,669,386
    Trainable params: 1,669,194
    Non-trainable params: 192
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model5.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model5minist/{}'.format(time()))
model5.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3



    ---------------------------------------------------------------------------

    UnknownError                              Traceback (most recent call last)

    <ipython-input-17-5612c4cce4c5> in <module>
          8           epochs=3, batch_size=128,verbose=1,
          9           validation_data=(x_test, y_test),
    ---> 10           callbacks=[tensorboard])
    

    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/keras/engine/training.py in fit(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, max_queue_size, workers, use_multiprocessing, **kwargs)
       1637           initial_epoch=initial_epoch,
       1638           steps_per_epoch=steps_per_epoch,
    -> 1639           validation_steps=validation_steps)
       1640 
       1641   def evaluate(self,


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/keras/engine/training_arrays.py in fit_loop(model, inputs, targets, sample_weights, batch_size, epochs, verbose, callbacks, val_inputs, val_targets, val_sample_weights, shuffle, initial_epoch, steps_per_epoch, validation_steps)
        213           ins_batch[i] = ins_batch[i].toarray()
        214 
    --> 215         outs = f(ins_batch)
        216         if not isinstance(outs, list):
        217           outs = [outs]


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/keras/backend.py in __call__(self, inputs)
       2984 
       2985     fetched = self._callable_fn(*array_vals,
    -> 2986                                 run_metadata=self.run_metadata)
       2987     self._call_fetch_callbacks(fetched[-len(self._fetches):])
       2988     return fetched[:len(self.outputs)]


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/client/session.py in __call__(self, *args, **kwargs)
       1437           ret = tf_session.TF_SessionRunCallable(
       1438               self._session._session, self._handle, args, status,
    -> 1439               run_metadata_ptr)
       1440         if run_metadata:
       1441           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py in __exit__(self, type_arg, value_arg, traceback_arg)
        526             None, None,
        527             compat.as_text(c_api.TF_Message(self.status.status)),
    --> 528             c_api.TF_GetCode(self.status.status))
        529     # Delete the underlying status object from memory otherwise it stays alive
        530     # as there is a reference to status from this from the traceback due to


    UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
    	 [[{{node layer_conv1/Conv2D}} = Conv2D[T=DT_FLOAT, _class=["loc:@training/Adam/gradients/layer_conv1/Conv2D_grad/Conv2DBackpropFilter"], data_format="NCHW", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](training/Adam/gradients/layer_conv1/Conv2D_grad/Conv2DBackpropFilter-0-TransposeNHWCToNCHW-LayoutOptimizer, layer_conv1/Conv2D/ReadVariableOp)]]
    	 [[{{node loss/dense_1_loss/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/_179}} = _Recv[client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_758_l...pandDims_1", tensor_type=DT_INT32, _device="/job:localhost/replica:0/task:0/device:CPU:0"]()]]



```python
result = model5.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model5.metrics_names, result):
    print(name, value)
model5.save("Task12_minist5.h5")   
```


```python
model6 = Def_Model6()
model6.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 24, 24, 32)        832       
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
    dense_11 (Dense)             (None, 10)                10250     
    =================================================================
    Total params: 31,785,546
    Trainable params: 31,785,226
    Non-trainable params: 320
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model6.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model6minist/{}'.format(time()))
model6.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3
    42000/42000 [==============================] - 52s 1ms/step - loss: 0.2576 - acc: 0.9200 - val_loss: 0.1433 - val_acc: 0.9571
    Epoch 2/3
    42000/42000 [==============================] - 49s 1ms/step - loss: 0.0805 - acc: 0.9747 - val_loss: 0.0599 - val_acc: 0.9815
    Epoch 3/3
    42000/42000 [==============================] - 51s 1ms/step - loss: 0.0520 - acc: 0.9835 - val_loss: 0.0608 - val_acc: 0.9825





    <tensorflow.python.keras.callbacks.History at 0x7f6d9863c5f8>




```python
result = model6.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model6.metrics_names, result):
    print(name, value)
model6.save("Task12_minist6.h5")   
```

    28000/28000 [==============================] - 7s 244us/step
    loss 0.0607991006094214
    acc 0.9825357142857143



```python
model7 = Def_Model7()
model7.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    layer_conv1 (Conv2D)         (None, 24, 24, 128)       3328      
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
    dense_13 (Dense)             (None, 10)                10250     
    =================================================================
    Total params: 16,246,634
    Trainable params: 16,245,930
    Non-trainable params: 704
    _________________________________________________________________



```python
optimizer = Adam(lr=1e-4)
model7.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir='model7minist/{}'.format(time()))
model7.fit(x=x_train,
          y=y_train,
          epochs=3, batch_size=128,verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

    Train on 42000 samples, validate on 28000 samples
    Epoch 1/3
    42000/42000 [==============================] - 69s 2ms/step - loss: 0.2201 - acc: 0.9316 - val_loss: 0.0849 - val_acc: 0.9756
    Epoch 2/3
    42000/42000 [==============================] - 65s 2ms/step - loss: 0.0602 - acc: 0.9816 - val_loss: 0.0511 - val_acc: 0.9845
    Epoch 3/3
    42000/42000 [==============================] - 65s 2ms/step - loss: 0.0390 - acc: 0.9872 - val_loss: 0.0510 - val_acc: 0.9865





    <tensorflow.python.keras.callbacks.History at 0x7f6d7cf50d30>




```python
result = model7.evaluate(x=x_test,
                        y=y_test)
for name, value in zip(model7.metrics_names, result):
    print(name, value)
model7.save("Task12_minist7.h5")   
```

    28000/28000 [==============================] - 11s 387us/step
    loss 0.05101449726257748
    acc 0.9864642857142857



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
    plt.figure(figsize=(6,6), dpi=200)
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

    [[0.99172066 0.98955366 0.98788311 0.98232591 0.99169184 0.98742138
      0.99132321 0.97956449 0.97148218 0.98049645]
     [0.98816356 0.99332698 0.97985154 0.98721137 0.98204936 0.98355521
      0.98739647 0.99017948 0.98515982 0.97530864]
     [0.98993891 0.99143673 0.98385093 0.98476258 0.98684705 0.9854845
      0.98935594 0.98484338 0.97827319 0.97789567]]



![png](/images/Task12_MNIST_files/Task12_MNIST_34_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model2.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.99208063 0.98733777 0.98253742 0.98232591 0.98753776 0.9783805
      0.99240781 0.98458961 0.97223265 0.96631206]
     [0.98958707 0.99362854 0.97731301 0.98267327 0.98013493 0.98185404
      0.98634567 0.97382372 0.97995461 0.98056855]
     [0.99083228 0.99047317 0.97991825 0.98249956 0.98382242 0.9801142
      0.98936745 0.97917708 0.97607836 0.97338811]]



![png](/images/Task12_MNIST_files/Task12_MNIST_35_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model3.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.99208063 0.9920861  0.98503207 0.97490279 0.98074018 0.98034591
      0.98951555 0.9798995  0.97861163 0.98191489]
     [0.98781362 0.99020537 0.97840708 0.98782235 0.98858013 0.98343849
      0.98915793 0.98617667 0.97168405 0.97226124]
     [0.98994253 0.99114485 0.9817084  0.98132005 0.98464455 0.98188976
      0.98933671 0.98302806 0.97513554 0.97706422]]



![png](/images/Task12_MNIST_files/Task12_MNIST_36_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model4.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.99208063 0.98955366 0.97576622 0.97242842 0.97922961 0.98506289
      0.98987708 0.98291457 0.96622889 0.98510638]
     [0.98745969 0.99206601 0.982771   0.99135135 0.98518237 0.97509728
      0.98560115 0.9825854  0.98132622 0.95562436]
     [0.98976477 0.99080824 0.97925608 0.98179872 0.98219697 0.98005475
      0.98773449 0.98274996 0.97371904 0.97014144]]



![png](/images/Task12_MNIST_files/Task12_MNIST_37_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model5.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.9974802  0.99493511 0.95509622 0.97030753 0.93731118 0.98781447
      0.98987708 0.98291457 0.96585366 0.9893617 ]
     [0.96719023 0.9769972  0.99332839 0.99456522 0.99758842 0.98356164
      0.98737829 0.97930574 0.98281787 0.92109607]
     [0.98210172 0.98588457 0.97383721 0.98228663 0.9665109  0.98568347
      0.98862611 0.98110684 0.97426192 0.95400923]]



![png](/images/Task12_MNIST_files/Task12_MNIST_38_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model6.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.98524118 0.99746755 0.97968639 0.99328385 0.96487915 0.98073899
      0.99855387 0.98023451 0.9564728  0.98475177]
     [0.99382716 0.97735732 0.98991718 0.97569444 0.99726776 0.98929421
      0.97116737 0.98220879 0.99570313 0.95824707]
     [0.98951555 0.98731004 0.98477521 0.98441058 0.98080614 0.98499803
      0.98467023 0.98122066 0.97569378 0.97131864]]



![png](/images/Task12_MNIST_files/Task12_MNIST_39_1.png)



    <Figure size 432x288 with 0 Axes>



```python
# Predicting the Test set results
y_pred = model7.predict(x_test)
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print_confusion_matrix(cm)
```

    [[0.9949604  0.99936689 0.97790449 0.9688936  0.9913142  0.98781447
      0.99313087 0.98760469 0.97523452 0.98687943]
     [0.99103621 0.97468354 0.99528473 0.99890671 0.9916887  0.97365362
      0.99169675 0.98397864 0.9897182  0.97614872]
     [0.99299443 0.9868709  0.98651807 0.98367127 0.99150142 0.98068293
      0.99241329 0.98578833 0.98242298 0.98148475]]



![png](/images/Task12_MNIST_files/Task12_MNIST_40_1.png)



    <Figure size 432x288 with 0 Axes>



```python
weights, biases = model1.layers[0].get_weights()
```


```python
weights.shape
```




    (5, 5, 1, 32)




```python
t = weights.reshape(5,5,32)
```


```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(t[:,:,filter_index])
    filter_index += 1
```


![png](/images/Task12_MNIST_files/Task12_MNIST_44_0.png)



```python
layer_outputs = [layer.output for layer in model1.layers[:12]] 
# Extracts the outputs of the top 12 layers

activation_model = models.Model(inputs=model1.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
```


```python
imgplot = plt.imshow(x_test[6].reshape(28,28))
plt.show()
print(y_test[6])
```


![png](/images/Task12_MNIST_files/Task12_MNIST_46_0.png)


    [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]



```python
activations = activation_model.predict(x_test[5].reshape(1,28,28,1))
# Returns a list of five Numpy arrays: one array per layer activation
```


```python
first_layer_activation = activations[1]
print(first_layer_activation.shape)
```

    (1, 28, 28, 32)



```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```


![png](/images/Task12_MNIST_files/Task12_MNIST_49_0.png)



```python
first_layer_activation = activations[0]
print(first_layer_activation.shape)
```

    (1, 28, 28, 32)



```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```


![png](/images/Task12_MNIST_files/Task12_MNIST_51_0.png)



```python
first_layer_activation = activations[2]
print(first_layer_activation.shape)
```

    (1, 28, 28, 32)



```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```


![png](/images/Task12_MNIST_files/Task12_MNIST_53_0.png)



```python
first_layer_activation = activations[3]
print(first_layer_activation.shape)
```

    (1, 28, 28, 32)



```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```


![png](/images/Task12_MNIST_files/Task12_MNIST_55_0.png)



```python
first_layer_activation = activations[4]
print(first_layer_activation.shape)
```

    (1, 14, 14, 32)



```python
col_size = 4
row_size = 8
filter_index = 0
fig, ax = plt.subplots(row_size, col_size, figsize=(20,20))
for row in range(0,row_size): 
  for col in range(0,col_size):
    ax[row][col].imshow(first_layer_activation[0, :, :, filter_index])
    filter_index += 1
```


![png](/images/Task12_MNIST_files/Task12_MNIST_57_0.png)



```python
print(model5.output)
flower_output = model5.output[1]
```

    Tensor("dense_1/Softmax:0", shape=(?, 10), dtype=float32)



```python
img = x_test[5]
print(img.shape)
img=img.reshape((28,28))
print(img.shape)
plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
print(x.shape)


preds = model5.predict(x)
#print ("Predicted: ", decode_predictions(preds, top=3)[0])

#985 is the class index for class 'Daisy' in Imagenet dataset on which my model is pre-trained
print(model5.output)
flower_output = model5.output[1]
last_conv_layer = model5.get_layer('max_pooling2d_1')

grads = K.gradients(flower_output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model5.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])

#2048 is the number of filters/channels in 'mixed10' layer
for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
print(heatmap.shape)
#     heatmap=heatmap.reshape((11,11))
plt.imshow(heatmap)
plt.savefig("Section2/Image"+str(j+1)+"/HeatmapForClass1")
```

    (28, 28, 1)
    (28, 28)
    (1, 28, 28, 1)



    ---------------------------------------------------------------------------

    UnknownError                              Traceback (most recent call last)

    <ipython-input-32-20494c68b4b1> in <module>
          9 
         10 
    ---> 11 preds = model5.predict(x)
         12 #print ("Predicted: ", decode_predictions(preds, top=3)[0])
         13 


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/keras/engine/training.py in predict(self, x, batch_size, verbose, steps, max_queue_size, workers, use_multiprocessing)
       1876     else:
       1877       return training_arrays.predict_loop(
    -> 1878           self, x, batch_size=batch_size, verbose=verbose, steps=steps)
       1879 
       1880   def train_on_batch(self, x, y=None, sample_weight=None, class_weight=None):


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/keras/engine/training_arrays.py in predict_loop(model, inputs, batch_size, verbose, steps)
        324         ins_batch[i] = ins_batch[i].toarray()
        325 
    --> 326       batch_outs = f(ins_batch)
        327       if not isinstance(batch_outs, list):
        328         batch_outs = [batch_outs]


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/keras/backend.py in __call__(self, inputs)
       2984 
       2985     fetched = self._callable_fn(*array_vals,
    -> 2986                                 run_metadata=self.run_metadata)
       2987     self._call_fetch_callbacks(fetched[-len(self._fetches):])
       2988     return fetched[:len(self.outputs)]


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/client/session.py in __call__(self, *args, **kwargs)
       1437           ret = tf_session.TF_SessionRunCallable(
       1438               self._session._session, self._handle, args, status,
    -> 1439               run_metadata_ptr)
       1440         if run_metadata:
       1441           proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    ~/anaconda3/envs/gpu_tf1/lib/python3.6/site-packages/tensorflow/python/framework/errors_impl.py in __exit__(self, type_arg, value_arg, traceback_arg)
        526             None, None,
        527             compat.as_text(c_api.TF_Message(self.status.status)),
    --> 528             c_api.TF_GetCode(self.status.status))
        529     # Delete the underlying status object from memory otherwise it stays alive
        530     # as there is a reference to status from this from the traceback due to


    UnknownError: Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.
    	 [[{{node layer_conv1/Conv2D}} = Conv2D[T=DT_FLOAT, data_format="NCHW", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](layer_conv1/Conv2D-0-TransposeNHWCToNCHW-LayoutOptimizer, layer_conv1/Conv2D/ReadVariableOp)]]



![png](/images/Task12_MNIST_files/Task12_MNIST_59_2.png)



```python

```
