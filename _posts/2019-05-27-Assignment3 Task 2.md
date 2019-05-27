---
title: "Assginment 3 Task 2"
date: 2019-05-27
tags: [CNN, Deep learning]
excerpt: ""
mathjax: "true"
---
```python
# coding: utf-8

# In[115]:


import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import math
import numpy as np
import os
import gzip
import glob
from time import time
import cv2
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input ,Concatenate
from tensorflow.python.keras.layers import Reshape, MaxPooling2D,Dropout,Conv2DTranspose,UpSampling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,BatchNormalization
from tensorflow.python.keras.optimizers import Adam,SGD
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard


# In[145]:


def loadData():
    x=[]
    y=[]
    for i in os.listdir("Q2/Data")  :
        path_mask = "Q2/Mask/" +"_groundtruth_(1)_Image"+ i[14:]
        path_image = "Q2/Data/" + i
        image = cv2.imread(path_image)
        mask = cv2.imread(path_mask)
        x.append(image)
        y.append(mask)
    return np.array(x),np.array(y)
    


# In[146]:


x,y=loadData()
imageSize=x[0].shape


# In[147]:


print(x.shape,y.shape)


# In[148]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
x,y=0,0


# In[151]:


main_input = Input(shape=imageSize,name='main_input')

X1 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv1',padding='same')(main_input)
X1 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv2',padding='same')(X1)
X1 = BatchNormalization()(X1)
X1 = Dropout(0.2)(X1)
X2 = MaxPooling2D(pool_size=2, strides=2)(X1)

X2 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv3',padding='same')(X2)
X2 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv4',padding='same')(X2)
X2 = BatchNormalization()(X2)
X2 = Dropout(0.2)(X2)
X3 = MaxPooling2D(pool_size=2, strides=2)(X2)

X3 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv5',padding='same')(X3)
X3 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv6',padding='same')(X3)
X3 = Conv2D(kernel_size=1, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv7',padding='same')(X3)
X3 = BatchNormalization()(X3)
X3 = Dropout(0.2)(X3)
X4 = MaxPooling2D(pool_size=2, strides=2)(X3)

X4 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=128,activation='relu', name='layer_conv8',padding='same')(X4)
X4 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=128,activation='relu', name='layer_conv9',padding='same')(X4)
X4 = Conv2D(kernel_size=1, strides=1, kernel_initializer='normal',filters=128,activation='relu', name='layer_conv10',padding='same')(X4)
X4 = BatchNormalization()(X4)
X4 = Dropout(0.2)(X4)
X5 = MaxPooling2D(pool_size=2, strides=2)(X4)

X5 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=512,activation='relu', name='layer_conv11',padding='same')(X5)
X5 = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=512,activation='relu', name='layer_conv12',padding='same')(X5)
X5 = BatchNormalization()(X5)
X5 = Dropout(0.2)(X5)
X5 = Conv2D(kernel_size=1, strides=1, kernel_initializer='normal',filters=512,activation='relu', name='layer_conv13',padding='same')(X5)




X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=512,activation='relu', name='layer_conv14',padding='same')(X5)
X = Conv2DTranspose(kernel_size=3, strides=1, kernel_initializer='normal',filters=512,activation='relu', name='layer_deconv1',padding='same')(X)
concat = Concatenate(axis=3)
X=concat([X5, X])
X=UpSampling2D(size=(2, 2))(X)

X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=128,activation='relu', name='layer_conv15',padding='same')(X)
X = Conv2DTranspose(kernel_size=3, strides=1, kernel_initializer='normal',filters=128,activation='relu', name='layer_deconv2',padding='same')(X)
X=UpSampling2D(size=(2, 2))(X)


X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv16',padding='same')(X)
X = Conv2DTranspose(kernel_size=(4,1), strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_deconv3')(X)
concat = Concatenate(axis=3)
X=concat([X, X3])
X=UpSampling2D(size=(2, 2))(X)


X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv17',padding='same')(X)
X = Conv2DTranspose(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_deconv4',padding='same')(X)
concat = Concatenate(axis=3)
X=concat([X, X2])
X=UpSampling2D(size=(2, 2))(X)

X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv18',padding='same')(X)
X = Conv2DTranspose(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_deconv5',padding='same')(X)
concat = Concatenate(axis=3)
X=concat([X, X1])
X = Conv2D(kernel_size=2, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv19',padding='same')(X)
X = Conv2D(kernel_size=2, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_deconv20',padding='same')(X)
X = Conv2D(kernel_size=2, strides=1, kernel_initializer='normal',filters=16,activation='relu', name='layer_deconv21',padding='same')(X)

output= Conv2D(kernel_size=1, strides=1, kernel_initializer='normal',filters=3,activation='relu', name='output',padding='same')(X)


model = Model(inputs=[main_input], outputs=[output])


# In[152]:


model.summary()


# In[29]:


model.compile(optimizer=Adam(lr=1e-4),
              loss={'output' : 'mean_squared_error'},
              metrics=['accuracy'])


# In[ ]:


# And trained it via:
model.fit({'main_input': x_train},
          {'output': y_train},
          epochs=40, batch_size=3,verbose=1,
          validation_data=({'main_input': x_test},
                        {'output': y_test}))

model.save('my_model_concat.h5')

```

