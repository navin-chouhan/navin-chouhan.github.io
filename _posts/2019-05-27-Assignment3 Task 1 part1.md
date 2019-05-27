---
title: "Assginment 3 Task 1 Part 1 "
date: 2019-05-27
tags: [CNN, Deep learning]
excerpt: ""
mathjax: "true"
---


```python

# coding: utf-8

# In[27]:


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
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D,Dropout
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,BatchNormalization
from tensorflow.python.keras.optimizers import Adam,SGD
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Lambda


# In[28]:


def loadData():
    x=[]
    y=[]
    cnt=-1
    for i in glob.glob("Q1/task1/Data/*"):
        cnt=cnt+1
        #print(i)
        cnt1=0
        for j in glob.glob(i+"/*.txt"):
            lines=open(j).readlines()
            for im in lines:
                im=im.split(",")
                impath=i+ "/"+ im[0]
                #if(cnt1==2):
                 #   break;
                if(os.path.isfile(impath)):
                    cnt1=cnt1+1
                    image=cv2.imread(impath)
                    image=cv2.resize(image,(350, 350))
                    if(image.shape!=(350, 350, 3)):
                        print("not all same",image.shape)
                    #print(image.shape)
                    x1,y1,x2,y2=int(im[1]),int(im[2]),int(im[3]),int(im[4])
                    y.append([cnt,x1,y1,x2,y2])
                    x.append(image)   
    return np.array(x),np.array(y),cnt+1

def toOneHot(y,numOfClasses):
    if(type(y)==list):
        size=len(y)
    else:
        size=y.size
    y_cls=[[0 for i in range(numOfClasses)] for i in range(size)]
    for i in range(size):
        y_cls[i][y[i]]=1
    return(np.array(y_cls))

def drawRectangle(image,cordinates):
    plt.imshow(image)
    x1,y1=cordinates[0],cordinates[1]
    width=cordinates[2]-cordinates[0]
    height=cordinates[3]-cordinates[1]
    # Add the patch to the Axes
    plt.gca().add_patch(Rectangle((x1,y1),width,height,linewidth=1,edgecolor='r',facecolor='none'))


# In[29]:


x,y,numOfclasses=loadData()
totalImages=x.size
imageSize=x[0].shape


# In[30]:





# In[32]:


#x=x.reshape(totalImages,imageSize[0]*imageSize[1]*imageSize[2])
x_train, x_test, y_train_cls, y_test_cls = train_test_split(x, y, test_size=0.20, random_state=42)


# In[33]:


train_size = y_train_cls.shape[0]
test_size = y_test_cls.shape[0]
print(train_size,test_size)
y_cls1=([i[0] for i in y_train_cls])
y_cls=toOneHot(y_cls1,numOfclasses)
y_cls1=([i[0] for i in y_test_cls])
y_cls_test=toOneHot(y_cls1,numOfclasses)
y_cord_train=np.array([i[1:] for i in y_train_cls])
y_cord_test=np.array([i[1:] for i in y_test_cls])


# In[34]:


main_input = Input(shape=(None,None,3),name='main_input')
X = Lambda(lambda image: tf.image.resize_images( image, ( 350, 350 ) ) ) ( main_input )
X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv1')(X)

#X = Dropout(0.1)(X)
X = MaxPooling2D(pool_size=2, strides=2)(X)
X = Conv2D(kernel_size=5, strides=2, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv2')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
#X = Dropout(0.3)(X)
X = Conv2D(kernel_size=5, strides=2, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv3')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=64,activation='relu', name='layer_conv4')(X)
#X = Dropout(0.1)(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv5')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=2)(X)
#X = Dropout(0.1)(X)
X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv6')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
#X = Dropout(0.3)(X)
X = Flatten()(X)


# In[35]:


output1 = Dense(128,kernel_initializer='normal', activation='relu')(X)
output1 = Dense(numOfclasses, kernel_initializer='normal',activation='sigmoid', name='output1')(output1)


# In[36]:


output2 = Dense(128,kernel_initializer='normal', activation='relu')(X)
output2 = Dense(4, kernel_initializer='normal', name='output2')(output2)


# In[37]:


model = Model(inputs=[main_input], outputs=[output1,output2])


# In[38]:


model.summary()


# In[39]:


model.compile(optimizer=Adam(lr=1e-4),
              loss={'output1': 'binary_crossentropy','output2' : 'mean_squared_error'},
              metrics=['accuracy'])


# In[40]:


# And trained it via:
model.fit({'main_input': x_train},
          {'output1': y_cls,'output2':y_cord_train},
          epochs=1, batch_size=128,verbose=1,
          validation_data=({'main_input': x_test},
                        {'output1': y_cls_test,'output2' : y_cord_test}))


# In[ ]:


model.save('my_mode.h5')

```

