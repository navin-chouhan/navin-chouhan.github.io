---
title: "Assginment 3 Task 1 Part 2 "
date: 2019-05-27
tags: [CNN, Deep learning]
excerpt: ""
mathjax: "true"
---


```python

# coding: utf-8

# In[1]:


import tensorflow as tf
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#from matplotlib.patches import Rectangle
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


# In[37]:


def loadData():
    x=[]
    y=[]
    path='Q1/task2/Four_Slap_Fingerprint/Ground_truth/'
    a=os.listdir(path)
    for i in a[1:10]:
        impath="Q1/q1task2/Four_Slap_Fingerprint/Image/"+i[0:len(i)-4] + ".jpg"
        
        image=cv2.imread(impath)
        sh1,sh2=image.shape[0],image.shape[1]
        
        image=cv2.resize(image,(900, 900))
        x.append(image)
        
        lines = open( path + i ).readlines()
        
        line = lines[0].split(',')
        l=[int(line[0]),int(line[1]),int(line[2]),int(line[3])]
        y1=resize(sh1,sh2,l)
        
        line = lines[1].split(',')
        l=[int(line[0]),int(line[1]),int(line[2]),int(line[3])]
        y2=resize(sh1,sh2,l)
        
        line = lines[2].split(',')
        l=[int(line[0]),int(line[1]),int(line[2]),int(line[3])]
        y3=resize(sh1,sh2,l)
        
        line = lines[3].split(',')
        l=[int(line[0]),int(line[1]),int(line[2]),int(line[3])]
        y4=resize(sh1,sh2,l)
        
        y.append([y1,y2,y3,y4])
        
    return np.array(x),np.array(y)

# def drawRectangle(image,cordinates):
#     cordinates1=cordinates[0]
#     cordinates2=cordinates[1]
#     cordinates3=cordinates[2]
#     cordinates4=cordinates[3]
    
#     plt.imshow(image)
#     x1,y1=cordinates1[0],cordinates1[1]
#     width=cordinates1[2]-cordinates1[0]
#     height=cordinates1[3]-cordinates1[1]
#     # Add the patch to the Axes
#     plt.gca().add_patch(Rectangle((y1,x1),height,width,linewidth=1,edgecolor='r',facecolor='none'))
        
#     x1,y1=cordinates2[0],cordinates2[1]
#     width=cordinates2[2]-cordinates2[0]
#     height=cordinates2[3]-cordinates2[1]
#     # Add the patch to the Axes
#     plt.gca().add_patch(Rectangle((y1,x1),height,width,linewidth=1,edgecolor='g',facecolor='none'))
        
#     x1,y1=cordinates3[0],cordinates3[1]
#     width=cordinates3[2]-cordinates3[0]
#     height=cordinates3[3]-cordinates3[1]
#     # Add the patch to the Axes
#     plt.gca().add_patch(Rectangle((y1,x1),height,width,linewidth=1,edgecolor='b',facecolor='none'))
        
#     x1,y1=cordinates4[0],cordinates4[1]
#     width=cordinates4[2]-cordinates4[0]
#     height=cordinates4[3]-cordinates4[1]
#     # Add the patch to the Axes
#     plt.gca().add_patch(Rectangle((y1,x1),height,width,linewidth=1,edgecolor='r',facecolor='none'))
                


# In[38]:


def resize(sh1,sh2,groundTruth):
    finalSize=900
    a= int(groundTruth[0]*(finalSize/sh1))
    b= int(groundTruth[1]*(finalSize/sh2))
    c= int(groundTruth[2]*(finalSize/sh1))
    d= int(groundTruth[3]*(finalSize/sh2))
    gt=[a,b,c,d]
    return gt


# In[39]:


x,y=loadData()


# In[40]:


print(x.shape,y.shape)


# In[41]:


#drawRectangle(x[0],y[0])


# In[42]:



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
x_train, x_valid, y_test, y_valid = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
x,y=0,0


# In[43]:


y1_train=np.array([i[0] for i in y_train])
y2_train=np.array([i[1] for i in y_train])
y3_train=np.array([i[2] for i in y_train])
y4_train=np.array([i[3] for i in y_train])

y1_test=np.array([i[0] for i in y_test])
y2_test=np.array([i[1] for i in y_test])
y3_test=np.array([i[2] for i in y_test])
y4_test=np.array([i[3] for i in y_test])

y1_valid=np.array([i[0] for i in y_valid])
y2_valid=np.array([i[1] for i in y_valid])
y3_valid=np.array([i[2] for i in y_valid])
y4_valid=np.array([i[3] for i in y_valid])

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape,x_valid.shape,y_valid.shape)


# In[44]:


train_size = y_train.shape[0]
test_size = y_test.shape[0]
valid_size = y_valid.shape[0]
y_train,y_test=0,0
print(train_size,test_size,valid_size)


# In[45]:


imageSize=(900,900,3)


# In[88]:


main_input = Input(shape=imageSize,name='main_input')

X = Conv2D(kernel_size=5, strides=2, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv1')(main_input)
X = Dropout(0.2)(X)
X = MaxPooling2D(pool_size=4, strides=2)(X)

X = Conv2D(kernel_size=5, strides=2, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv2')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
X = Dropout(0.3)(X)

X = Conv2D(kernel_size=5, strides=2, kernel_initializer='normal',filters=32,activation='relu', name='layer_conv3')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)

X = Conv2D(kernel_size=5, strides=2, kernel_initializer='normal',filters=16,activation='relu', name='layer_conv4')(X)
X = Dropout(0.3)(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)

X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=16,activation='relu', name='layer_conv5')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
X = Dropout(0.3)(X)

X = Conv2D(kernel_size=3, strides=1, kernel_initializer='normal',filters=16,activation='relu', name='layer_conv6')(X)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=2, strides=1)(X)
X = Dropout(0.3)(X)

X = Flatten()(X)


# In[89]:


output1 = Dense(128,kernel_initializer='normal', activation='relu')(X)
output1 = Dense(4, kernel_initializer='normal', name='output1')(output1)


# In[90]:


output2 = Dense(128,kernel_initializer='normal', activation='relu')(X)
output2 = Dense(4, kernel_initializer='normal', name='output2')(output2)


# In[91]:


output3 = Dense(128,kernel_initializer='normal', activation='relu')(X)
output3 = Dense(4, kernel_initializer='normal', name='output3')(output3)


# In[92]:


output4 = Dense(128,kernel_initializer='normal', activation='relu')(X)
output4 = Dense(4, kernel_initializer='normal', name='output4')(output4)


# In[93]:


model = Model(inputs=[main_input], outputs=[output1,output2,output3,output4])


# In[94]:


model.summary()


# In[36]:


model.compile(optimizer=Adam(lr=1e-4),
              loss={'output1' : 'mean_squared_error', 'output2' : 'mean_squared_error' ,'output3' : 'mean_squared_error' ,'output4' : 'mean_squared_error'},
              metrics=['accuracy'])


# In[37]:


model.fit({'main_input': x_train},
          {'output1': y1_train,'output2': y2_train,'output3': y3_train,'output4': y4_train},
          epochs=5, batch_size=10,verbose=1,
          validation_data=({'main_input': x_test},
                        {'output1': y1_test,'output2': y2_test,'output3': y3_test,'output4': y4_test}))


# In[ ]:
a=model.evaluate({'main_input': x_test},{'output1': y1_test,'output2': y2_test,'output3': y3_test,'output4': y4_test})

print(a)
model.save("q1task2.h5")


# In[ ]:



    

```

