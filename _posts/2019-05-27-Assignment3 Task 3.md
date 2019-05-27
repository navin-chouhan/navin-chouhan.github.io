---
title: "Assginment 3 Task 3 "
date: 2019-05-27
tags: [CNN, Deep learning]
excerpt: ""
mathjax: "true"
---


```python
import sys
import getopt

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model
import argparse




def reshp(imgSize, groundTruth):
    ydim=450/imgSize[0]
    xdim=300/imgSize[1]
    
    return [int(groundTruth[0]*ydim), int(groundTruth[1]*xdim)]

def loadData(stg):
    x=[]
    y=[]
    for i in os.listdir(stg + "/Data") :
        img_path = stg + "/Data/" + i
        l= len(i)
        grd_truth_path = stg + "/Ground_truth/" + i[:l-5] + "_gt.txt"

        op=open(grd_truth_path)
        line=op.readlines()[0].split(" ")
        temp =[int(line[0]),int(line[1])]
        
        img=cv2.imread(img_path)
        imgSize=img.shape
        temp=reshp(imgSize, temp)
        y.append(temp)
        
        img=cv2.resize(img,(300, 450))
        x.append(img)
    x=np.array(x)
    y=np.array(y)
    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    x=0
    y=0

    return x_train,x_test,y_train,y_test

def modell(x_train,y_train,epoch):
    model1 = Sequential()
    model1.add(Conv2D(32, kernel_size=(3,3), input_shape=(450,300,3)))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Conv2D(32, kernel_size=(3,3)))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Conv2D(32, kernel_size=(3,3)))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Conv2D(32, kernel_size=(3,3)))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Conv2D(32, kernel_size=(3,3)))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Conv2D(32, kernel_size=(3,3)))
    model1.add(MaxPooling2D(pool_size=(2,2)))

    model1.add(Flatten()) # Flattening the 2D arrays for fully connected layers

    model1.add(Dense(1024, activation=tf.nn.relu))
    model1.add(Dropout(0.2))
    model1.add(Dense(512, activation=tf.nn.relu))
    model1.add(Dropout(0.2))
    model1.add(Dense(128, activation=tf.nn.relu))
    model1.add(Dropout(0.2))
    model1.add(Dense(32, activation=tf.nn.relu))
    model1.add(Dropout(0.2))
    model1.add(Dense(2, activation=tf.nn.relu, name= "output"))

    model1.compile(optimizer='adam', 
                        loss='mean_squared_error', 
                        metrics=['accuracy'])

    model1.fit(x_train,y_train, epochs=epoch, batch_size=128)
    #model1.save("mod.h5")

    return model1

# def main(args):
#   print("hi")
#   if args.phase=="train":
#       stg=input("Enter the training folder: ");
#       epoch=args.epochs

#       x_train,x_test,y_train,y_test=loadData(stg)
#       moddel=modell(x_train,y_train,epoch)
#       moddel.save("mod.h5")
#   else:
#       stg=input("Enter the testing folder: ");
#       x=[]
#       files=[]
#       for i in os.listdir(stg + "/Data")[:10] :
#           img_path = stg + "/Data/" + i
#           files.append(i)         
#           img=cv2.imread(img_path)
#           img=cv2.resize(img,(300, 450))          
#           x.append(img)

#       x=np.array(x)
#       final_model=load_model('mod.h5')
#       ans=final_model.predict(x)

#       for i in range(ans.shape[0]) :
#           f= open('output/'+files[i].split('.')[0]+'.txt',"w")
#           f.write(str(int(ans[i][0]))+' '+str(int(ans[i][1])))
#           f.close()


# args = parser.parse_args()
# main(args)
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--epochs', default=10, type=int)
    args = parser.parse_args()

    if args.phase == "train":
        stg=input("Enter the training folder: ");
        epoch=args.epochs

        x_train,x_test,y_train,y_test=loadData(stg)
        moddel=modell(x_train,y_train,epoch)
        moddel.save("mod.h5")
    else:
        stg=input("Enter the testing folder: ");
        x=[]
        files=[]
        for i in os.listdir(stg) :
            img_path = stg + "/" + i
            files.append(i)         
            img=cv2.imread(img_path)
            img=cv2.resize(img,(300, 450))          
            x.append(img)

        x=np.array(x)
        final_model=load_model('mod.h5')
        ans=final_model.predict(x)
        os.mkdir("output")
        for i in range(ans.shape[0]) :
            f= open('output/'+files[i].split('.')[0]+'.txt',"w")
            f.write(str(int(ans[i][0]))+' '+str(int(ans[i][1])))
            f.close()

```

