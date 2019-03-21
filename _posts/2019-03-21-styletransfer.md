---
title: "Coloring Black&White movies Using Style Transform"
date: 2019-03-21
tags: [cycleGAN, Deep learning, Style Transfer]
header:
  image: "/images/p1.png"
excerpt: ""
mathjax: "true"
---

# Problem Statement
Neural style transform is one of the most fascinating and interesting
application of Deep Learning.  With the help of this neural style transform
we are going to color black and white videos.
Let’s say we have a very old Bollywood movie in black and white and we
have a latest Bollywood coloured movie.  Now using this colored movie we
are going to color our old black and white movie and obtain a coloured
version of that old movie.

# Motivation and Challenges
Nowadays many people want to enjoy old movies and games, but of their
old color contrast and shady black and white color people avoid watching
them.  This might be the reason behind all this remakes coming nowadays.
Using neural style transfer we can provide new life to all this old movies.
Also it can provide a new look to many modern games and videos and can
boast today’s entertainment industry.
Neural style transfer seems to be very attractive and fascinating but
implementing this network can be bit challenging in various aspects.
* Style transfer is used to redefine images and small clips with few objects
involved at a time in each frame but the most challenging part of this
project is to apply style transfer to a movie with lots of objects involved at
a time in each frame.
* Other problems involved are whether all the objects are getting
appropriate color, our network is able to distinguish between variety of
objects, etc.
* Problems related to network may occur at the time of implementation.
Dealing with all this problems will be challenging part for us.

# Work Done
The current scenario of the work done is as follows:  1) We are trying to
collect and read as many papers related to style transfer as possible so
that we can get essence of it.  2) Basic understanding of cycleGAN,
textureGAN and related stuff.  3) Turning Fortnite into PUBG with Deep
Learning using (CycleGAN)

# Proposed Methodology
The steps or let's say methodology in which are thinking of proceeding are as follows:
* Dividing the input video and the style video into numerous frames.
* Learning and setting up neural network f:X -> Y,  which will transform the frames data sets from X domain (Black & White) to Y domain (colored).
* Learning another network f: Y-> X , which is the exact reverse of the above network.
* Computing the cycle loss and training the model with appropriate data set.
* Discriminator network decides whether the frame is transformed or whether it is original in particular domain.
<img src="{{ site.url }}{{ site.baseurl }}/images/p2.png" alt="linearly separable data">


+ [link](/images/Presentaion.pdf) to Presentaion slide.
+ [link](/images/Grp_16.mp4) to Video Explanation.
