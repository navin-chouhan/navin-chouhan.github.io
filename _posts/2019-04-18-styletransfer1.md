---
title: "[First Update]Coloring Black&White movies Using Style Transform"
date: 2019-03-21
tags: [cycleGAN, Deep learning, Style Transfer]
header:
  image: "/images/p1.png"
excerpt: ""
mathjax: "true"
---

# Problem Statement
Neural style transfer is one of the most fascinating and interesting
application of Deep Learning. With the help of this neural style transfer
we are going to color black and white videos.
Let’s say we have a very old Bollywood movie in black and white and we
have a latest Bollywood coloured movie. Now using this colored movie we
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

# Data sets
This black and white image from Mughal-e-Azam is transformed into a
coloured image using style transfer.

# Work Done
The current scenario of the work done is as follows:  1) We are trying to
collect and read as many papers related to style transfer as possible so
that we can get essence of it.  2) Basic understanding of cycleGAN,
textureGAN and related stuff.  3) Turning Fortnite into PUBG with Deep
Learning using (CycleGAN)

# Proposed Methodology
The steps or let’s say methodology in which are thinking of proceeding are
as follows:
* Dividing the input video and the style video into numerous frames.
* Learning neural network f:X → Yi.e.Generator (X 2Y ),
which will transform the frames data sets from X domain (Black White)
to Y domain (colored).
* Learning another network f:Y → Xi.e.Generator (Y 2X ),
which will transform the frames data sets from Y domain (colored) to X
domain (Black White), which is the exact reverse of the above network.
4) Learning a network called Discriminator X, which will classify whether
the frame belongs to domain X or it is transformed into domain X from
another domain.
* Learning another network called Discriminator Y having similar
functioning like of Discriminator X, this network will classify whether the
frame belongs to domain Y or it is transformed into domain Y from
another domain.
* Computing all the losses obtained from Discriminator X, Discriminator
Y, Generator X2Y and Generator Y2X and training all the networks
simultaneously and training the model with appropriate data set.
<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/images/image.IK8L0Z.png" alt="Figure: Block diagram of network when input is from Domain X">
  <figcaption>Figure: Block diagram of network when input is from Domain X</figcaption>
</figure>


# Methodology explored
Architecture of Generators:
The Generator have three components:
* Encoder: It is a CNN network consisting 3 convolutional layers.
* Transformer: It is a residual network consisting 6 resnet blocks.
* Decoder: It is a deconvolution network consisting 3 deconv layers.
Architecture of Discriminators:
The Discriminator is a simple classifier consisting several convolutional
layers.
Training all the networks needs all the losses to be computed. And the
losses obtained from all the networks are as follows:
* Loss of Discriminator X
D_X_loss = L2(dec_X, 1) + L2(dec_gen_X, 0)
dec_X = score of discriminator when frame is from X domain.
dec_gen_X = score of discriminator when frame is transformed into X
domain.
* Loss of Discriminator y
D_Y_loss = L2(dec_Y, 1) + L2(dec_gen_Y, 0)
dec_Y = score of discriminator when frame is from Y domain.
dec_gen_Y = score of discriminator when frame is transformed into Y
domain.
* Loss of Generator X2Y
g_loss_X2Y = L2(dec_gen_Y, 1) + Lambda * cyc_loss.
4) Loss of Generator Y2X
g_loss_Y2X = L2(dec_gen_X, 1) + Lambda * cyc_loss.
Where,
cyc_loss = L2(input_X, cyc_X) + L2(input_Y, cyc_Y)
cyc_X = Frame generated when input_X (frame of X domain) is
transformed into Y domain and then again transformed into X domain.
cyc_Y = Reverse of cyc_X .

<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/images/image.DFEB0Z.png" alt="Figure: Resnet Architecture">
  <figcaption>Figure: Resnet Architecture</figcaption>
</figure>

<figure>
  <img src="{{ site.url }}{{ site.baseurl }}/images/image.LQI7ZZ.png" alt="Figure: Generator Architecture">
  <figcaption>Figure: Generator Architecture</figcaption>
</figure>

# Results and their discussion
* Generator X2Y tries to learn the model in such a way that the
Discriminator Y is not able to differentiate between the generated frame
and the real frame.
* Similarly Generator Y2X tries to learn the model in such a way that the
Discriminator X is not able to differentiate between the generated frame
and the real frame.
* After successfully training the model both the Generators would be able
to make fool of the Discriminator.

# Conclusion and Future Work
At this stage we can conclude that now we have basic understanding of
CycleGAN, it’s architecture and how to implement it. From now onward
our major focus will be on the implementation part. We will try to achieve
best model by varying the architecture of Generators and Disrciminator.

# Work Done
The current scenario of the work done is as follows:
* We are trying to collect and read as many papers related to style
transfer as possible so that we can get essence of it.
* Basic understanding of CycleGAN, TextureGAN and related stuff.
* Trying to implement CycleGAN model.
* Reading and understanding blogs like Turning Fortnite into PUBG with
Deep Learning (using CycleGAN).
* Also we are following prof.Andrew Ng tutorial on style transfer.

<iframe width="720" height="720" src="/images/DL_project_2/DL_project_2.mp4" frameborder="0" allowfullscreen></iframe>

+ [link](/images/DL_project_2/DL_project_2.pdf) to Presentaion slide.
<!-- + [link](https://drive.google.com/open?id=1jQ-QWi7m7--m-qSnHcyRUXFfoVgFd-nz) to Video Explanation. -->
