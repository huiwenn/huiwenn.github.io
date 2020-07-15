---
layout: post
title: "Found Materials Drawing Machine"
date: 2020-07-01 01:09:00
categories: objects
img: "/assets/img/drawingmachine/vid.gif"
---

<!--more-->

![robot]({{ '/assets/img/drawingmachine/vid.gif' | relative_url }})
{: style="width: 60%;" class="center"}
&nbsp;

I built a drawing robot with wood found in the desert! The machanism and software is heavily based on [BranchioGraph](https://brachiograph.readthedocs.io/en/latest/index.html), with a different orientation that required some trignometry to figure out.


![robot2]({{ '/assets/img/bbd/15-2.jpeg' | relative_url }})
*very ad-hoc calibration setup ft. printed protractor and earing*

![robot3]({{ '/assets/img/bbd/15-3.jpeg' | relative_url }})
*first run! It drew a bonfire.*
{: style="width: 60%;" class="center"}

Even after calibration the drawing is still a little... wonky, because of the elasticity in the sticks, the looseness of the joints, and inacuracies of servo controls. In some ways, though, these faults added a layer of intermediacy to the drawing (or the act of drawing), which one may see as part of the machine's _touch_. 

![pallet racks]({{ '/assets/img/bbd/13-7.jpeg' | relative_url }})
![vectors]({{ '/assets/img/bbd/15-4.jpeg' | relative_url }})
![drawing]({{ '/assets/img/bbd/15-5.jpeg' | relative_url }})
*A drawing of a vectorized picture of us building pallet racks.*
{: style="width: 60%;" class="center"}

&nbsp;

The bot even participated in our weekly _drink-and-draw_ event! People blew into a alcohol sensor appended on to raspberry Pi, whose reading were used to parametrize the  `open-cv` vectorize function. So technically we drank and it drew. Here is [a video](https://twitter.com/_yokaii_/status/1249096320083607553 ) of the bot drawing. 


&nbsp;
You can find my code for it [here](https://github.com/guiguiguiguigui/chatsubo-e).

