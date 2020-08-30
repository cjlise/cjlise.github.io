---
title: "Product Adoption Simulation With Network And Broadcast Influence Using NetLogo"
date: 2020-08-30
categories: machine-learning
tags: [Simulation, Agent Based Modeling, NetLogo, Diffusion, ABM]
header: 
   image: "/images/MachineLearning/landscape-4527525_200.jpg"
excerpt: "Simulation, Agent Based Modeling, NetLogo, Diffusion, ABM"
---

# Product Adoption Simulation With Network And Broadcast Influence Using NetLogo   
## WHAT IS IT?  
The purpose of this model is to evaluate the adoption of a new product accross a social network. It also take into account broadcast influence. 

## HOW IT WORKS
First, a social network must be created. To do this, we use the “Preferential Attachment” method. In this method, we start with two nodes connected by an edge. Then, at each step, a node is added. The new node chooses to connect to an existing node randomly, but with a bias based on the number of connections, or “degree”, the existing node already has. So the higher the degree of an existing node, the more likely a new node will connect to it. A new node “prefers” to connect to an existing node with a higher number of connections. (See the “Preferential Attachment” Sample Model.)

We use the Bass Model to simulate the new product adoption. This model uses two parameters:  

* The marketing effect, which is the marketing effect: Broadcast-influence variable  
* The Network or word-of-mouth effect: social-influence variable   
Moreover The word-of-mouth effect only works through immediate neighbors in a network.

At each step, a member of the network will have the following choice:  

* Adopt from advertising with a probability p controlled by the Broadcast-influence variable
* Adopt from social influence with the coefficient social-influence * ((number of adopted neighbors)/(total number of neighbors))

## HOW TO USE IT
I. Setting Up the Network Use the POPULATION slider to select the number of people you want to exist in the social network. The SETUP button provides a starting point for the network (two people connected by a link). Click the CREATE-NETWORK button to allow the preferential attachment network to fully form. It will stop when the POPULATION number of people is reached, resetting ticks to 0 and releasing the button. The LAYOUT? switch controls whether or not the layout procedure is run. This procedure intends to make the network structure easier to see by moving the nodes around.

Use the NUM-SEED-ADOPTERS slider to define the number of initial adopters. The social adoption will spread from those initial adopters. The initial adopters are displayed in green.

II. Spread the product adoption The product adoption diffusion is controlled by the broadcast-influence and social-influence sliders. You can start or stop the process by clicking on the go button. The process automatically stops when the product is adopted by the whole population. 

The model can be lanched in this [link](https://github.com/cjlise/MachineLearning/blob/master/ABM/ProductAdoptionSocialNetwork-JLise.html).  
 
 ![NetLogo Model](/images/MachineLearning/NetLogo-ProductAdoption.jpg "NetLogo Model")



	