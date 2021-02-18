---
title: "Multi-class Multi-label Classification Model based on the O\*NET Job Database"
date: 2021-02-18
categories: machine-learning
tags: [Classification, Deep Learning, Keras, Tensorflow, Sk-learn, Neural Network, NLP, Machine Learning, Python, Multi-class, Multi-label]
header: 
   image: "/images/MachineLearning/landscape-4527525_200.jpg"
excerpt: "Classification, Deep Learning, Keras, Tensorflow, Sk-learn, Neural Network, NLP, Machine Learning, Python, Multi-class, Multi-label"
---


This project is a kind of real life use case: there is one database (O\*NET database) which is not designed to be used for machine learning, and we want to build a model that predicts a job title and abilities/skills from a job description. As the raw data is not initially framed for machine learning, we will have to extensively rearange it to get a dataset usable for machine learning. 
Then we will build a deep learning model using NLP and Keras. And finally we will use this model in a web application that we will deploy on HEROKU using Flask.   
 

## 1. Dataset design and model creation 
The Jupyter notebook below covers the dataset creation, and the model design.  

[ONET Job Database Model](https://github.com/cjlise/MachineLearning/blob/master/DeepLearning/OnetJobDatabaseAnalysis.ipynb) 


## 2. Web application
The [Web Application](https://thawing-hamlet-30375.herokuapp.com/) using the model parameters have been deployed on HEROKU, using Flask.

![Web App](/images/DeepLearning/projects/onet-webapp.png "Web App")
   
	