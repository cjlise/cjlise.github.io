---
title: "Deep Learning Lab 2: Autograd: automatic differentiation"
date: 2019-12-02
categories: machine-learning
tags: [Deep Leaning, PyTorch, Machine Learning, Gradient, Tensors, Autograd, Non-linear regression]
header: 
   image: "/images/DeepLearning/web-3706561_200.jpg"
excerpt: "Deep Leaning, PyTorch, Machine Learning, Gradient, Tensors, Autograd, Non-linear regression"
mathjax: "true"
---

# Deep Learning Lab 2
This lab covers is basically a <a href="https://pytorch.org/">pyTorch </a> autograd (automatic differentiation). At the end a simple non-linear regression model is presented. In details the following topics are covered: 
* Visualizing the computational graph
* Gradient accumulation
* Non-linear regression with a neural network

## Non-linear regression 
The sample in the Jupyter notebook present the non-linear model below: 
$$y = -(0.1 * x_1^2 + 3 sin(0.1 x_2) +0.1)$$   
![Non-linear function](/images/DeepLearning/Labs/Lab2-NeuralNetwork.jpg "Non-linear function")

We will solve this problem using a simple neural network using a single hidden layer with no activation function:   
![Neural Network](/images/DeepLearning/Labs/Lab2-NeuralNetwork.jpg "Neural Network")
 

 

Here is the <a href="https://colab.research.google.com/drive/1a2tyhCuuOAyX47dpv7jFf_QmIH9rtFCC">link</a> to access the notebook on Google colab. It doesn't work directly on github. 






