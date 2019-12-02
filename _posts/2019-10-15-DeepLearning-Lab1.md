---
title: "Deep Learning Lab 1: Pytorch tensors tutorial"
date: 2019-10-15
categories: machine-learning
tags: [Deep Leaning, PyTorch, Machine Learning, Linear Regression, Gradient, Tensors]
header: 
   image: "/images/DeepLearning/web-3706561_200.jpg"
excerpt: "Deep Leaning, PyTorch, Machine Learning, Linear Regression, Gradient, Tensors"
mathjax: "true"
---

# Deep Learning Lab 1
This lab covers is basically a <a href="https://pytorch.org/">pyTorch </a> tensors tutorial. At the end a simple linear regression model is presented. In details the following topics are covered: 
* PyTorch Tensor basics
* Broadcasting 
* Linear regression

# Linear regression 
The sample in the Jupyter notebook present the linear model below: 
$y = x w + b$ 
with:   
* y: a vector of 30 rows   
* x: a matrix of dimension (30,2)   
* b: a constant. But in the formula above we can see it as b * vector of 1 of dimension 30 using broadcasting  

Let's call y the real solution and $\hat y$ the predicted value.  
The loss function is $ L(w_s, b_s) = \|y - \hat y\|^2 = \displaystyle\sum_{k=1}^{30} (y_k - \hat y_k)^2 $

And we have to minimize the loss function L to find the solution. 

$$\hat y = x w_s + b_s $$ 
$$\hat y_k = x_k,1 w_s1 + x_k,2 w_s2 $$ 

The loss function gradient is given by:   
$$\nabla _{w_s}L(w,b) =  \begin{bmatrix}
                          \displaystyle\sum_{k=1}^{30} -2 x_{k,1} (y_k - \hat y_k) \\
                          \displaystyle\sum_{k=1}^{30} -2 x_{k,2} (y_k - \hat y_k)
						\end{bmatrix} = -2x^t (y - \hat y)$$

$$\nabla _{b_s}L(w,b) =  \begin{bmatrix}
                          \displaystyle\sum_{k=1}^{30} -2  (y_k - \hat y_k) \\
                          \displaystyle\sum_{k=1}^{30} -2  (y_k - \hat y_k)
						\end{bmatrix} $$  
  
The gradient based algorithm to compute $w_s$ and $b_s$ at each iteration is: 
* $$w_s^{j+1} = w_s^j - \eta \nabla _{w_s}L(w_s^j,b_s^j)$$ 
* $$b_s^{j+1} = b_s^j - \eta \nabla _{b_s}L(w_s^j,b_s^j)$$  

   

Here is the <a href="https://colab.research.google.com/drive/1T6x-ToztavZ1DJf5FBFDQQA54rm6xo6q">link</a> to access the notebook on Google colab. It doesn't work directly on github. 





