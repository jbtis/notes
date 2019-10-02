# Perceptrons and Sigmoid Neurons

## What are perceptrons?

Perceptrons are the old neuron models used that were first used. The model consists of any given number of inputs and a only one output.
The model can be described by the following threshold function:

<img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\begin{cases}&space;1&space;&\mbox{if&space;}&space;\boldsymbol&space;w&space;\cdot&space;\boldsymbol&space;x&space;&plus;&space;b&space;>&space;0,&space;\\&space;0&space;&&space;\mbox{otherwise&space;}&space;\end{cases}" title="f(x) = \begin{cases} 1 &\mbox{if } \boldsymbol w \cdot \boldsymbol x + b > 0, \\ 0 & \mbox{otherwise } \end{cases}" />

Let's break down the formula:
- __*x*__ is an input vector, that is values from previous layers of the neural network or from the data itself
- __*w*__  is also an input vector that represents the weight of that connection
- *b* is a number which is a bias. This number will either "help" or "hinder" the ability of the neuron to make it past the threshold value.

So if the dot product with the help of the bias make it past the threshold, the perceptron will 'fire' and output a 1, if not then it will output a 0. 

<img src = "https://miro.medium.com/max/1794/1*n6sJ4yZQzwKL9wnF5wnVNg.png" width = 400>


