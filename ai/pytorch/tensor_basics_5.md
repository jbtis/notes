
### Reduction Tensor Ops and Arg Max

Reduction operations reduce the number of elements inside a tensor. Perform operations within a single tensor.


```python
import torch
import numpy as np
```


```python
t1 = torch.ones(3,3)
t2 = torch.ones(4,4) + 1
t3 = torch.ones(3,3) + 2
```

#### The most basic reduction operation is tensor()


```python
t1.sum() #sum all the elements inside the tensor, result of the call is a scalar valued tensor
```




    tensor(9.)




```python
t1.sum().numel() #since t1.sun() returns a tensor, we can still perform tensor operations on it
```




    1




```python
t2.numel() #number of elments in the tensor
```




    16




```python
t1.sum().numel() < t1.sum() #since this evaluate to true, we can say sum is a reduction operation
```




    tensor(True)



#### There are many other tensor reduction ops that reduce the tensor to a scalar valued tensor


```python
t3.prod()
```




    tensor(19683.)




```python
t2.mean()
```




    tensor(2.)




```python
t1.std()
```




    tensor(0.)



#### We can decide wich axis to choose to start reducing the dimension of a tensor, thus the reduction operation don't have to always return a scalar valued tensor 


```python
t1 = torch.tensor([
    [1,2,3,4],
    [1,2,3,4],
    [1,2,3,4]
], dtype = torch.float32)
```


```python
t1.shape #we can see this tensor has to axis (basically a 3x4 matrix)
```




    torch.Size([3, 4])




```python
len(t1.shape) # rank
```




    2




```python
t2 = t1.sum(dim = 0) #row reduction, summation of the elements of the first axis
t2
```




    tensor([ 3.,  6.,  9., 12.])




```python
t2.shape
```




    torch.Size([4])




```python
t3 = t1.sum(dim = 1) #column reduction reduction, summation of the elements of the second axis
t3
```




    tensor([10., 10., 10.])




```python
t3.shape
```




    torch.Size([3])



#### Argmax tensor reduction operation 

Argmax tells uns the index of the an element inside a tensor that when applied to a fucntion it results on the highest value of that function. Normally used in output prediction of a neural network


```python
t1 = torch.tensor([
    [1,0,0,4],
    [1,1,0,7],
    [1,5,3,-5]
], dtype = torch.float32) #max value 7
```


```python
t1.max() #biggest value of tensor
```




    tensor(7.)




```python
t1.argmax() #returns index location of the max value in flattened tensor, if no argument is specified
```




    tensor(7)




```python
t1.flatten()
```




    tensor([ 1.,  0.,  0.,  4.,  1.,  1.,  0.,  7.,  1.,  5.,  3., -5.])




```python
t1.max(dim = 0) #max in first axis. The max function also gives as the argmax function as indices = tensor
```




    torch.return_types.max(
    values=tensor([1., 5., 3., 7.]),
    indices=tensor([2, 2, 2, 1]))




```python
t1.max(dim = 1) #max in second axis
```




    torch.return_types.max(
    values=tensor([4., 7., 5.]),
    indices=tensor([3, 3, 1]))




```python
t1.argmax(dim = 0) #equivaent to the second line returned by max
```




    tensor([2, 2, 2, 1])




```python
t1.argmax(dim = 1)
```




    tensor([3, 3, 1])



#### Extracting from a tensor 

Converting from 1 element scalar value tensor to normal python scalar


```python
t1 = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype = torch.float32)
```


```python
t1.mean()
```




    tensor(5.)




```python
t1.mean().item() #only one element tensors can be converted to scalars
```




    5.0



Extracting a tensor to a list


```python
t1.mean(dim = 1)
```




    tensor([2., 5., 8.])




```python
t1.mean(dim = 1).tolist()
```




    [2.0, 5.0, 8.0]




```python
t1.mean(dim = 1).numpy()
```




    array([2., 5., 8.], dtype=float32)


