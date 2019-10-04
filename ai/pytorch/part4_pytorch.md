
## Element-wise operations 

Element-wise (or component wise or point wise) operations are operations between elements of two different tensors that are corresponding elements
corresponding elemetns are elements that ocuppy the same index position.


```python
import torch
import numpy as np
```


```python
t1 = torch.tensor([
    [1,2],
    [3,4]
], dtype = torch.float32)
```


```python
t2 = torch.tensor([
    [6,7],
    [8,9]
], dtype = torch.float32)
```


```python
t1[0] #first axis example
```




    tensor([1., 2.])




```python
t2[0][0] #second axis example
```




    tensor(6.)




```python
print(t1[0][0])
print(t2[0][0])
```

    tensor(1.)
    tensor(6.)
    

*Element-wise operations are only possible within two tensor with the same shape*


```python
t1 + t2 #element-wise adittion (same for subtract, multiply, divide)
```




    tensor([[ 7.,  9.],
            [11., 13.]])



#### The following operations work because of tensor broadcasting:
This is basically operator overlaoding


```python
t1 - 4
```




    tensor([[-3., -2.],
            [-1.,  0.]])




```python
t1 * 7
```




    tensor([[ 7., 14.],
            [21., 28.]])




```python
t1 > 0
```




    tensor([[True, True],
            [True, True]])




```python
t1.sub(4)
```




    tensor([[-3., -2.],
            [-1.,  0.]])




```python
t1.mul(7)
```




    tensor([[ 7., 14.],
            [21., 28.]])




```python
t1.div(8)
```




    tensor([[0.1250, 0.2500],
            [0.3750, 0.5000]])




```python
t3 = torch.tensor([
    [1,2],
    [3,4]
], dtype = torch.uint8)
```

`t3` cannot operate with `t1` or `t2`. 


```python
t1 + t3
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-88-4ac166e17b2d> in <module>
    ----> 1 t1 + t3
    

    RuntimeError: expected device cpu and dtype Float but got device cpu and dtype Byte


Dividing quotient


```python
t3/8
```




    tensor([[0, 0],
            [0, 0]], dtype=torch.uint8)



### A closer look into the broadcast functionality 


```python
np.broadcast_to(2, t1.shape) #broadcasting scalar value 2 
```




    array([[2, 2],
           [2, 2]])



So doing:


```python
t1 + 8
```




    tensor([[ 9., 10.],
            [11., 12.]])



is the shorthand of doing:


```python
t1 + torch.tensor(
    np.broadcast_to(8, t1.shape), 
    dtype = torch.float32
)
```




    tensor([[ 9., 10.],
            [11., 12.]])



#### Another Example 


```python
t1 = torch.ones(2,2)
```


```python
t1.type(torch.float32)
```




    tensor([[1., 1.],
            [1., 1.]])




```python
t1.type()
```




    'torch.FloatTensor'




```python
t1.shape
```




    torch.Size([2, 2])




```python
t2 = torch.tensor([4,6], dtype = torch.float32)
```


```python
t2.shape
```




    torch.Size([2])




```python
t1 + t2 # Adding different shape tensors is possible thanks to broadcasting 
```




    tensor([[5., 7.],
            [5., 7.]])




```python
t1 = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype = torch.float32)
```


```python
t1.le(1)
```




    tensor([[ True, False, False],
            [False, False, False],
            [False, False, False]])




```python
t1.eq(3)
```




    tensor([[False, False,  True],
            [False, False, False],
            [False, False, False]])




```python
t1.lt(4)
```




    tensor([[ True,  True,  True],
            [False, False, False],
            [False, False, False]])




```python
t1.abs()
```




    tensor([[1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])




```python
t1.neg()
```




    tensor([[-1., -2., -3.],
            [-4., -5., -6.],
            [-7., -8., -9.]])



Each of this would be the same as doing:


```python
t1 <= torch.tensor(
    np.broadcast_to(7, t1.shape),
    dtype = torch.float32
)
```




    tensor([[ True,  True,  True],
            [ True,  True,  True],
            [ True, False, False]])



or


```python
t1 <= torch.tensor([
    [7,7,7],
    [7,7,7],
    [7,7,7]],
    dtype = torch.float32
)
```




    tensor([[ True,  True,  True],
            [ True,  True,  True],
            [ True, False, False]])



which is really tedious!

*The takeaway: Get good at broadcasting! Very handy for advanced (efficient) neural network programming,it is a tool for more complex thought*
