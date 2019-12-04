
### Tensor operation types
- Reshaping 
- Element-wise
- Reduction operations
- Access operations

We will first focus on reshaping


```python
import torch
```


```python
t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype = torch.float32)
```

#### Reshaping operations


```python
t.size()
```




    torch.Size([4, 4])




```python
t.shape
```




    torch.Size([4, 4])




```python
len(t.shape) #rank
```




    2




```python
t.numel()
```




    16




```python
t.reshape(16,1)
```




    tensor([[1.],
            [1.],
            [1.],
            [1.],
            [2.],
            [2.],
            [2.],
            [2.],
            [3.],
            [3.],
            [3.],
            [3.],
            [4.],
            [4.],
            [4.],
            [4.]])




```python
t.reshape(8,2)
```




    tensor([[1., 1.],
            [1., 1.],
            [2., 2.],
            [2., 2.],
            [3., 3.],
            [3., 3.],
            [4., 4.],
            [4., 4.]])




```python
t.reshape(2,8)
```




    tensor([[1., 1., 1., 1., 2., 2., 2., 2.],
            [3., 3., 3., 3., 4., 4., 4., 4.]])




```python
t.reshape(1,16)
```




    tensor([[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 4., 4., 4., 4.]])




```python
t.numel()
```




    16




```python
t.reshape(2,2,2,2)
```




    tensor([[[[1., 1.],
              [1., 1.]],
    
             [[2., 2.],
              [2., 2.]]],
    
    
            [[[3., 3.],
              [3., 3.]],
    
             [[4., 4.],
              [4., 4.]]]])




```python
t = t.reshape(16,1)
```


```python
t.shape
```




    torch.Size([16, 1])




```python
t = t.reshape(16,1).squeeze() #get rid of ais of length 1
```


```python
t.shape
```




    torch.Size([16])




```python
t = t.unsqueeze(dim = 5) # added 6 length 1 axis to the tensor
```


```python
t.shape
```




    torch.Size([1, 1, 1, 1, 1, 1, 16])



Flatennig a tensor: remove all axes except 1 (creates a 1d array with all scalar components of a tensor)
Convolutional layer -> fully connected layer requires flatennig of a tensor. We can also concatenate tensors.


```python
def flatten(t):
    t = t.reshape(1,-1) #-1 becomes whatever needed to account all scalars in the nd tensor input
    t = t.squeeze()
    return t
```


```python
a = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
]
)
```


```python
a.shape
```




    torch.Size([3, 3])




```python
a = flatten(a)
```


```python
a.shape
```




    torch.Size([9])




```python
a = a.unsqueeze(dim = 0)
```


```python
a.shape
```




    torch.Size([1, 9])




```python
a.size()[0] #size of the first axis
a.size()[1] #size of the second axis
```




    9




```python
a.shape[0]
```




    1




```python
def flattenx(t):
    t = t.reshape(1, t.shape[0]*t.shape[1]) #would only work fr 2d tensor to 1d tensor convertion
    t = t.squeeze()
    return t
```


```python
b = torch.tensor([
    [1,2,3,4],
    [4,5,6,5],
    [7,8,9,9],
    [1,1,1,1]
]
)
```


```python
flattenx(b) == flatten(b)
```




    tensor([True, True, True, True, True, True, True, True, True, True, True, True,
            True, True, True, True])


