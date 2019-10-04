
### Tensor Basics


```python
import torch
```


```python
print(torch.__version__)
```

    1.2.0
    


```python
torch.cuda.is_available()
```




    False




```python
torch.version.cuda
```




    '10.0'




```python
t = [1,2,3]
t = torch.tensor(t)
```


```python
type(t)
```




    torch.Tensor




```python
t.shape
```




    torch.Size([3])




```python
t = [[1,2,3],[3,4,5],[5,6,7]]
```


```python
t = torch.tensor(t)
```


```python
a = torch.tensor(([1,2,3],[3,4,5],[5,6,7]))
```


```python
a.shape
```




    torch.Size([3, 3])




```python
a[0]
```




    tensor([1, 2, 3])




```python
a[1][2]
```




    tensor(5)




```python
a = [
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [8,8,8]
]
```


```python
a = torch.tensor(a)
```


```python
a
```




    tensor([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [8, 8, 8]])




```python
a.reshape(6,2)
```




    tensor([[1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 8],
            [8, 8]])




```python
import numpy as np 
```


```python
print(a.dtype)
print(a.device)
print(a.layout)
```

    torch.int64
    cpu
    torch.strided
    


```python
data = np.array([1,2,3])
type(data)
```




    numpy.ndarray




```python
  torch.Tensor(data)
```




    tensor([1., 2., 3.])




```python
torch.tensor(data) ##data type matches input data
```




    tensor([1, 2, 3], dtype=torch.int32)




```python
torch.as_tensor(data)
```




    tensor([1, 2, 3], dtype=torch.int32)




```python
torch.from_numpy(data)
```




    tensor([1, 2, 3], dtype=torch.int32)




```python
torch.eye(4)
```




    tensor([[1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.]])




```python
torch.zeros(3,3)
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])




```python
torch.ones(2,2)
```




    tensor([[1., 1.],
            [1., 1.]])




```python
torch.rand(4,4)
```




    tensor([[0.9186, 0.6569, 0.1486, 0.6594],
            [0.9430, 0.1013, 0.5323, 0.7142],
            [0.4000, 0.3188, 0.1529, 0.2570],
            [0.3321, 0.1528, 0.1200, 0.7683]])



### Create tensors using the 4 options 


```python
data = np.array([1,2,3])
```


```python
t1 = torch.Tensor(data) #constructor
t2 = torch.tensor(data) # factory fucnction <- returns object (more dynamic object creation)
t3 = torch.as_tensor(data) #factory functions
t4 = torch.from_numpy(data) #fatory function
```


```python
print(t1)
print(t2)
print(t3)
print(t4)
```

    tensor([1., 2., 3.])
    tensor([1, 2, 3], dtype=torch.int32)
    tensor([1, 2, 3], dtype=torch.int32)
    tensor([1, 2, 3], dtype=torch.int32)
    


```python
print(t1.dtype) #constructor uses global default dtype value
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)
```

    torch.float32
    torch.int32
    torch.int32
    torch.int32
    


```python
torch.get_default_dtype()
```




    torch.float32



#### Factory functions have type inference:


```python
torch.tensor(np.array([1,2,3]))
```




    tensor([1, 2, 3], dtype=torch.int32)




```python
torch.tensor(np.array([1.,2.,3.]))
```




    tensor([1., 2., 3.], dtype=torch.float64)



#### Set data type


```python
torch.tensor(np.array([1,2,3]), dtype = torch.float64)
```




    tensor([1., 2., 3.], dtype=torch.float64)



#### Constructor and Factory Function Distinct Behavior while modifying array but not tensor



```python
data = np.array([1,2,3])
```


```python
t1 = torch.Tensor(data)
t2 = torch.tensor(data)
t3 = torch.as_tensor(data)
t4 = torch.from_numpy(data) #only affects numpy arrays
```


```python
data[0] = 0
data[1] = 0 
data[2] = 0
```

`torch.Tensor` and `torch.tensor` contain the original data from the array. Create adittional copy in memory.


```python
print(t1)
print(t2)
```

    tensor([1., 2., 3.])
    tensor([1, 2, 3], dtype=torch.int32)
    

`torch.as_tensor` and `torch.from_numpy` mimic the current array values. Share memory with numpy array.


```python
print(t3)
print(t4)
```

    tensor([0, 0, 0], dtype=torch.int32)
    tensor([0, 0, 0], dtype=torch.int32)
    

Use torch.tensor() for everyday use and torch.as_tensor() for performance.
The latters needs to copy numpy array to the GPU.


```python

```
