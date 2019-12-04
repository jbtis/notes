
### Concatenating vs Stacking 

Concatenating joins a sequence of tensor along a existing axis while Stacking joins a sequene of tensors alonga new axis.

#### Tensor Recap


```python
import torch
```


```python
t1 = torch.tensor([1,1,1])
```


```python
t1
```




    tensor([1, 1, 1])




```python
t1.unsqueeze(dim = 0) #create a new axis on position 0
```




    tensor([[1, 1, 1]])




```python
t1.unsqueeze(dim = 0).shape
```




    torch.Size([1, 3])




```python
t1.unsqueeze(dim = 1) #create new ais on position 1
```




    tensor([[1],
            [1],
            [1]])




```python
t1.unsqueeze(dim = 1).shape
```




    torch.Size([3, 1])




```python
t1 = torch.tensor([1,1,1])
t2 = torch.tensor([2,2,2])
t3 = torch.tensor([3,3,3])
```


```python
tcat = torch.cat((t1,t2,t3), dim = 0)
tcat
```




    tensor([1, 1, 1, 2, 2, 2, 3, 3, 3])




```python
tcat.shape
```




    torch.Size([9])




```python
tstack = torch.stack((t1,t2,t3), dim = 0)
tstack
```




    tensor([[1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]])




```python
tstack.shape
```




    torch.Size([3, 3])




```python
t = torch.tensor([1,2,3])
```


```python
t
```




    tensor([1, 2, 3])




```python
t.shape
```




    torch.Size([3])




```python
t.unsqueeze(dim = 0)
```




    tensor([[1, 2, 3]])




```python
t = torch.tensor([[1,2,3,4],[5,6,7,8]])
```


```python
t.shape
```




    torch.Size([2, 4])




```python
t = torch.tensor([[[1,2],[3,4],[5,6]],[[1,2],[3,4],[5,6]]])
```


```python
t.shape
```




    torch.Size([2, 3, 2])




```python
tx = torch.tensor([1,1,1])
ty = torch.tensor([2,2,2])
tz = torch.tensor([3,3,3])
```
