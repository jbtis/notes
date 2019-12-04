
### Tensor batch processing (reshaping operations) 


```python
import torch
```


```python
t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])
```


```python
t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])
```


```python
t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])
```


```python
t = torch.stack((t1,t2,t3)) #3 tensors along a new axis
```


```python
t.shape #first numer represents the batch, other axis represent height and width
```




    torch.Size([3, 4, 4])




```python
t = t.reshape(3,1,4,4)
t
```




    tensor([[[[1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1],
              [1, 1, 1, 1]]],
    
    
            [[[2, 2, 2, 2],
              [2, 2, 2, 2],
              [2, 2, 2, 2],
              [2, 2, 2, 2]]],
    
    
            [[[3, 3, 3, 3],
              [3, 3, 3, 3],
              [3, 3, 3, 3],
              [3, 3, 3, 3]]]])




```python
t[0] #first image
```




    tensor([[[1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1],
             [1, 1, 1, 1]]])




```python
t[0][0] #first color-channel of first image
```




    tensor([[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]])




```python
t[0][0][0] #first row of first color channel of first image
```




    tensor([1, 1, 1, 1])




```python
t[0][0][0][0] #first pixel of first row of first color channel of first image
```




    tensor(1)



When we flatten the image, we don't want to faltten the whole tensor into 1 axis. We want to flatten each of the images of the batch.


```python
t.flatten()
```




    tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])




```python
t.flatten(start_dim =1) #wich flatten to start
```




    tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]])




```python
t.flatten(start_dim = 1).shape #3 single channel 4x4 flatenned images
```




    torch.Size([3, 16])




```python
t.shape
```




    torch.Size([3, 1, 4, 4])




```python
t.flatten(start_dim = 2)
```




    tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    
            [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
    
            [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]])




```python
t.flatten(start_dim = 2).shape
```




    torch.Size([3, 1, 16])



### Flatennig RGB Images



```python
r = torch.ones(1,2,2) #this reads as one color channel with 2x2 images
```


```python
g = torch.ones(1,2,2) + 1
```


```python
b = torch.ones(1,2,2) + 2
```


```python
img = torch.cat((r,g,b), dim = 0)
```


```python
img2 = torch.stack((r,g,b))
```


```python
img.shape
```




    torch.Size([3, 2, 2])




```python
img2.shape
```




    torch.Size([3, 1, 2, 2])




```python
x = torch.rand(2,3)
x
```




    tensor([[0.7801, 0.1428, 0.4391],
            [0.8629, 0.0271, 0.7648]])




```python
torch.cat((x,x,x), dim = 0)
```




    tensor([[0.7801, 0.1428, 0.4391],
            [0.8629, 0.0271, 0.7648],
            [0.7801, 0.1428, 0.4391],
            [0.8629, 0.0271, 0.7648],
            [0.7801, 0.1428, 0.4391],
            [0.8629, 0.0271, 0.7648]])




```python
torch.cat((x,x,x), dim = 0).shape
```




    torch.Size([6, 3])




```python
torch.cat((x,x,x), dim = 1)
```




    tensor([[0.7801, 0.1428, 0.4391, 0.7801, 0.1428, 0.4391, 0.7801, 0.1428, 0.4391],
            [0.8629, 0.0271, 0.7648, 0.8629, 0.0271, 0.7648, 0.8629, 0.0271, 0.7648]])




```python
torch.cat((x,x,x), dim = 1).shape
```




    torch.Size([2, 9])




```python
q1 = torch.ones(3,3)
q2 = torch.ones(3,3) + 1
q3 = torch.ones(3,3) + 2
```


```python
qstack = torch.stack((q1,q2,q3)) #stack creates a new axis of stacked tensors
```


```python
qstack.shape
```




    torch.Size([3, 3, 3])




```python
qcat = torch.cat((q1,q2,q3), 0) #concatanate puts the tensor in the same axis
```


```python
qcat.shape
```




    torch.Size([9, 3])




```python
qstack 
```




    tensor([[[1., 1., 1.],
             [1., 1., 1.],
             [1., 1., 1.]],
    
            [[2., 2., 2.],
             [2., 2., 2.],
             [2., 2., 2.]],
    
            [[3., 3., 3.],
             [3., 3., 3.],
             [3., 3., 3.]]])




```python
qcat
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [2., 2., 2.],
            [2., 2., 2.],
            [2., 2., 2.],
            [3., 3., 3.],
            [3., 3., 3.],
            [3., 3., 3.]])




```python
r1 = torch.ones(1,2,2)
g1 = torch.ones(1,2,2) + 1
b1 = torch.ones(1,2,2) + 2
```


```python
rgb_stack = torch.stack((r1,g1,b1))
rgb_cat = torch.cat((r1,g1,b1), dim = 0)
```


```python
rgb_stack 
```




    tensor([[[[1., 1.],
              [1., 1.]]],
    
    
            [[[2., 2.],
              [2., 2.]]],
    
    
            [[[3., 3.],
              [3., 3.]]]])




```python
rgb_cat
```




    tensor([[[1., 1.],
             [1., 1.]],
    
            [[2., 2.],
             [2., 2.]],
    
            [[3., 3.],
             [3., 3.]]])




```python
rgb_stack.shape
```




    torch.Size([3, 1, 2, 2])




```python
rgb_cat.shape
```




    torch.Size([3, 2, 2])



A length 1 axis parameter just adds new brackets wraping up all other axes


```python
torch.ones(1,2,2)
```




    tensor([[[1., 1.],
             [1., 1.]]])




```python
torch.ones(2,2)
```




    tensor([[1., 1.],
            [1., 1.]])



Applying `torch.cat` to our rgb channels gives us 3 2x2 images `(3,2,2)`. Now we want to flatten the 2x2 images, so we apply the flatten function starting from the 1st dimension (0th dimension is batch size in this case). We don't want to flatten our batch size, we just want to flatten the images. Therefore from a `(3,2,2)` we flatten to a `(3,4)`


```python
rgb_flattened = rgb_cat.flatten(start_dim = 1)
```


```python
rgb_flattened.shape
```




    torch.Size([3, 4])


