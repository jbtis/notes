
### Passing an element through the CNN 

In this notebook, we are going to create a train_loader and a network to then pass data to it and guess our first prediction.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
```


```python
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\train-images-idx3-ubyte.gz
    

    26427392it [00:04, 5517375.26it/s]                              
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\train-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\train-labels-idx1-ubyte.gz
    

    32768it [00:00, 104228.44it/s]           
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
    

    4423680it [00:01, 4011432.38it/s]                            
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
    

    8192it [00:00, 38994.86it/s]            
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Processing...
    Done!
    


```python
class Network(nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
        
    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t,kernel_size = 2, stride = 2)
        
        t = F.relu(self.fc1(t.reshape(-1,12*4*4)))
        t = F.relu(self.fc2(t))
        t = self.out(t) 
        
        return t
```


```python
torch.set_grad_enabled(False) #disable dynamic graph computation
```




    <torch.autograd.grad_mode.set_grad_enabled at 0x15de9afb860>




```python
network = Network()
```


```python
sample = next(iter(train_set))
```


```python
image, label = sample
image.shape
```




    torch.Size([1, 28, 28])




```python
image.unsqueeze(0).shape #create 4 dimsensional tensor since methods in pytorch require 4 dimensional tensors 
```




    torch.Size([1, 1, 28, 28])




```python
pred = network(image.unsqueeze(0))
```


```python
pred.shape #output if the network
```




    torch.Size([1, 10])




```python
pred
```




    tensor([[-0.0069, -0.0037, -0.0841, -0.0337, -0.0479, -0.0970,  0.0096, -0.0091,  0.0961,  0.0740]])




```python
label
```




    9




```python
pred.argmax(dim = 1) #he network predited tensor[8]
```




    tensor([8])




```python
F.softmax(pred, dim = 1) #preditions in terms of probabilities
```




    tensor([[0.1002, 0.1005, 0.0927, 0.0975, 0.0961, 0.0915, 0.1018, 0.0999, 0.1110, 0.1086]])




```python
F.softmax(pred, dim = 1).sum()
```




    tensor(1.0000)



Note that the initial predcitons of the networks will be different depending on he instance since first weights are generated randomly


```python
net1 = Network()
```


```python
net1(image.unsqueeze(0))
```




    tensor([[-0.0619, -0.0789,  0.0691,  0.0337,  0.0321, -0.0345, -0.1699,  0.0241,  0.1406, -0.2004]])




```python
net2 = Network()
```


```python
net2(image.unsqueeze(0))
```




    tensor([[ 0.0676, -0.0083, -0.1249, -0.0877,  0.0150, -0.0110,  0.0703, -0.0950, -0.0898,  0.0369]])




```python

```
