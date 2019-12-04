
### Objected Oriented Neural Networks


```python
#Dummy network class
class Network:
    def __init__(self):
        self.layer = None
        
    def forward(self, t):
        t = self.layer(t)
        return t
```


```python
a = Network()
```


```python
import torch.nn as nn
```


```python
#Network class that inherits PyTorch Neural Network
class Network(nn.Module): #extending nn.Module class
    def __init__(self):
        super(Network,self).__init__() #calling the super constructor
        
        #convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=12)
        
        #linear (or fully connected or dense) layers
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        
        #output layer
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        t = self.layer(t) #dyanmic
        return t
        
```


```python
network = Network()
network
```




    Network(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 12, kernel_size=(12, 12), stride=(1, 1))
      (fc1): Linear(in_features=192, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=60, bias=True)
      (out): Linear(in_features=60, out_features=10, bias=True)
    )




```python

```
