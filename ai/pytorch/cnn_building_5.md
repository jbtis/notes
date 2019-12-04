

```python
import torch.nn as nn
```


```python
class Network:
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5) #each has its own sets of weights
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 5)
        
        self.fc1 = nn.Linear(in_features = 12*4*4, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 60)
        self.out = nn.Linear(in_features = 60, out_features = 10)
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) hidden convolutional layer 1
        t = self.conv1(t) #note: we dont explicit call the forward method
        t = F.relu(t) 
        
        #note: sometimes activation functions are called activation layers but for simplicity we will only call
        #layers the object that have weights encapslated in it. In this case, our non-linear activation function of choice
        #is the relu function
        
        t = F.max_pool2d(t, kernel_size =2, stride =2)
        
        # (3) hidden convolutional layer 2
        t = self.conv2(t)
        t = F.relu(t)
        F.max_pool2d(t, kernel_size=2 ,stride=2)
        
        
        # (4) hidden dense layer 1
        t = t.reshape(-1,12*4*4) #flattened tensor
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden dense layer
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output(prediction) layer
        t = self.out(t)
        # = F.softmax(t, dim = 1) cross entropy loss function will be used so no need to softmax
        
        return t
```


```python

```
