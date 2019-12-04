
### Using TensorBoard with PyTorch - Deep Learning Metrics 


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth = 120)
torch.set_grad_enabled(True) #already enabled by default

from torch.utils.tensorboard import SummaryWriter
```


```python
print(torch.__version__)
print(torchvision.__version__)
```

    1.2.0
    0.4.0
    


```python
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()
```


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
        
        t = F.relu(self.fc1(t.reshape(-1,12*4*4))) #flattening the tensor
        t = F.relu(self.fc2(t))
        t = self.out(t) 
        
        return t
```


```python
train_set = torchvision.datasets.FashionMNIST(
    train = True
    ,download = True
    ,root = '/PyTorch/data'
    ,transform = transforms.Compose([
        transforms.ToTensor()
    ])
)
```


```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100, shuffle = True)
```

#### TensorBoard (Network Graph and Images) 


```python
tb = SummaryWriter()

network = Network()
images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images) #grid of images that will appear on tensorboard

tb.add_image('images', grid)
tb.add_graph(network, images)
tb.close()
```

#### Iterating with products

The motivation of iterating with a dictionary is that we only need to nest our network training model with a single for loop that iterates over the different parameter values


```python
from itertools import product
```


```python
parameters = dict(
    lr = [.01, .001]
    ,batch_size = [10,100,1000]
    ,shuffle = [True, False]
)
```


```python
param_values = [v for v in parameters.values()]
param_values
```




    [[0.01, 0.001], [10, 100, 1000], [True, False]]




```python
for lr, batch_size, shuffle in product(*param_values):
    print(lr, batch_size, shuffle)
```

    0.01 10 True
    0.01 10 False
    0.01 100 True
    0.01 100 False
    0.01 1000 True
    0.01 1000 False
    0.001 10 True
    0.001 10 False
    0.001 100 True
    0.001 100 False
    0.001 1000 True
    0.001 1000 False
    

Now we only need one for loop  instead of having nested for loops for batch_size, list, etc

#### Enhanced Training Loop


```python
network = Network()

for lr, batch_size, shuffle in product(*param_values):
    comment = f' batch_size={batch_size} lr={lr} shuffle={shuffle}'

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
optimizer = optim.Adam(network.parameters(), lr = lr)

images, labels = next(iter(train_loader))
grid = torchvision.utils.make_grid(images) 

comment = f'batch_size={batch_size} lr={lr}' #comment appended to the name of the run
tb = SummaryWriter(comment=comment)
tb.add_image('images', grid)
tb.add_graph(network, images)

for epoch in range(5): #specify how many epochs we want to train

    total_loss = 0
    total_correct = 0

    for batch in train_loader: #loop over number of batches in the train loader
        images, labels = batch

        preds = network(images) #pass batch
        loss = F.cross_entropy(preds,labels) #calculate loss

        optimizer.zero_grad() #have to clear gradients since the function backward accumulates gradients
        loss.backward() #update weights
        optimizer.step()

        total_loss += loss.item() *batch_size #make numbers comparable by multiplying by batch_size
        total_correct += get_num_correct(preds, labels)

    #tensorboard stuff 'x' is the tag
    tb.add_scalar('Loss', total_loss, epoch) 
    tb.add_scalar('Number Correct', total_correct, epoch)
    tb.add_scalar('Accuracy', total_correct/len(train_set), epoch)

    #Generalization (see explanation below to see what this loop is doing)
    for name,weight in network.named_parameters(): 
        tb.add_histogram(name, weight, epoch)
        tb.add_histogram(f'{name}.grad',weight.grad, epoch)

    tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
    tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
    tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)

    print("epoch:", epoch, "total_correct:", total_correct, "loss,", total_loss)

tb.close()
```

    epoch: 0 total_correct: 28277 loss, 92011.5836262703
    epoch: 1 total_correct: 41740 loss, 48538.254380226135
    epoch: 2 total_correct: 43926 loss, 42067.18385219574
    epoch: 3 total_correct: 45276 loss, 38587.98676729202
    epoch: 4 total_correct: 46254 loss, 36507.716953754425
    

#### Generalization explanation


```python
for name,weight in network.named_parameters():
    print(name, weight.shape)
```

    conv1.weight torch.Size([6, 1, 5, 5])
    conv1.bias torch.Size([6])
    conv2.weight torch.Size([12, 6, 5, 5])
    conv2.bias torch.Size([12])
    fc1.weight torch.Size([120, 192])
    fc1.bias torch.Size([120])
    fc2.weight torch.Size([60, 120])
    fc2.bias torch.Size([60])
    out.weight torch.Size([10, 60])
    out.bias torch.Size([10])
    


```python
for name,weight in network.named_parameters():
    print(f'{name}.grad', weight.grad.shape)
```

    conv1.weight.grad torch.Size([6, 1, 5, 5])
    conv1.bias.grad torch.Size([6])
    conv2.weight.grad torch.Size([12, 6, 5, 5])
    conv2.bias.grad torch.Size([12])
    fc1.weight.grad torch.Size([120, 192])
    fc1.bias.grad torch.Size([120])
    fc2.weight.grad torch.Size([60, 120])
    fc2.bias.grad torch.Size([60])
    out.weight.grad torch.Size([10, 60])
    out.bias.grad torch.Size([10])
    


```python

```
