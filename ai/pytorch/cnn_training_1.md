
### CNN Training 

#### Training Process

* Get batch from training set
* Pass batch through network
* Calculate loss (difference between true values and predicted values) `Loss Function`
* Calculate the gradient of the loss function w.r.t network weights `Back Propagation`
* Update the weights using the gradients to reduce the loss `Optimization Algorithm`
* Repeat untill epoch is completed
* Repeat untill desired accuracy is reached


```python
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth = 120)
torch.set_grad_enabled(True) #already enabled by default
```




    <torch.autograd.grad_mode.set_grad_enabled at 0x1fc30e6a400>




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
network = Network()
```


```python
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FasionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
```


```python
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
batch = next(iter(train_loader))
images, labels = batch
```

#### Calculating the loss 


```python
preds = network(images)
loss = F.cross_entropy(preds,labels)
loss.item() 
#as we train our nn, this loss should decrease
```




    2.303912878036499



#### Calculating the gradient 


```python
print(network.conv1.weight.grad)
```

    None
    


```python
loss.backward() #calculating the gradient
```


```python
network.conv1.weight.grad.shape #same shape as weight tensor. For each weight element there is a corresponding gradient 
```




    torch.Size([6, 1, 5, 5])




```python
network.conv1.weight.grad #this gradients will be used by the optimizer to update the weights
```




    tensor([[[[ 1.6573e-03,  1.0300e-03,  9.3414e-05,  2.0863e-04, -2.1120e-04],
              [ 1.7717e-03,  9.4615e-04,  6.2268e-05,  1.2607e-04, -3.4146e-04],
              [ 1.9514e-03,  7.0665e-04, -9.7001e-05, -2.2121e-04, -1.9728e-04],
              [ 2.0385e-03,  1.0743e-03,  3.3273e-04,  1.9201e-04, -1.3389e-04],
              [ 2.3796e-03,  1.7755e-03,  4.7150e-04,  2.4722e-04,  2.2123e-04]]],
    
    
            [[[-9.1514e-05, -1.5591e-04, -5.0517e-05, -2.7421e-05, -2.6105e-05],
              [-1.2750e-04, -1.4445e-04, -4.2162e-05, -5.0695e-05,  1.2313e-05],
              [-2.8051e-05, -1.4134e-04, -4.1034e-05, -7.6081e-06,  4.3479e-05],
              [-2.9813e-05, -9.1014e-05, -5.2956e-05, -1.0106e-04, -6.6471e-05],
              [ 4.2697e-05, -1.9406e-05, -2.9743e-04, -3.3462e-04, -4.0435e-04]]],
    
    
            [[[ 3.1496e-04,  3.5055e-04,  4.2023e-04,  3.2901e-04,  4.8517e-04],
              [ 4.2878e-04,  1.7776e-04,  3.3726e-05,  1.0961e-04,  3.1113e-04],
              [ 2.0817e-04,  1.3810e-04,  1.9827e-04,  8.0670e-05,  4.6385e-04],
              [ 3.4902e-04,  2.1116e-04,  1.8901e-04,  4.8843e-05,  4.3539e-04],
              [ 3.7972e-04,  3.8806e-04,  1.4596e-04,  1.8787e-04,  4.3308e-04]]],
    
    
            [[[-1.0687e-03, -1.3712e-03, -8.3067e-04,  7.1200e-04,  1.4980e-03],
              [-1.0997e-03, -1.4544e-03, -1.0865e-03,  5.2323e-04,  1.4627e-03],
              [-1.1858e-03, -1.4822e-03, -9.8289e-04,  7.4243e-04,  1.2932e-03],
              [-7.9037e-04, -7.9502e-04, -3.7577e-04,  8.3486e-04,  1.5243e-03],
              [-5.2364e-04, -8.3913e-04, -5.2596e-04,  9.6610e-04,  1.5870e-03]]],
    
    
            [[[-8.0185e-04, -3.6497e-04, -1.1059e-03, -1.1179e-03, -1.3387e-03],
              [-1.0730e-03, -8.7966e-04, -8.7374e-04, -1.3701e-03, -1.6305e-03],
              [-9.1590e-04, -1.0641e-03, -8.8626e-04, -1.3856e-03, -1.5635e-03],
              [-1.2802e-03, -1.4827e-03, -1.2559e-03, -1.8421e-03, -1.7436e-03],
              [-1.4695e-03, -1.5670e-03, -1.3757e-03, -1.7106e-03, -1.8347e-03]]],
    
    
            [[[-2.4750e-03, -7.8184e-04, -1.0304e-03, -1.9920e-03, -1.2342e-03],
              [-2.4094e-03, -7.5614e-04, -6.3091e-04, -1.5513e-03, -1.2344e-03],
              [-2.0408e-03, -9.1335e-04, -4.7782e-04, -1.2468e-03, -1.0484e-03],
              [-1.7015e-03, -5.1392e-04, -3.0192e-04, -6.8138e-04, -1.0370e-03],
              [-1.2463e-03, -1.2598e-04,  6.8651e-05, -1.9667e-04, -1.1653e-03]]]])



#### Updating the weights


```python
optimizer = optim.Adam(network.parameters(), lr = 0.01) #passing network parameters (weights) and learning rate 
#note: since the network parameters are the weights, the optimizer can update the weights using .step() function
```


```python
loss.item()
```




    2.2590622901916504




```python
get_num_correct(preds, labels)
```




    15




```python
optimizer.step() #update weights (step in direction of loss fucntion minimun)
```


```python
preds = network(images)
loss = F.cross_entropy(preds, labels)
```


```python
loss.item()
```




    2.2590622901916504




```python
get_num_correct(preds, labels)
```




    15



#### Putting it all together: Training a singe batch


```python
n1 = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size = 100)
optimizer = optim.Adam(n1.parameters(), lr = 0.01) #optimizer learning rate tells how much you want to step towards the minimun

batch = next(iter(train_loader)) #getting batch
images, labels = batch #separating batch

preds = n1(images) #passing images trough network
loss = F.cross_entropy(preds,labels) #calculating loss funtion

loss.backward() #calculate gradient
optimizer.step() #update weights

#------------------------------------

print('loss1:', loss.item())
preds = network(images)
loss = F.cross_entropy(preds,labels)
print('loss2:', loss.item())


```

    loss1: 2.303184747695923
    loss2: 2.2590622901916504
    

Important note: loss.backward() is able to work due to Python's work under the hood, since it has graphs that have all the computaions.


```python

```
