
### CNN Batch Processing

In last notebook, we saw how to process a single image. A single image was passed to the network using `next(itet(train_Set))` which gives the next element of a data set which in this case was a 1,28,28 tensor. Now we want to be able to process batches. This will be done using the previously mentioned Data Loader instace.


```python
import torch 
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

torch.set_printoptions(linewidth=120)
```


```python
print(torch.__version__)
print(torchvision.__version__)
```

    1.2.0
    0.4.0
    


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
torch.set_grad_enabled(False)
```




    <torch.autograd.grad_mode.set_grad_enabled at 0x1be69cc48d0>




```python
n1 = Network()
```


```python
data_loader = torch.utils.data.DataLoader(
    train_set
    ,batch_size = 10
)
```


```python
batch = next(iter(data_loader)) #each batch will give us 10 images
```


```python
images,labels = batch
```


```python
images.shape # 10 28x28 images with a single color channel 
```




    torch.Size([10, 1, 28, 28])




```python
print(labels)
print(labels.shape)
```

    tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5])
    torch.Size([10])
    


```python
preds = n1(images)
```


```python
preds.shape
```




    torch.Size([10, 10])




```python
preds #each column represents the predcition for each of the 10 images
```




    tensor([[-0.1490,  0.1044,  0.1109,  0.0092,  0.0159,  0.0738, -0.0593,  0.1296,  0.0441,  0.0386],
            [-0.1613,  0.1145,  0.1210,  0.0147,  0.0051,  0.0514, -0.0373,  0.1136,  0.0578,  0.0420],
            [-0.1610,  0.1066,  0.1085, -0.0077,  0.0181,  0.0466, -0.0458,  0.1074,  0.0599,  0.0338],
            [-0.1575,  0.1027,  0.1081,  0.0028,  0.0206,  0.0449, -0.0466,  0.1137,  0.0609,  0.0350],
            [-0.1609,  0.1035,  0.1073,  0.0055,  0.0088,  0.0633, -0.0408,  0.1214,  0.0510,  0.0382],
            [-0.1623,  0.1155,  0.1260,  0.0099,  0.0119,  0.0523, -0.0396,  0.1141,  0.0544,  0.0347],
            [-0.1436,  0.1057,  0.1168,  0.0065,  0.0160,  0.0762, -0.0503,  0.1218,  0.0494,  0.0360],
            [-0.1626,  0.1221,  0.1253,  0.0163,  0.0157,  0.0577, -0.0491,  0.1234,  0.0520,  0.0386],
            [-0.1585,  0.1070,  0.1107, -0.0092,  0.0202,  0.0455, -0.0519,  0.1170,  0.0565,  0.0283],
            [-0.1602,  0.1069,  0.1233,  0.0052,  0.0184,  0.0561, -0.0417,  0.1163,  0.0531,  0.0389]])




```python
F.softmax(preds, dim = 1)
```




    tensor([[0.0832, 0.1072, 0.1079, 0.0975, 0.0981, 0.1040, 0.0910, 0.1099, 0.1009, 0.1004],
            [0.0821, 0.1082, 0.1089, 0.0980, 0.0970, 0.1016, 0.0930, 0.1081, 0.1023, 0.1007],
            [0.0826, 0.1080, 0.1082, 0.0963, 0.0988, 0.1017, 0.0927, 0.1081, 0.1031, 0.1004],
            [0.0828, 0.1074, 0.1080, 0.0972, 0.0989, 0.1014, 0.0925, 0.1086, 0.1030, 0.1004],
            [0.0824, 0.1073, 0.1077, 0.0973, 0.0976, 0.1031, 0.0929, 0.1093, 0.1018, 0.1005],
            [0.0821, 0.1084, 0.1095, 0.0975, 0.0977, 0.1018, 0.0928, 0.1082, 0.1020, 0.1000],
            [0.0835, 0.1072, 0.1084, 0.0970, 0.0980, 0.1041, 0.0917, 0.1089, 0.1013, 0.1000],
            [0.0819, 0.1088, 0.1092, 0.0979, 0.0979, 0.1021, 0.0917, 0.1090, 0.1015, 0.1001],
            [0.0828, 0.1080, 0.1084, 0.0962, 0.0991, 0.1016, 0.0922, 0.1091, 0.1027, 0.0999],
            [0.0823, 0.1075, 0.1093, 0.0971, 0.0984, 0.1022, 0.0926, 0.1085, 0.1019, 0.1004]])




```python
F.softmax(preds[3], dim = 0)
```




    tensor([0.0828, 0.1074, 0.1080, 0.0972, 0.0989, 0.1014, 0.0925, 0.1086, 0.1030, 0.1004])




```python
preds.argmax(dim = 1)
```




    tensor([7, 2, 2, 7, 7, 2, 7, 2, 7, 2])




```python
preds.argmax(dim =1).eq(labels)
```




    tensor([False, False, False, False, False,  True,  True,  True, False, False])




```python
preds.argmax(dim=1).eq(labels).sum()
```




    tensor(3)




```python
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum()
```


```python
get_num_correct(preds,labels)
```




    tensor(3)




```python

```
