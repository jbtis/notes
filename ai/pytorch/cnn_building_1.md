
### CNN Project Preparation

1. Prepare the data
2. Build the model
3. Train the model
4. Analyze the model's results

#### First, we will focus on preparing the data 

*ETL Method*
 - Extract: Fashion MNIST data from source 
 - Transform: Put data in tensor form
 - Load: Put data in object to make it more accessible


```python
import torch
import torchvision #access to data, models, transforms, utils, etc
import torchvision.transforms as transforms
```


```python
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST', #where to store data
    train = True, #train parameter
    download = True, #download the data if not present in location specified
    transform = transforms.Compose([
        transforms.ToTensor() #transform data to tensor
    ])
)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\train-images-idx3-ubyte.gz
    

    26427392it [00:04, 6185206.12it/s]                              
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\train-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\train-labels-idx1-ubyte.gz
    

    32768it [00:00, 103645.14it/s]           
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\train-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\t10k-images-idx3-ubyte.gz
    

    4423680it [00:01, 4028324.83it/s]                            
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\t10k-images-idx3-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz
    

    8192it [00:00, 38562.47it/s]            
    

    Extracting ./data/FashionMNIST\FashionMNIST\raw\t10k-labels-idx1-ubyte.gz to ./data/FashionMNIST\FashionMNIST\raw
    Processing...
    Done!
    


```python
#wrap train_set in DataLoader instance, (batch size,thread management ,suffle and other methods now available)
train_loader = torch.utils.data.DataLoader(train_set) 
```

We completed the ETL process:
- First we extracted the data using `torchvision.datasets.FashionMNIST`
- Then we transformed the data to tensorm form using `torchvision.transforms`
- Then we wrapped our `train_set` with the DataLoader instance `train_loader`
- So now the data is ready to be feeded into the model!
