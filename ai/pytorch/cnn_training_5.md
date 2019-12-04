
### Implementing RunBuilder and RunManager


```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict
```

#### Run Builder:


```python
class RunBuilder():
    @staticmethod
    def get_runs(params):
        
        Run = namedtuple('Run', params.keys())
        runs = []
        
        for v in product(*params.values()):
            runs.append(Run(*v))
            
        return runs
```

#### Run Manager: 


```python
class RunManager():
    def __init__(self):
        
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
        
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tb = None
        
       
    def begin_run(self, run, network, loader):
        
        self.run_start_time = time.time()
        
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment = f'{run}')
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network, images)
        
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
            
    def begin_epoch(self):
        self.epoch_start_time = time.time()

        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
            
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')

        #jupyter notebook exclusive
        clear_output(wait=True)
        display(df)
            
    def track_loss(self, loss):
        self.epoch_loss += loss.item() *self.loader.batch_size

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    def save(self, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')

        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


```

#### CNN


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
params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [100, 10000]
    ,shuffle = [True, False]
   #,num_workers = [0] #adding more workers dosent show more 
)

m = RunManager()
for run in RunBuilder.get_runs(params):
    
    network = Network()
    loader = torch.utils.data.DataLoader(train_set, batch_size = 100, shuffle = run.shuffle) #add num_workers (depends on system)
    optimizer = optim.Adam(network.parameters(), lr = run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:
            
            images = batch[0]
            labels = batch[1]
            preds = network(images) # Pass batch
            loss = F.cross_entropy(preds,labels) # Calculate loss
            optimizer.zero_grad() #Zero gradients
            loss.backward() #Calculate gradients
            optimizer.step() #Update Weights
            
            m.track_loss(loss)
            m.track_num_correct(preds, labels)
            
        m.end_epoch()
    m.end_run()
m.save('results')
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>run</th>
      <th>epoch</th>
      <th>loss</th>
      <th>accuracy</th>
      <th>epoch duration</th>
      <th>run duration</th>
      <th>lr</th>
      <th>batch_size</th>
      <th>shuffle</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0.578718</td>
      <td>0.778783</td>
      <td>34.698157</td>
      <td>37.081699</td>
      <td>0.010</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>0.380809</td>
      <td>0.860467</td>
      <td>35.002478</td>
      <td>72.314570</td>
      <td>0.010</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>0.349033</td>
      <td>0.871817</td>
      <td>34.449179</td>
      <td>106.906368</td>
      <td>0.010</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>4</td>
      <td>0.338301</td>
      <td>0.876567</td>
      <td>32.924309</td>
      <td>139.992245</td>
      <td>0.010</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0.325529</td>
      <td>0.880417</td>
      <td>38.133672</td>
      <td>178.247591</td>
      <td>0.010</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>1</td>
      <td>0.555221</td>
      <td>0.790267</td>
      <td>35.758330</td>
      <td>35.973751</td>
      <td>0.010</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2</td>
      <td>2</td>
      <td>0.378431</td>
      <td>0.859917</td>
      <td>33.609254</td>
      <td>69.735621</td>
      <td>0.010</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2</td>
      <td>3</td>
      <td>0.348769</td>
      <td>0.869800</td>
      <td>35.363923</td>
      <td>105.223217</td>
      <td>0.010</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2</td>
      <td>4</td>
      <td>0.333753</td>
      <td>0.877083</td>
      <td>43.818249</td>
      <td>149.159150</td>
      <td>0.010</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>5</td>
      <td>0.325719</td>
      <td>0.879617</td>
      <td>40.709126</td>
      <td>190.026853</td>
      <td>0.010</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3</td>
      <td>1</td>
      <td>0.556878</td>
      <td>0.788900</td>
      <td>34.849469</td>
      <td>35.191555</td>
      <td>0.010</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3</td>
      <td>2</td>
      <td>0.386786</td>
      <td>0.857700</td>
      <td>33.399672</td>
      <td>68.764762</td>
      <td>0.010</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>12</th>
      <td>3</td>
      <td>3</td>
      <td>0.354397</td>
      <td>0.868400</td>
      <td>33.520464</td>
      <td>102.434827</td>
      <td>0.010</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3</td>
      <td>4</td>
      <td>0.337233</td>
      <td>0.875633</td>
      <td>33.743012</td>
      <td>136.379302</td>
      <td>0.010</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3</td>
      <td>5</td>
      <td>0.331104</td>
      <td>0.877917</td>
      <td>33.318512</td>
      <td>169.826470</td>
      <td>0.010</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15</th>
      <td>4</td>
      <td>1</td>
      <td>0.575099</td>
      <td>0.781267</td>
      <td>34.303735</td>
      <td>34.491233</td>
      <td>0.010</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>2</td>
      <td>0.387579</td>
      <td>0.857117</td>
      <td>36.617333</td>
      <td>71.319003</td>
      <td>0.010</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>4</td>
      <td>3</td>
      <td>0.352693</td>
      <td>0.870067</td>
      <td>35.176301</td>
      <td>106.709732</td>
      <td>0.010</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4</td>
      <td>4</td>
      <td>0.341690</td>
      <td>0.873533</td>
      <td>33.426249</td>
      <td>140.266631</td>
      <td>0.010</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4</td>
      <td>5</td>
      <td>0.326774</td>
      <td>0.879117</td>
      <td>36.694730</td>
      <td>177.099990</td>
      <td>0.010</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>20</th>
      <td>5</td>
      <td>1</td>
      <td>0.826087</td>
      <td>0.683533</td>
      <td>33.882965</td>
      <td>34.108362</td>
      <td>0.001</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>21</th>
      <td>5</td>
      <td>2</td>
      <td>0.521854</td>
      <td>0.797817</td>
      <td>33.499558</td>
      <td>67.745553</td>
      <td>0.001</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5</td>
      <td>3</td>
      <td>0.447330</td>
      <td>0.834533</td>
      <td>33.764243</td>
      <td>101.642441</td>
      <td>0.001</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5</td>
      <td>4</td>
      <td>0.395488</td>
      <td>0.854783</td>
      <td>33.855573</td>
      <td>135.632651</td>
      <td>0.001</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5</td>
      <td>5</td>
      <td>0.360470</td>
      <td>0.867450</td>
      <td>33.952345</td>
      <td>169.747561</td>
      <td>0.001</td>
      <td>1000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6</td>
      <td>1</td>
      <td>0.765698</td>
      <td>0.711433</td>
      <td>33.229992</td>
      <td>33.437436</td>
      <td>0.001</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6</td>
      <td>2</td>
      <td>0.495482</td>
      <td>0.816267</td>
      <td>33.825385</td>
      <td>67.393472</td>
      <td>0.001</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>27</th>
      <td>6</td>
      <td>3</td>
      <td>0.425911</td>
      <td>0.846317</td>
      <td>33.806617</td>
      <td>101.420502</td>
      <td>0.001</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>28</th>
      <td>6</td>
      <td>4</td>
      <td>0.388968</td>
      <td>0.858433</td>
      <td>33.844954</td>
      <td>135.407076</td>
      <td>0.001</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6</td>
      <td>5</td>
      <td>0.362625</td>
      <td>0.867000</td>
      <td>33.640971</td>
      <td>169.255753</td>
      <td>0.001</td>
      <td>1000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>30</th>
      <td>7</td>
      <td>1</td>
      <td>0.785048</td>
      <td>0.707083</td>
      <td>33.280290</td>
      <td>33.487734</td>
      <td>0.001</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>31</th>
      <td>7</td>
      <td>2</td>
      <td>0.507538</td>
      <td>0.811500</td>
      <td>33.447599</td>
      <td>67.098884</td>
      <td>0.001</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>32</th>
      <td>7</td>
      <td>3</td>
      <td>0.432486</td>
      <td>0.843033</td>
      <td>33.664826</td>
      <td>100.897353</td>
      <td>0.001</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>33</th>
      <td>7</td>
      <td>4</td>
      <td>0.385979</td>
      <td>0.861867</td>
      <td>33.803067</td>
      <td>134.825087</td>
      <td>0.001</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7</td>
      <td>5</td>
      <td>0.355567</td>
      <td>0.870617</td>
      <td>34.122543</td>
      <td>169.071301</td>
      <td>0.001</td>
      <td>10000</td>
      <td>True</td>
    </tr>
    <tr>
      <th>35</th>
      <td>8</td>
      <td>1</td>
      <td>0.750059</td>
      <td>0.714850</td>
      <td>33.544860</td>
      <td>33.728368</td>
      <td>0.001</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>36</th>
      <td>8</td>
      <td>2</td>
      <td>0.497299</td>
      <td>0.811783</td>
      <td>33.460577</td>
      <td>67.335553</td>
      <td>0.001</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>37</th>
      <td>8</td>
      <td>3</td>
      <td>0.430111</td>
      <td>0.841417</td>
      <td>33.807615</td>
      <td>101.276811</td>
      <td>0.001</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>38</th>
      <td>8</td>
      <td>4</td>
      <td>0.390085</td>
      <td>0.857983</td>
      <td>32.113810</td>
      <td>133.554184</td>
      <td>0.001</td>
      <td>10000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>39</th>
      <td>8</td>
      <td>5</td>
      <td>0.361798</td>
      <td>0.867567</td>
      <td>32.182652</td>
      <td>165.892420</td>
      <td>0.001</td>
      <td>10000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



```python

```


```python

```
