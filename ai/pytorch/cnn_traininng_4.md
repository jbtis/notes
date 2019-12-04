
### Run Builder Class

Motivation: Handling hyperparameters in a cleaner, more organized way


```python
from collections import OrderedDict
from collections import namedtuple
from itertools import product
```


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


```python
params = OrderedDict(
    lr = [.01, .001]
    ,batch_size = [1000, 10000]
)
```


```python
runs = RunBuilder().get_runs(params)
runs
```




    [Run(lr=0.01, batch_size=1000),
     Run(lr=0.01, batch_size=10000),
     Run(lr=0.001, batch_size=1000),
     Run(lr=0.001, batch_size=10000)]




```python
print(runs[0].lr, runs[0].batch_size) #print automatically generated
```

    0.01 1000
    


```python
for run in runs:
        print(run, run.lr, run.batch_size) #automatically updates
```

    Run(lr=0.01, batch_size=1000) 0.01 1000
    Run(lr=0.01, batch_size=10000) 0.01 10000
    Run(lr=0.001, batch_size=1000) 0.001 1000
    Run(lr=0.001, batch_size=10000) 0.001 10000
    


```python

```
