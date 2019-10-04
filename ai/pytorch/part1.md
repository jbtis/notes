### Installing PyTorch

- Download Anaconda packet manager
- In Anaconda prompt: `conda install Pytorch -c Pytorch`
- To check package: `conda list pytorch`

In Jupyter Notebooks:

```python
  import torch
  print(torch.__version__)
  torch.cuda.is_available()
  torch.version.cuda
  ```
  
  Others:
  - [nbconvert](https://ipython.org/ipython-doc/dev/notebook/nbconvert.html)
