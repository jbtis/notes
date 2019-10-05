pytorch -> thin framewrok, close to progrmming nn from scratch.

TensorFlow -> Keras ? -> Really abstracted

Dynamic Graphs vs. Static Graphs -> PyTorch has Dynamic Graphs
Research greatly benefits from computational graphs
Computational graphs are used to graph the functions that occur inside the neural networks


CUDA
GPU - gaphics processing unit - good at specialized computations

Parallel computing
computation -> indpendent smaller computation -> synchronez lter to mak larger compuation
GPUS -> thousands of cores!!!

Task parallel -> parallel programming apporaches -> gpu

Neural Networks are -> Embarrasingly paralel -> perfectly parallel, no need to separate the tasks
Convolution is parallel

CUDA software (API) for NVDIDIA gpus


pytorch critical functions in c/c++


Use CUDA thorugh PyTorch
t = torch.tensor([1,2,3])
t
t = t.cuda()
t

moving data from cpu to gpu -> costly
we dont need to always compute using gpu if the amount of workload is small, just use cpu

GPGPU computing

GPU stack: GPU ardware -> CUDA softare architecture -> libraries (cudnn) -> frameworks (pytorch) -> apps

------------------------------
number,array,2d array, nd-array -> cs
scalar,vector,matrix,  nd-tensor -> math
  0       1      2   ,  n-> indices required to refer to a specific elements

in ml just use nd-tensors for everything!

the dimensions of a tensor dosent tell us how many components exist inside

tensor attributes
rank: number of dimensions of a tensor (how many indices needed to access specific element within the tensor) (also length of shape)
axes: specific dimension of a tensor, length of axes: how many elements available in that axis, elements of the last axes are always numbers
every other axes will contain a ndimensional array
shape: determined by the length of each axis, encodes all relevant information of a tensor! [x,y,z,q] -> last axis has the acual elements

reshapnig vectors
-data stays the same
-regroup data
-shape reveals number of elemetns in the tensor

------------------------------------
tensor input [B,C,H,W]
batch size, color channel, height width 
after filter of cnn we now have feature maps (each filter represents particular filters from the image) (no longer color channels)
------------------------------------
Data pre processing

tensors contain uniform numerical data (tensor operations only happen with tensor with same data type)
tensor operations must happen between operations in the same device
stride - default

attributes:
data uniform type
device
arrangment

tensor can be created using existing data or by functions such us torch.rand(4,4)

-----------------------------------------
in a fully connected layer, input is flatenned first

---------------------------------------------
element wise operations: 
Operation between elements of two different tensors that are corresponding elements
corresponding elemetns are elements that ocuppy the same index position



