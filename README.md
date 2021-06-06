[![License](https://img.shields.io/badge/License-Apache2-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)

# PyProf: PyTorch Profiling tool

PyProf is a tool that profiles and analyzes the GPU performance of PyTorch
models. PyProf aggregates kernel performance from `Nsight Systems
<https://developer.nvidia.com/nsight-systems>`_ or `NvProf
<https://developer.nvidia.com/nvidia-visual-profiler>`_ and provides the 
following additional features:

* Identifies the layer that launched a kernel: e.g. the association of 
  `ComputeOffsetsKernel` with a concrete PyTorch layer or API is not obvious.

* Identifies the tensor dimensions and precision: without knowing the tensor 
  dimensions and precision, it's impossible to reason about whether the actual 
  (silicon) kernel time is close to maximum performance of such a kernel on 
  the GPU. Knowing the tensor dimensions and precision, we can figure out the 
  FLOPs and bandwidth required by a layer, and then determine how close to 
  maximum performance the kernel is for that operation.

* Forward-backward correlation: PyProf determines what the forward pass step 
  is that resulted in the particular weight and data gradients (wgrad, dgrad), 
  which makes it possible to determine the tensor dimensions required by these
  backprop steps to assess their performance.
 
* Determines Tensor Core usage: PyProf can highlight the kernels that use 
  `Tensor Cores <https://developer.nvidia.com/tensor-cores>`_.
 
* Correlate the line in the user's code that launched a particular kernel (program trace).

## Installation

```bash
# clone
$ git clone https://github.com/adityaiitb/PyProf.git

# install
$ pip3 install . --user

# verify
$ pip3 list | grep pyprof
```

## Usage

There are four steps to the tool flow.

1. **Import library and annotate code.**

```python
import torch.cuda.profiler as profiler
import pyprof
pyprof.init()
```

Run the training / inference loop within the [PyTorch NVTX context
manager](https://pytorch.org/docs/stable/_modules/torch/autograd/profiler.html#emit_nvtx)
as shown below. In addition, you can use `profiler.start()` and
`profiler.stop()` to pick an iteration(s) for which you would like to
capture data.

```python
iters = 500
iter_to_capture = 100

# Define network, loss function, optimizer etc.

# PyTorch NVTX context manager
with torch.autograd.profiler.emit_nvtx():

    for iter in range(iters):

        if iter == iter_to_capture:
            profiler.start()

        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        if iter == iter_to_capture:
            profiler.stop()
```

2. **Profile using Nsight Systems or NVProf to obtain a SQLite3 database.**

> NVProf is currently being phased out, and it is recommended to use Nsight Systems.

#### Profile with Nsight Systems

Generate a SQLite database as follows.

```bash
$ nsys profile 
    -f true                  # Overwrite existing files
    -o net                   # Create net.qdrep (used by Nsys viewer)
    -c cudaProfilerApi       # Optional argument required for profiler start/stop
    --stop-on-range-end true # Optional argument required for profiler start/stop
    --export sqlite          # Export net.sql (similar to NVProf) 
    python net.py
```

If using `profiler.start()` and `profiler.stop()` in `net.py`, the options
`-c cudaProfilerApi --stop-on-range-end true` are required.

> If you experience slow profiling, `nsys` contains an option `-s none`
> which disables CPU sampling and significantly speeds up profiling.

#### Profile with NVProf

Generate a SQL (NVVP) file. This file can also be opened with Nvidia
Visual Profiler (NVVP).

If you used `profiler.start()` and `profiler.stop()`, then do

```bash
$ nvprof 
    -f 
    -o net.sql 
    --profile-from-start off  # Profiling start/stop inside net.py
    python net.py
```

If you did not use `profiler.start()` and `profiler.stop()`, then do

```bash
$ nvprof
    -f            # Overwrite existing file
    -o net.sql    # Create net.sql
    python net.py
```

If you get a message such as `ERR_NVGPUCTRPERM The user running
<tool_name/application_name> does not have permission to access NVIDIA
GPU Performance Counters on the target device`, follow the
steps [here](docs/hardware_counters.md).

3. **Parse the SQL file.**

Run parser on the SQL file. The output is an ASCII file. Each line
is a python dictionary which contains information about the kernel name,
duration, parameters etc. This file can be used as input to other custom
scripts as well. Nsys will create a file called net.sqlite.

```bash
$ python -m pyprof.parse net.sqlite > net.dict
```

4. **Use this information to calculate flops and bytes.**

## Advanced Usage

With some additional annotations in the model code, you can get
even richer profiling information e.g. the name of the layer, say
`BERT:Encoder_2:FFN:LayerNorm`, associated with every GPU kernel. It
is also easy to enable profiling of modules and functions with custom
`forward` and `backward` methods. One can also extend the tool to
add bytes and flops calculations for such custom functions. See
[here](./docs/advanced.md) for instructions.


## Slides and Recorded Talks

- Nvidia GPU Technology Conference (GTC), 2020: [Slides](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21143-automating-end-to-end-pytorch-profiling.pdf), [Video](https://developer.nvidia.com/gtc/2020/video/s21143).

## Citation

If you use PyProf and would like to cite us, we suggest the following.

```
@misc{nvidia-pyprof,
  author = {Agrawal, Aditya and Kolodziej, Marek},
  title = {"PyProf"},
  year = {2019},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/adityaiitb/PyProf}}
}
```

## Contributing and Reporting Issues

Contributions are more than welcome. To contribute make a pull request
and follow the guidelines [here](./docs/CONTRIBUTING.md).
