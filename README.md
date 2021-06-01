[![License](https://img.shields.io/badge/License-Apache2-green.svg)](http://www.apache.org/licenses/LICENSE-2.0)

PyProf - PyTorch Profiling tool
===============================

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

Installation Instructions
-------------------------

```bash
# clone
$ git clone https://github.com/adityaiitb/PyProf.git

# install
$ pip3 install . --user

# verify
$ pip3 list | grep pyprof
```

Quick Start Instructions
------------------------

* Add the following lines to the PyTorch network you want to profile: ::

    import torch.cuda.profiler as profiler
    import pyprof
    pyprof.init()

* Profile with NVProf or Nsight Systems to generate a SQL file. ::

    $ nsys profile -f true -o net --export sqlite python net.py

* Run the parse.py script to generate the dictionary. ::
  
    $ python -m pyprof.parse net.sqlite > net.dict

* Run the prof.py script to generate the reports. ::

    $ python -m pyprof.prof --csv net.dict

Documentation
-------------

The User Guide can be found in the 
`documentation for current release 
<https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/index.html>`_, and 
provides instructions on how to install and profile with PyProf.

A complete `Quick Start Guide <https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/quickstart.html>`_ 
provides step-by-step instructions to get you quickly started using PyProf.

An `FAQ <https://docs.nvidia.com/deeplearning/frameworks/pyprof-user-guide/faqs.html>`_ provides
answers for frequently asked questions.

The `Release Notes 
<https://docs.nvidia.com/deeplearning/frameworks/pyprof-release-notes/index.html>`_
indicate the required versions of the NVIDIA Driver and CUDA, and also describe 
which GPUs are supported by PyProf

Presentation
------------
Automating End-to-End PyTorch Profiling. [Video](https://developer.nvidia.com/gtc/2020/video/s21143), [Slides](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21143-automating-end-to-end-pytorch-profiling.pdf).

Contributing
------------

Contributions to PyProf are more than welcome. To
contribute make a pull request and follow the guidelines outlined in
the [Contributing](CONTRIBUTING.md) document.

Reporting problems, asking questions
------------------------------------

We appreciate any feedback, questions or bug reporting regarding this
project. When help with code is needed, follow the process outlined in
the Stack Overflow (https://stackoverflow.com/help/mcve)
document. Ensure posted examples are:

* minimal – use as little code as possible that still produces the
  same problem

* complete – provide all parts needed to reproduce the problem. Check
  if you can strip external dependency and still show the problem. The
  less time we spend on reproducing problems the more time we have to
  fix it

* verifiable – test the code you're about to provide to make sure it
  reproduces the problem. Remove all other problems that are not
  related to your request/question.

