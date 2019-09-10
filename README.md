# Deep learning at scale: tools and solutions

## Welcome

Welcome to the exercises for our O'Reilly AI tutorial! This repo will contain the the examples and exercises that are used in [our presentation at the OReilly AI Conference](https://conferences.oreilly.com/artificial-intelligence/ai-ca/public/schedule/detail/77041). You will interact with our software PEDL (Productivity Engine for Deep Learning).

## Prerequisites

In order to leverage our tutorial, we assume you have basic knowledge of deep learning.

Please come with a laptop with a stable WiFi connection. We will provide you with access to a compute environment and computational resources on a cloud platform. The cluster address will be distributed at the doorway when you enter the room.

To watch the workloads running on the cluster and interact with the cluster, you will need a browser.

Additionally, to launch new workloads in the cluster and more comprehensive feature support, you will need to download and install a command-line interface (CLI). Please download materials and install the CLI in advance, as the conference WiFi bandwidth is limited. This option is recommended but not required.

### CLI Installation

Please first download our [CLI Python Wheel distribution](https://github.com/determined-ai/OReilly/raw/master/pedl-0.9.3-py35.py36.py37-none-any.whl) and then follow the instructions below to install:

```bash
pip install pedl-*.whl
```

We suggest installing the CLI into a [virtualenv](https://virtualenvwrapper.readthedocs.io/en/latest/install.html), although this is optional. To install the CLI into a virtualenv, first activate the virtualenv and then type the command above.

After the CLI has been installed, it should be configured to connect to the PEDL master at the appropriate IP address. This can be accomplished by setting the `PEDL_MASTER` environmental variable:

```bash
export PEDL_MASTER=<master IP>
```

More information about using the PEDL CLI can be found with the command `pedl --help`.

## Files in repo

- The `faster_rcnn_inception_v2_coco` folder adapts an open source model from the [TensorFlow Object Detection repo](https://github.com/tensorflow/models/tree/master/research/object_detection) to work with PEDL. The files there provide a wrapper to submit jobs. Example jobs, using a Faster R-CNN with Inception v2 architecture trained on the COCO dataset, can be found at the cluster address that was distributed as you entered the room. This folder is provided for illustration; **please do not start experiments using files in this folder.**

- The `reproducibility` and `hyperparameter_tuning` folders contain code to work with the MNIST dataset this tutorial. Please start experiments from the files in this folder.

- The `examples` folder contains extra example code for working with MNIST. Feel free to take a look.

## How did we do?

[Let us know!](https://forms.gle/Zcvmxq8sBfE73GoKA)
