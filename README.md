# Deep learning at scale: tools and solutions

## Welcome

Welcome to the exercises for our O'Reilly AI tutorial! This repo will contain the the examples and exercises that are used in [our presentation at the OReilly AI Conference](https://conferences.oreilly.com/artificial-intelligence/ai-ca/public/schedule/detail/77041). You will interact with our software PEDL (Productivity Engine for Deep Learning).

## Prerequisites

In order to leverage our tutorial, we assume you have basic knowledge of deep learning. 

Please come with a laptop with a stable WiFi connection. We will provide you with access to a compute environment and computational resources on a cloud platform. The cluster address will be distributed at the doorway when you enter the room.

To interact with the cluster, you will need a browser.

Additionally, to submit jobs to the cluster and more comprehensive feature support, you will need to download and install a command-line interface (CLI). Please download materials and install the CLI in advance, as the conference WiFi bandwidth is limited. This option is recommended but not required. 

To install: please first download our [CLI Python Wheel distribution](https://github.com/determined-ai/OReilly/raw/master/pedl-0.9.3-py35.py36.py37-none-any.whl) and then follow the [instructions](https://docs.determined.ai/latest/install-cli.html) to install.

## Files in repo

- The RCNN folder adapts an open source model from the [TensorFlow Object Detection repo](https://github.com/tensorflow/models/tree/master/research/object_detection) to work with PEDL. The files there provide a wrapper to submit jobs. Example jobs, using a Faster R-CNN with Inception v2 architecture trained on the COCO dataset, can be found at the cluster address that was distributed as you entered the room. This folder is provided for illustration; please do not start experiments using files in this folder. 

- The examples folder contains code to work with the MNIST dataset during this tutorial. Please start experiments from the files in this folder. 
