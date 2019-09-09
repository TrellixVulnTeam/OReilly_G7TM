# Tensorflow Object Detection Example

**Please do not start experiments from this directory during the tutorial.** 

It is unlikely you will see any interesting results within our time frame, while tying up the resources we've allocated for the tutorial. 

## Setup
1. Build the image with `make build`
2. Replace instances of REPLACE_WITH_PATH in yaml experiment config files
3. Download the warm start checkpoint and sample data set from [s3](https://s3.console.aws.amazon.com/s3/buckets/determined-ai-poc-data/synapse)
4. Ensure data can be bind mounted into the the container as expected in the config files


## Hyperparameters
Hyperparameters override pipeline.config values. An example of a pipeline.config is
included in data/faster_rcnn_inception_v2_coco/pipeline.config.
