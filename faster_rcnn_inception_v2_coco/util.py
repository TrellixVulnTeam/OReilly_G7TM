import logging
import os
import tarfile

import requests
import tensorflow as tf

WORK_DIRECTORY = "/tmp/pedl-mnist-estimator-work-dir"
MNIST_TF_RECORDS_FILE = "mnist-tfrecord.tar.gz"
MNIST_TF_RECORDS_URL = (
    "https://s3-us-west-2.amazonaws.com/" "determined-ai-test-data/" + MNIST_TF_RECORDS_FILE
)


def download_mnist_tfrecords() -> str:
    """
    Return the path of a directory with the MNIST dataset in TFRecord format.
    The dataset will be downloaded into WORK_DIRECTORY, if it is not already
    present.
    """
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)

    filepath = os.path.join(WORK_DIRECTORY, MNIST_TF_RECORDS_FILE)
    if not tf.gfile.Exists(filepath):
        logging.info("Downloading {}".format(MNIST_TF_RECORDS_URL))

        r = requests.get(MNIST_TF_RECORDS_URL)
        with tf.gfile.Open(filepath, "wb") as f:
            f.write(r.content)
            logging.info("Downloaded {} ({} bytes)".format(MNIST_TF_RECORDS_FILE, f.size()))

        logging.info("Extracting {} to {}".format(MNIST_TF_RECORDS_FILE, WORK_DIRECTORY))
        with tarfile.open(filepath, mode="r:gz") as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=WORK_DIRECTORY)

    data_dir = os.path.join(WORK_DIRECTORY, "mnist-tfrecord")
    assert tf.gfile.Exists(data_dir)
    return data_dir
