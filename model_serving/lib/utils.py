import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


# Image visualization code from https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1
def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def describe_graph(graph_def, show_nodes=False):
    print("\n\nGraph Nodes")
    print('Input Feature Nodes: {}\n'.format([node.name for node in graph_def.node if node.op=='Placeholder']))

    print('Unused Nodes: {}\n'.format([node.name for node in graph_def.node if 'unused'  in node.name]))

    print('Output Nodes: {}\n'.format([node.name for node in graph_def.node if 'softmax' in node.name]))

    print('Quanitization Nodes: {}\n'.format([node.name for node in graph_def.node if 'quant' in node.name]))

    print('Constant Count: {}\n'.format(len([node for node in graph_def.node if node.op=='Const'])))

    print('Variable Count: {}\n'.format(len([node for node in graph_def.node if 'Variable' in node.op])))

    print('Identity Count: {}\n'.format(len([node for node in graph_def.node if node.op=='Identity'])))

    print('Total nodes: {}\n'.format(len(graph_def.node)))

    if show_nodes==True:
        for node in graph_def.node:
            print('Op:{} - Name: {}'.format(node.op, node.name))


def get_graph_def_from_saved_model(saved_model_dir):
    from tensorflow.python.saved_model import tag_constants
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
            session,
            tags=[tag_constants.SERVING],
            export_dir=saved_model_dir
        )

    return meta_graph_def.graph_def


def get_size(model_dir):
    pb_size = os.path.getsize(os.path.join(model_dir,'saved_model.pb'))

    variables_size = 0
    if os.path.exists(os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
        variables_size = os.path.getsize(os.path.join(model_dir,'variables/variables.data-00000-of-00001'))
        variables_size += os.path.getsize(os.path.join(model_dir,'variables/variables.index'))

    print("\nSaved Model Size")
    print("Model size: {} KB".format(round(pb_size/(1024.0),3)))
    print("Variables size: {} KB".format(round(variables_size/(1024.0),3)))
    print("Total Size: {} KB".format(round((pb_size + variables_size)/(1024.0),3)))


def get_graph_def_from_file(graph_filepath):
    from tensorflow.python import ops

    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            return graph_def
