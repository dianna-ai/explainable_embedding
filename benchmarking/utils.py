import os
import random

from matplotlib import pyplot as plt
import numpy as np


def load_img(path, target_size):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


class ImageNetModel:
    def __init__(self):
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras import backend as keras_backend
        self.model = ResNet50()
        self.input_size = (224, 224)

    def run_on_batch(self, x):
        return self.model.predict(x, verbose=0)


def class_name(idx):
    from tensorflow.keras.applications.resnet50 import decode_predictions
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]


def plot_saliency_map_on_image(image, saliency, ax=None, vmin=None, vmax=None, title="Explanation",
                               do_cbar=True, add_value_limits_to_title=False, central_value=None, alpha=0.5, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = ax.get_figure()
    if add_value_limits_to_title:
        if vmin is None:
            vmin_title = saliency.min()
        else:
            vmin_title = vmin
        if vmax is None:
            vmax_title = saliency.max()
        else:
            vmax_title = vmax
        title = f"{title} vmin = {vmin_title:.2f}, vmax = {vmax_title:.2f}"
    ax.set_title(title)
    ax.axis('off')
    ax.imshow(image)
    cmap = plt.cm.get_cmap('bwr', 9)

    if central_value is not None:
        assert vmin is not None, 'central value and vmin combination unsupported'
        assert vmax is not None, 'central value and vmax combination unsupported'

        radius = max(central_value - vmin, vmax - central_value)
        vmin = central_value - radius
        vmax = central_value + radius


    im = ax.imshow(saliency, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    if do_cbar:
        plt.colorbar(im, ax=ax)
    return fig


# from https://stackoverflow.com/a/52897216/1199693
def set_all_the_seeds(seed_value=0):
    import tensorflow as tf

    os.environ['PYTHONHASHSEED']=str(seed_value)

    random.seed(seed_value)

    np.random.seed(seed_value)

    tf.random.set_seed(seed_value)

    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
