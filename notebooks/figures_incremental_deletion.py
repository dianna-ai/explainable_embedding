#!/usr/bin/env python
# coding: utf-8

# script based on 05_metrics_incremental_deletion_for_distance_bee_wrt_fly.ipynb
# and 06_metrics_incremental_deletion_for_distance_bee_wrt_bee2.ipynb

import warnings
import time
import pickle
import numpy as np
from pathlib import Path
# keras model and preprocessing tools
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras import backend as K
from keras import utils
# dianna library for explanation
from dianna import visualization
# for plotting
from matplotlib import pyplot as plt
from distance_explainer import DistanceExplainer
from sklearn.metrics import pairwise_distances
# local helper module
import distance_metrics

warnings.filterwarnings('ignore') # disable warnings relateds to versions of tf


class Model():
    def __init__(self):
        K.set_learning_phase(0)
        self.model = ResNet50()
        self.input_size = (224, 224)

    def run_on_batch(self, x):
        return self.model.predict(x, verbose=0)


def load_img(path, model):
    img = utils.load_img(path, target_size=model.input_size)
    x = utils.img_to_array(img)
    x = preprocess_input(x)
    return img, x

def to_img(x):
    z = np.copy(x)
    z = preprocess_input(z)
    img = utils.array_to_img(z)
    return img

def class_name(idx):
    return decode_predictions(np.eye(1, 1000, idx))[0][0][1]

def generate_examples(img_data, salient_order, impute_value, model, embedded_reference, step=5500, n_examples=5):
    '''Helper function to generate examples for removing pixels.
    '''
    n_removed = []
    scores = []
    imgs = []

    embedded_original_image = model.model.predict(img_data[None, ...])
    original_distance = (pairwise_distances(embedded_original_image, embedded_reference,  # copied from distance_explainer
                                       metric='cosine') / 2)[0,0]

    imputed_image = np.copy(img_data)
    for k in range(n_examples):
        i, j = zip(*salient_order[k * step: (k + 1) * step])
        imputed_image[i, j] = impute_value

        prediction = model.model.predict(imputed_image[None, ...])
        embedded_img = prediction

        distance = (pairwise_distances(embedded_img, embedded_reference,  # copied from distance_explainer
                                       metric='cosine') / 2)[0,0]

        score = (distance / original_distance - 1) * 100

        # score = prediction[label]
        scores.append(score)
        imgs.append(to_img(imputed_image))
        n_removed.append((k + 1) * step)

    return imgs, scores, n_removed


def make_figures(reference_img_filename: str, output_folder: Path):
    fn_tag = reference_img_filename.split('.')[0]

    np.random.seed(0)
    model = Model()

    img, img_data = load_img(Path('.') / 'data' / 'bee.jpg', model)

    reference_img, reference_img_data = load_img(Path('.') / 'data' / reference_img_filename, model)

    embedded_reference = model.run_on_batch(np.expand_dims(reference_img_data, 0))

    attribution_filepath = output_folder / f"attribution_{fn_tag}.npy"

    if attribution_filepath.exists():
        attribution = np.load(attribution_filepath)
    else:
        attribution, _ = DistanceExplainer(axis_labels=['x','y','channels']).explain_image_distance(model.run_on_batch, img_data, embedded_reference)
        np.save(attribution_filepath, attribution)

    salience_map = attribution[0]

    _ = visualization.plot_image(salience_map, utils.img_to_array(img) / 255., heatmap_cmap='jet', show_plot=False)

    examples_dirpath = output_folder / f"{fn_tag}_generated_examples"

    if examples_dirpath.exists():
        imgs = [x for x in np.load(examples_dirpath / "imgs.npz").values()]
        preds = [x for x in np.load(examples_dirpath / "preds.npz").values()]
        nremoved = [x for x in np.load(examples_dirpath / "nremoved.npz").values()]
        imgs_reverse = [x for x in np.load(examples_dirpath / "imgs_reverse.npz").values()]
        preds_reverse = [x for x in np.load(examples_dirpath / "preds_reverse.npz").values()]
        nremoved_reverse = [x for x in np.load(examples_dirpath / "nremoved_reverse.npz").values()]
        rand_imgs = [x for x in np.load(examples_dirpath / "rand_imgs.npz").values()]
        rand_preds = [x for x in np.load(examples_dirpath / "rand_preds.npz").values()]
        rand_nremoved = [x for x in np.load(examples_dirpath / "rand_nremoved.npz").values()]
    else:
        start_time = time.time()
        print("generating examples...")

        _salient_order = np.stack(np.unravel_index(np.argsort(salience_map, axis=None),
                                  salience_map.shape[:2]), axis=1)[::-1] # Get indices after sorting attribution ("relevances" in earlier notebooks)
        channel_mean = np.mean(img_data, axis=(0,1))

        imgs, preds, nremoved = generate_examples(img_data, _salient_order, channel_mean, model, embedded_reference)
        imgs_reverse, preds_reverse, nremoved_reverse = generate_examples(img_data, _salient_order[::-1], channel_mean, model, embedded_reference)

        # Remove at random
        np.random.shuffle(_salient_order)
        rand_imgs, rand_preds, rand_nremoved = generate_examples(img_data, _salient_order, channel_mean, model, embedded_reference)
        del _salient_order

        examples_dirpath.mkdir(exist_ok=True, parents=True)
        np.savez(examples_dirpath / "imgs.npz", *imgs)
        np.savez(examples_dirpath / "preds.npz", *preds)
        np.savez(examples_dirpath / "nremoved.npz", *nremoved)
        np.savez(examples_dirpath / "imgs_reverse.npz", *imgs_reverse)
        np.savez(examples_dirpath / "preds_reverse.npz", *preds_reverse)
        np.savez(examples_dirpath / "nremoved_reverse.npz", *nremoved_reverse)
        np.savez(examples_dirpath / "rand_imgs.npz", *rand_imgs)
        np.savez(examples_dirpath / "rand_preds.npz", *rand_preds)
        np.savez(examples_dirpath / "rand_nremoved.npz", *rand_nremoved)

        elapsed_time = time.time() - start_time
        print(f"... done generating examples, took {elapsed_time} seconds")

    # We visualize the effect of deleting pixels in order of RISE vs a random order below. The method of deletion was replacing pixels with the mean color in the image.

    # Visualize the examples
    npixels = img_data.shape[0] * img_data.shape[1]
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 7), layout="constrained")
    half_image_size = img_data.shape[0]/2

    for i, (im, rand_img, img_reverse, removed) in enumerate(zip(imgs, rand_imgs, imgs_reverse, nremoved)):
        column = axes[:, i]

        for a in column:
            a.axis('off')

        column[0].set_title("$\delta$ distance: {:.2f}%".format(preds[i]))
        column[0].imshow(im)
        column[1].set_title("$\delta$ distance: {:.2f}%".format(preds_reverse[i]))
        column[1].imshow(img_reverse)
        column[2].set_title("$\delta$ distance: {:.2f}%".format(rand_preds[i]))
        column[2].imshow(rand_img)

        column[2].text(half_image_size, img_data.shape[0] + 20,
                       'Removed: {:.2f}'.format(removed / npixels),
                       horizontalalignment='center', verticalalignment='center')

    axes[0, 0].text(-20, half_image_size, 'nearing pixels first',
                    horizontalalignment='center', verticalalignment='center', rotation=90)
    axes[1, 0].text(-20, half_image_size, 'distancing pixels first',
                    horizontalalignment='center', verticalalignment='center', rotation=90)
    axes[2, 0].text(-20, half_image_size, 'random pixels',
                    horizontalalignment='center', verticalalignment='center', rotation=90)

    fig.suptitle(f"Distance to image '{fn_tag}' after removing pixels", size=20)
    fig.savefig(output_folder / f"figure_bee_vs_{fn_tag}_deleted_pixels.pdf", dpi=200)

    # #### 4 - Evaluation and Visualization using Incremental Deletion
    # We now introduce our metric `Incremental_deletion` and call its visualize method to show the correctness of the explantion. Incremental deletion expects the model used for inference and `step` which defines the amount of pixels to delete per iteration.

    deleter_dirpath = output_folder / f"{fn_tag}_deleter"

    if deleter_dirpath.exists():
        with open(deleter_dirpath / "results.pkl", 'rb') as fh:
            results = pickle.load(fh)
        with open(deleter_dirpath / "results_reversed.pkl", 'rb') as fh:
            results_reversed = pickle.load(fh)
    else:
        start_time = time.time()
        print("running DistanceIncrementalDeletion deleter...")

        deleter = distance_metrics.DistanceIncrementalDeletion(model.model.predict, reference_img_data, 224)  # Run this and see your system burning

        results = deleter(img_data, salience_map[None, ..., 0], 10, verbose=0)
        results_reversed = deleter(img_data, -salience_map[None, ..., 0], 10, verbose=0)
        
        deleter_dirpath.mkdir(exist_ok=True, parents=True)
        with open(deleter_dirpath / "results.pkl", "wb") as fh:
            pickle.dump(results, fh)
        with open(deleter_dirpath / "results_reversed.pkl", "wb") as fh:
            pickle.dump(results_reversed, fh)

        elapsed_time = time.time() - start_time
        print(f"...done running deleter, took {elapsed_time} seconds")

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout='constrained')
    distance_metrics.plot_deletion_curves(ax, (np.array(results['salient_scores'][0]), np.array(results['random_scores'][0])),
                                          ('MoRF', 'RaRF'))
    fig.savefig(output_folder / f"figure_bee_vs_{fn_tag}_MoRFvsRaRF.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(7, 7), layout='constrained')
    distance_metrics.plot_deletion_curves(ax, (np.array(results_reversed['salient_scores'][0]), np.array(results_reversed['random_scores'][0])),
                                          ('LeRF', 'RaRF'))
    fig.savefig(output_folder / f"figure_bee_vs_{fn_tag}_LeRFvsRaRF.pdf")


if __name__ == "__main__":
    output_folder = Path("figures_incremental_deletion")
    output_folder.mkdir(exist_ok=True, parents=True)
    make_figures('fly.jpg', output_folder)
    make_figures('bee2.jpg', output_folder)
