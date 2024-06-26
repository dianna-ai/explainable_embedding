from typing import Optional, Union, Callable

import dianna
from dianna.visualization import plot_image
from matplotlib.figure import Figure
from numpy.typing import NDArray
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from metrics import IncrementalDeletion

import matplotlib.pyplot as plt


class DistanceIncrementalDeletion(IncrementalDeletion):
    def __init__(self,
                 model: Union[Callable, str],
                 reference_image: NDArray,
                 step: int,
                 n_samples: int = 1,
                 preprocess_function: Optional[Callable] = None) -> None:
        '''
        Args:
            model: The model to make predictions and that is to be explained.
            step: The amount of pixels to be deleted per iteration.
            preprocess_function: function to preprocess input for model.

        '''
        super().__init__(model, step, n_samples, preprocess_function)
        self.reference_image = reference_image


    def evaluate(self,
                 input_img: NDArray,
                 salient_order: NDArray,
                 batch_size: int = 1,
                 impute_method: Union[NDArray, float, str] = 'channel_mean',
                 **model_kwargs) -> NDArray:
        """Override of base class

        Args:
            input_img: The image to evaluate on.
            salience_map: The salient scores for input_img given the model.
            batch_size: Batch size to use for model inference.
            model_kwargs: Keyword arguments specific to the model.
        Returns:
            The model scores for each iteration of deleted pixels.
        # """

        if salient_order.shape[0] != np.prod(input_img.shape[:2]) or salient_order.shape[1] != 2:
            salient_order = self.get_salient_order(salient_order)
            raise ValueError(f'Shapes of `salient_order` {salient_order.shape} and \
                              `input_img` {input_img.shape} do not align.')

        impute_value = self._make_impute(input_img, impute_method)
        eval_img = np.copy(input_img)  # Perform deletion on a copy

        n_iters = np.prod(input_img.shape[:2]) // (self.step * batch_size)
        scores = np.empty(shape=(n_iters * batch_size + 1))
        embedded_image = self.model(eval_img[None, ...], **model_kwargs)
        embedded_reference_image = self.model(self.reference_image[None, ...], **model_kwargs)
        original_distance = (pairwise_distances(embedded_image, embedded_reference_image,  # copied from distance_explainer
                                       metric='cosine') / 2)[0, 0]

        scores[0] = 0
        for k in tqdm(range(n_iters), desc='Evaluating'):
            # Create batch and score model
            partial_order = salient_order[k * self.step * batch_size: (k + 1) * self.step * batch_size]
            batch = self._create_batch(eval_img, partial_order, impute_value, batch_size)
            embedded_batch = self.model(batch, **model_kwargs)[:]

            distances = []
            for embedded_batch_image in embedded_batch:

                distance = (pairwise_distances(embedded_batch_image[None,...], embedded_reference_image,  # copied from distance_explainer
                                               metric='cosine') / 2)[0, 0]
                distances.append(distance)


            scores[k * batch_size + 1: (k + 1) * batch_size + 1] = np.array(distances) / original_distance - 1

        return scores


def visualize(salience_map: NDArray,
              image_data: NDArray,
              scores: tuple,
              labels: tuple,
              save_to: Optional[str] = None,
              show_plot: bool = True,
              ax_image: Optional[plt.Axes] = None,
              ax_deletion: Optional[plt.Axes] = None,
              **kwargs) -> Figure:
    '''Visualize the computed scores and its AUC score.

    Args:
        scores: The model scores to be visualized
        save_to: path to save the image to
    '''
    if ax_image is None or ax_deletion is None or ax_image.get_figure() != ax_deletion.get_figure():
        fig, ax = plt.subplots(1, 2)
        ax_image = ax[0]
        ax_deletion = ax[1]
    else:
        fig = ax_image.get_figure()
    plot_image(salience_map, image_data, show_plot=False, ax=ax_image)

    # plot deletion curves
    for score, label in zip(scores, labels):
        n_steps = score.size
        x = np.arange(n_steps) / n_steps
        curve, = ax_deletion.plot(x, score)
        curve.set_label(label)
        ax_deletion.set_xlim(0, 1.)
        ax_deletion.fill_between(x, 0, score, alpha=.4)
        ax_deletion.set_title('Model score after removing fraction of pixels', **kwargs)
        ax_deletion.set_xlabel('Fraction of removed pixels', **kwargs)
        ax_deletion.set_ylabel('Model score', **kwargs)

    plt.legend()

    # Save or show
    if show_plot:
        plt.show()
    if save_to:
        fig.savefig(save_to, dpi=200)

    return fig