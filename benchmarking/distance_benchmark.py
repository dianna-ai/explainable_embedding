from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Iterable

import dianna
from matplotlib import pyplot as plt

from benchmarking.utils import ImageNetModel, load_img, plot_saliency_map_on_image


@dataclass
class Config:
    experiment_name: str
    mask_selection_method: str
    mask_selection_percentage_min: Iterable[int]
    mask_selection_percentage_max: Iterable[int]
    mask_distance_power: float
    weight_range_normalization: bool
    distance_metric: str
    number_of_masks: Union[str, int]
    p_keep: Optional[float]
    feature_res: int


def run_image_vs_image_experiment(case, config: Config, ax: plt.Axes):
    match case:
        case 'bee_vs_fly':
            output_folder = Path('output') / config.experiment_name
            model = ImageNetModel()
            input_image_file_name = 'bee.jpg'
            reference_image_file_name = 'fly.jpg'

            explain_and_plot_image_vs_image_to_ax(input_image_file_name,
                                                  reference_image_file_name,
                                                  model,
                                                  config,
                                                  ax)

            output_folder.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_folder / 'bee_vs_fly.png')
        case _:
            ...


def explain_and_plot_image_vs_image_to_ax(input_image_file_name, reference_image_file_name, model, config, ax):
    input_image_path = Path(__file__).parent.parent / 'data/images/' / input_image_file_name
    reference_image_path = Path(__file__).parent.parent / 'data/images/' / reference_image_file_name
    input_image_file_name, input_arr = load_img(input_image_path, model.input_size)
    reference_image_file_name, reference_arr = load_img(reference_image_path, model.input_size)
    percentage = config.mask_selection_percentage_min
    embedded_reference = model.run_on_batch(reference_arr)
    saliency = dianna.explain_image_distance(model.run_on_batch, input_arr[0],
                                             embedded_reference,
                                             p_keep_lowest_distances=percentage / 100,
                                             n_masks=config.number_of_masks,
                                             axis_labels={2: 'channels'})
    plot_saliency_map_on_image(input_image_file_name, saliency[0], ax=ax, title=percentage,
                               add_value_limits_to_title=True,
                               do_cbar=False)


def run_image_captioning_experiment(case, config: Config, ax):
    pass


def run_benchmark(config, ax):
    imagenet_cases = ['bee_vs_fly', 'labradoodles', 'dogcar_vs_car', 'dogcar_vs_dog', 'flower_vs_car', 'car_vs_bike']
    for imagenet_case in imagenet_cases:
        run_image_vs_image_experiment(imagenet_case, config, ax)

    image_captioning_cases = ['foo', 'bar']
    for case in image_captioning_cases:
        run_image_captioning_experiment(case, config, ax)

    # something with molecules?


config_options = Config(
    experiment_name='tune_percentages_1.0.0',
    mask_selection_method='',  # range, random, all
    mask_selection_percentage_min=20,  # 0-100
    mask_selection_percentage_max=100,  # 0-100
    mask_distance_power=1,  # 0.5, 1, 2 ... 100
    # Normalize weights to [0, 1] before taking dot product with masks.
    weight_range_normalization=False,  # True, False
    distance_metric='cosine',  # cosine, euclidian, manhatten?
    number_of_masks=1000,  # auto, [1, -> ]
    p_keep=0.5,  # None (auto), 0 - 1
    feature_res=8,  # [1, ->]
)

fig, ax = plt.subplots(1, 1)
run_benchmark(config_options, ax)
