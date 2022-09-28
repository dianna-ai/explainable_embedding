import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union, Optional, Iterable

import PIL.Image
import dataconf
import dianna
import torch
import yaml
from PIL.Image import Image
from dianna import utils
from matplotlib import pyplot as plt
import clip

from utils import ImageNetModel, load_img, plot_saliency_map_on_image


@dataclass
class Config:
    experiment_name: str
    mask_selection_method: str
    mask_selection_range_min: Iterable[int]
    mask_selection_range_max: Iterable[int]
    mask_distance_power: float
    weight_range_normalization: bool
    distance_metric: str
    number_of_masks: Union[str, int]
    p_keep: Optional[float]
    feature_res: int


def run_image_vs_image_experiment(case, config: Config, output_folder: Path):
    model = ImageNetModel()

    fig, ax = plt.subplots(1, 1)

    match case:
        case 'bee_vs_fly':
            input_image_file_name = 'bee.jpg'
            reference_image_file_name = 'fly.jpg'
        case 'labradoodles':
            input_image_file_name = 'labradoodle1.jpg'
            reference_image_file_name = 'labradoodle2.jpg'
        case 'dogcar_vs_car':
            input_image_file_name = 'dogcar.jpg'
            reference_image_file_name = 'car1.jpg'
        case 'dogcar_vs_dog':
            input_image_file_name = 'dogcar.jpg'
            reference_image_file_name = 'labradoodle1.jpg'
        case 'flower_vs_car':
            input_image_file_name = 'flower.jpg'
            reference_image_file_name = 'car1.jpg'
        case 'car_vs_bike':
            input_image_file_name = 'car2.png'
            reference_image_file_name = 'bike.jpg'
        case _:
            print(f"{case} is not a valid case in this experiment.")
    explain_and_plot_image_vs_image_to_ax(input_image_file_name,
                                          reference_image_file_name,
                                          model,
                                          config,
                                          ax,
                                          title=f'{case} {config.mask_selection_range_min} - {config.mask_selection_range_min}')

    plt.savefig(output_folder / (case + '.png'))


def explain_and_plot_image_vs_image_to_ax(input_image_file_name, reference_image_file_name, model, config, ax,
                                          title: str):
    input_image_path = Path(__file__).parent.parent / 'data/images/' / input_image_file_name
    reference_image_path = Path(__file__).parent.parent / 'data/images/' / reference_image_file_name
    input_image, input_arr = load_img(input_image_path, model.input_size)
    reference_image_file_name, reference_arr = load_img(reference_image_path, model.input_size)
    embedded_reference = model.run_on_batch(reference_arr)

    saliency, central_value = dianna.explain_image_distance(model.run_on_batch, input_arr[0],
                                                            embedded_reference,
                                                            mask_selection_range_max=config.mask_selection_range_max,
                                                            mask_selection_range_min=config.mask_selection_range_min,
                                                            n_masks=config.number_of_masks,
                                                            axis_labels={2: 'channels'})
    plot_saliency_map_on_image(input_image, saliency[0], ax=ax, title=title,
                               add_value_limits_to_title=True, vmin=saliency[0].min(), vmax=saliency[0].max(),
                               central_value=central_value)


def run_image_captioning_experiment(case, config: Config, output_folder: Path):
    fig, ax = plt.subplots(1, 1)

    match case:
        case 'bee image wrt a bee sitting on a flower':
            input_image_file_name = 'bee.jpg'
            caption = "a bee sitting on a flower"
        case 'bee image wrt a bee':
            input_image_file_name = 'bee.jpg'
            caption = "a bee"
        case 'bee image wrt an image of a bee':
            input_image_file_name = 'bee.jpg'
            caption = "an image of a bee"
        case 'bee image wrt a fly':
            input_image_file_name = 'bee.jpg'
            caption = "a fly"
        case 'bee image wrt a flower':
            input_image_file_name = 'bee.jpg'
            caption = "a flower"
        case 'labradoodles':
            input_image_file_name = 'labradoodle1.jpg'
            caption = 'a labradoodle'
        case 'dogcar_vs_car':
            input_image_file_name = 'dogcar.jpg'
            caption = 'a car'
        case 'dogcar_vs_dog':
            input_image_file_name = 'dogcar.jpg'
            caption = 'a dog'
        case 'flower_vs_car':
            input_image_file_name = 'flower.jpg'
            caption = 'a car'
        case 'car_vs_bicycle':
            input_image_file_name = 'car2.png'
            caption = 'a bicycle'
        case _:
            print(f"{case} is not a valid case in this experiment.")

    title = f'{case} {config.mask_selection_range_min} - {config.mask_selection_range_min}'

    # See first example at https://github.com/openai/CLIP#usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    input_image_path = Path(__file__).parent.parent / 'data/images/' / input_image_file_name
    input_image, input_arr = load_img(input_image_path, (224, 224))
    text = clip.tokenize([caption]).to(device)
    embedded_reference = model.encode_text(text).detach().numpy()

    def runner_function(x):
        lst = []
        for e in x:
            e = e[None, :]
            lst.append(model.encode_image(e).detach().numpy()[0])
        return lst

    saliency, central_value = dianna.explain_image_distance(runner_function,
                                                            input_image,
                                                            embedded_reference,
                                                            mask_selection_range_max=config.mask_selection_range_max,
                                                            mask_selection_range_min=config.mask_selection_range_min,
                                                            n_masks=config.number_of_masks,
                                                            axis_labels={1: 'channels'},
                                                            preprocess_function=lambda x: [
                                                                preprocess(PIL.Image.fromarray(e)) for e in x])

    plot_saliency_map_on_image(input_image, saliency[0], ax=ax, title=title,
                               add_value_limits_to_title=True, vmin=saliency[0].min(), vmax=saliency[0].max(),
                               central_value=central_value)

    plt.savefig(output_folder / (case + '.png'))


def run_benchmark(config, run_uid=None):
    if run_uid is None:
        run_uid = int(time.time())

    output_folder = Path('output') / f'{config.experiment_name}_{run_uid}'
    output_folder.mkdir(exist_ok=True, parents=True)
    with open(output_folder / 'config.yml', 'w') as file:
        yaml.dump(config, file)

    imagenet_cases = ['bee_vs_fly', 'labradoodles', 'dogcar_vs_car', 'dogcar_vs_dog', 'flower_vs_car', 'car_vs_bike']
    for imagenet_case in imagenet_cases:
        run_image_vs_image_experiment(imagenet_case, config, run_uid)

    image_captioning_cases = ['bee image wrt a bee sitting on a flower',
                              'bee image wrt a bee',
                              'bee image wrt an image of a bee',
                              'bee image wrt a fly',
                              'bee image wrt a flower',
                              'labradoodles',
                              'dogcar_vs_car',
                              'dogcar_vs_dog',
                              'flower_vs_car',
                              'car_vs_bicycle', ]

    for case in image_captioning_cases:
        run_image_captioning_experiment(case, config, output_folder)

    # something with molecules?


original_config_options = Config(
    experiment_name='tune_percentages',
    mask_selection_method='',  # range, random, all
    mask_selection_range_min=0,  # 0-1
    mask_selection_range_max=0.2,  # 0-1
    mask_distance_power=1,  # 0.5, 1, 2 ... 100
    # Normalize weights to [0, 1] before taking dot product with masks.
    weight_range_normalization=False,  # True, False
    distance_metric='cosine',  # cosine, euclidian, manhatten?
    number_of_masks=2000,  # auto, [1, -> ]
    p_keep=0.5,  # None (auto), 0 - 1
    feature_res=8,  # [1, ->]
)

furthest_masks_config_options = dataclasses.replace(original_config_options,
                                                    mask_selection_range_min=.8,
                                                    mask_selection_range_max=1,
                                                    )

run_benchmark(original_config_options)
# run_benchmark(furthest_masks_config_options)
