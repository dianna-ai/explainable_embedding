import time
from pathlib import Path

import PIL.Image
import clip
import numpy as np
import torch
import yaml
from dianna.methods.distance import DistanceExplainer
from matplotlib import pyplot as plt

from benchmarking.Config import Config, original_config_options
from utils import ImageNetModel, load_img, plot_saliency_map_on_image, set_all_the_seeds


def run_image_vs_image_experiment(case, config: Config, output_folder: Path):
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

    model = ImageNetModel()

    input_image_path = Path(__file__).parent.parent / 'data/images/' / input_image_file_name
    reference_image_path = Path(__file__).parent.parent / 'data/images/' / reference_image_file_name
    input_image, input_arr = load_img(input_image_path, model.input_size)
    reference_image_file_name, reference_arr = load_img(reference_image_path, model.input_size)
    embedded_reference = model.run_on_batch(reference_arr)

    run_and_analyse_explainer(case, config, embedded_reference, input_arr, input_image, model, output_folder)


def run_and_analyse_explainer(case, config, embedded_reference, input_arr, input_image, model, output_folder,
                              preprocess_function=None):
    set_all_the_seeds(config.random_seed)
    explainer = DistanceExplainer(mask_selection_range_max=config.mask_selection_range_max,
                                  mask_selection_range_min=config.mask_selection_range_min,
                                  mask_selection_negative_range_max=config.mask_selection_negative_range_max,
                                  mask_selection_negative_range_min=config.mask_selection_negative_range_min,
                                  n_masks=config.number_of_masks,
                                  axis_labels={2: 'channels'},
                                  preprocess_function=preprocess_function)
    saliency, value = explainer.explain_image_distance(model.run_on_batch, input_arr[0], embedded_reference)
    central_value = value if config.manual_central_value is None else config.manual_central_value
    fig, ax = plt.subplots(1, 1)
    np.save(output_folder / (case + '_saliency.npy'), saliency)
    plot_saliency_map_on_image(input_image, saliency[0], ax=ax,
                               title=f'{case} {config.mask_selection_range_min} - {config.mask_selection_range_min}',
                               add_value_limits_to_title=True, vmin=saliency[0].min(), vmax=saliency[0].max(),
                               central_value=central_value)
    plt.savefig(output_folder / (case + '.png'))


def run_image_captioning_experiment(case, config: Config, output_folder: Path):
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

    # See first example at https://github.com/openai/CLIP#usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    input_image_path = Path(__file__).parent.parent / 'data/images/' / input_image_file_name
    input_image, input_arr = load_img(input_image_path, (224, 224))
    text = clip.tokenize([caption]).to(device)
    embedded_reference = model.encode_text(text).detach().numpy()

    run_and_analyse_explainer(case, config, embedded_reference, input_arr, input_image, model, output_folder,
                              preprocess_function=lambda x: [preprocess(PIL.Image.fromarray(e)) for e in x])


def run_benchmark(config, run_uid=None):
    if run_uid is None:
        run_uid = int(time.time())

    output_folder = Path('output') / f'{config.experiment_name}_{run_uid}'
    output_folder.mkdir(exist_ok=True, parents=True)
    with open(output_folder / 'config.yml', 'w') as file:
        yaml.dump(config, file)

    imagenet_cases = ['bee_vs_fly', 'labradoodles', 'dogcar_vs_car', 'dogcar_vs_dog', 'flower_vs_car', 'car_vs_bike']
    imagenet_cases = []
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

    image_captioning_cases = ['bee image wrt a bee']

    for case in image_captioning_cases:
        run_image_captioning_experiment(case, config, output_folder)

    # something with molecules?


run_benchmark(original_config_options)
# run_benchmark(furthest_masks_config_options)
