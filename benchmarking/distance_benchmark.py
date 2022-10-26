import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import dianna
import git
import PIL.Image
import clip
import numpy as np
import torch
import yaml
from dianna.methods.distance import DistanceExplainer
from matplotlib import pyplot as plt

from Config import Config, original_config_options
from utils import ImageNetModel, load_img, plot_saliency_map_on_image, set_all_the_seeds


@dataclass
class ImageVsImageCase:
    name: str
    input_image_file_name: str
    reference_image_file_name: str


def run_image_vs_image_experiment(case: ImageVsImageCase, config: Config, output_folder: Path):
    model = ImageNetModel()

    input_image_path = Path(__file__).parent.parent / 'data/images/' / case.input_image_file_name
    reference_image_path = Path(__file__).parent.parent / 'data/images/' / case.reference_image_file_name
    input_image, input_arr = load_img(input_image_path, model.input_size)
    reference_image_file_name, reference_arr = load_img(reference_image_path, model.input_size)
    embedded_reference = model.run_on_batch(reference_arr)

    run_and_analyse_explainer(case.name, config, embedded_reference, input_arr[0], input_image, model.run_on_batch, output_folder)


@dataclass
class ImageCaptioningCase:
    name: str
    input_image_file_name: str
    caption: str


def run_image_captioning_experiment(case: ImageCaptioningCase, config: Config, output_folder: Path):
    # See first example at https://github.com/openai/CLIP#usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    input_image_path = Path(__file__).parent.parent / 'data/images/' / case.input_image_file_name
    input_image, input_arr = load_img(input_image_path, (224, 224))
    text = clip.tokenize([case.caption]).to(device)
    embedded_reference = model.encode_text(text).detach().numpy()

    def runner_function(x):
        lst = []
        for e in x:
            e = e[None, :]
            lst.append(model.encode_image(e).detach().numpy()[0])
        return lst

    run_and_analyse_explainer(case.name, config, embedded_reference, input_image, input_image, runner_function, output_folder,
                              preprocess_function=lambda x: [preprocess(PIL.Image.fromarray(e)) for e in x])


def run_and_analyse_explainer(case_name, config, embedded_reference, input_arr, input_image, model, output_folder,
                              preprocess_function=None):
    """

    :param case_name:
    :param config:
    :param embedded_reference:
    :param input_arr: first argument to the model
    :param input_image:
    :param model: model or function
    :param output_folder:
    :param preprocess_function:
    :return:
    """
    set_all_the_seeds(config.random_seed)

    start_time = time.time()
    explainer = DistanceExplainer(mask_selection_range_max=config.mask_selection_range_max,
                                  mask_selection_range_min=config.mask_selection_range_min,
                                  mask_selection_negative_range_max=config.mask_selection_negative_range_max,
                                  mask_selection_negative_range_min=config.mask_selection_negative_range_min,
                                  n_masks=config.number_of_masks,
                                  axis_labels={2: 'channels'},
                                  preprocess_function=preprocess_function)
    elapsed_time = time.time() - start_time
    with open(output_folder / 'elapsed_seconds.txt', 'w') as fh:
        fh.write(str(elapsed_time))

    saliency, value = explainer.explain_image_distance(model, input_arr, embedded_reference)
    central_value = value if config.manual_central_value is None else config.manual_central_value
    np.save(output_folder / 'masks.npy', explainer.masks)
    np.save(output_folder / 'predictions.npy', explainer.predictions)
    np.save(output_folder / 'saliency.npy', saliency)
    fig, ax = plt.subplots(1, 1)
    plot_saliency_map_on_image(input_image, saliency[0], ax=ax,
                               title=f'{case_name} {config.mask_selection_range_min} - {config.mask_selection_range_min}',
                               add_value_limits_to_title=True, vmin=saliency[0].min(), vmax=saliency[0].max(),
                               central_value=central_value)
    plt.savefig(output_folder / 'saliency.png')


def log_git_versions(output_folder):
    explainable_embedding_sha = git.Repo(search_parent_directories=True).head.object.hexsha
    dianna_sha = git.Repo(dianna.__path__[0], search_parent_directories=True).head.object.hexsha
    with open(output_folder / 'git_commits.txt', 'w') as file:
        file.write(f'explainable_embedding: {explainable_embedding_sha}')
        file.write(f'dianna: {dianna_sha}')


def run_benchmark(config, run_uid=None):
    if run_uid is None:
        run_uid = int(time.time())

    output_folder = Path('output') / f'{config.experiment_name}_{run_uid}'
    output_folder.mkdir(exist_ok=True, parents=True)
    with open(output_folder / 'config.yml', 'w') as file:
        yaml.dump(config, file)

    log_git_versions(output_folder)

    imagenet_cases = [ImageVsImageCase(name='bee_vs_fly',
                                       input_image_file_name='bee.jpg',
                                       reference_image_file_name='fly.jpg'),
                      ImageVsImageCase(name='labradoodles',
                                       input_image_file_name='labradoodle1.jpg',
                                       reference_image_file_name='labradoodle2.jpg'),
                      ImageVsImageCase(name='dogcar_vs_car',
                                       input_image_file_name='dogcar.jpg',
                                       reference_image_file_name='car1.jpg'),
                      ImageVsImageCase(name='dogcar_vs_dog',
                                       input_image_file_name='dogcar.jpg',
                                       reference_image_file_name='labradoodle1.jpg'),
                      ImageVsImageCase(name='flower_vs_car',
                                       input_image_file_name='flower.jpg',
                                       reference_image_file_name='car1.jpg'),
                      ImageVsImageCase(name='car_vs_bike',
                                       input_image_file_name='car2.png',
                                       reference_image_file_name='bike.jpg')]
    # imagenet_cases = []
    for imagenet_case in imagenet_cases:
        case_folder = output_folder / 'image_vs_image' / imagenet_case.name
        case_folder.mkdir(exist_ok=True, parents=True)
        run_image_vs_image_experiment(imagenet_case, config, case_folder)

    image_captioning_cases = [
        ImageCaptioningCase(name='bee image wrt a bee sitting on a flower',
                            input_image_file_name='bee.jpg',
                            caption="a bee sitting on a flower"),
        ImageCaptioningCase(name='bee image wrt a bee',
                            input_image_file_name='bee.jpg',
                            caption="a bee"),
        ImageCaptioningCase(name='bee image wrt an image of a bee',
                            input_image_file_name='bee.jpg',
                            caption="an image of a bee"),
        ImageCaptioningCase(name='bee image wrt a fly',
                            input_image_file_name='bee.jpg',
                            caption="a fly"),
        ImageCaptioningCase(name='bee image wrt a flower',
                            input_image_file_name='bee.jpg',
                            caption="a flower"),
        ImageCaptioningCase(name='labradoodles',
                            input_image_file_name='labradoodle1.jpg',
                            caption='a labradoodle'),
        ImageCaptioningCase(name='dogcar_vs_car',
                            input_image_file_name='dogcar.jpg',
                            caption='a car'),
        ImageCaptioningCase(name='dogcar_vs_dog',
                            input_image_file_name='dogcar.jpg',
                            caption='a dog'),
        ImageCaptioningCase(name='flower_vs_car',
                            input_image_file_name='flower.jpg',
                            caption='a car'),
        ImageCaptioningCase(name='car_vs_bicycle',
                            input_image_file_name='car2.png',
                            caption='a bicycle')
    ]
    for image_captioning_case in image_captioning_cases:
        case_folder = output_folder / 'image_captioning' / image_captioning_case.name
        case_folder.mkdir(exist_ok=True, parents=True)
        run_image_captioning_experiment(image_captioning_case, config, case_folder)

    # something with molecules?


run_benchmark(dataclasses.replace(original_config_options, number_of_masks=10))
