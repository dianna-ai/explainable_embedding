import pathlib
import time
from dataclasses import dataclass
from pathlib import Path
import numpy.typing

import PIL.Image
import git
import numpy as np
from distance_explainer import DistanceExplainer
from matplotlib import pyplot as plt

from .Config import Config
from .distance_benchmark_configs import runs_20240227
from .utils import load_img, plot_saliency_map_on_image, set_all_the_seeds


@dataclass
class ImageVsImageCase:
    name: str
    input_image_file_name: str
    reference_image_file_name: str


def run_image_vs_image_experiment(case: ImageVsImageCase, config: Config, output_folder: Path, analyse=True):
    # N.B.: imports must be here to make sure the GPU is used in multiprocessing mode, especially for tensorflow 
    from .utils import ImageNetModel
    model = ImageNetModel()

    input_image_path = Path(__file__).parent.parent / 'data/images/' / case.input_image_file_name
    reference_image_path = Path(__file__).parent.parent / 'data/images/' / case.reference_image_file_name
    input_image, input_arr = load_img(input_image_path, model.input_size)
    reference_image_file_name, reference_arr = load_img(reference_image_path, model.input_size)
    embedded_reference = model.run_on_batch(reference_arr)

    return run_and_analyse_explainer(case.name, config, embedded_reference, input_arr[0], input_image, model.run_on_batch, output_folder, analyse=analyse) + (input_image,)


@dataclass
class ImageCaptioningCase:
    name: str
    input_image_file_name: str
    caption: str


def run_image_captioning_experiment(case: ImageCaptioningCase, config: Config, output_folder: Path, analyse=True):
    # N.B.: imports must be here to make sure the GPU is used in multiprocessing mode, especially for tensorflow
    import torch
    import clip
    # See first example at https://github.com/openai/CLIP#usage
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    input_image_path = Path(__file__).parent.parent / 'data/images/' / case.input_image_file_name
    # N.B.: we don't use the second output of load_img, that is preprocessed for resnet50!
    input_image, _ = load_img(input_image_path, (224, 224))
    text = clip.tokenize([case.caption]).to(device)
    embedded_reference = model.encode_text(text).detach().cpu().numpy()

    def runner_function(x):
        lst = []
        for e in x:
            e = e[None, :]
            e_tensor = torch.Tensor(e).to(device)
            lst.append(model.encode_image(e_tensor).detach().cpu().numpy()[0])
        return lst

    return run_and_analyse_explainer(
        case.name,
        config,
        embedded_reference,
        np.array(input_image),
        input_image,
        runner_function,
        output_folder,
        preprocess_function=lambda x: [
            preprocess(PIL.Image.fromarray(e)) for e in x
        ],
        analyse=analyse,
    ) + (input_image,)


def run_explainer(case_name, config: Config, embedded_reference, input_arr, model, output_folder: Path,
                   preprocess_function=None) -> tuple[numpy.typing.NDArray, float]:
    print(f"running explainer for case {case_name} with config:")
    print(config)
    set_all_the_seeds(config.random_seed)

    start_time = time.time()
    explainer = DistanceExplainer(mask_selection_range_max=config.mask_selection_range_max,
                                  mask_selection_range_min=config.mask_selection_range_min,
                                  mask_selection_negative_range_max=config.mask_selection_negative_range_max,
                                  mask_selection_negative_range_min=config.mask_selection_negative_range_min,
                                  n_masks=config.number_of_masks,
                                  axis_labels={2: 'channels'},
                                  preprocess_function=preprocess_function,
                                  feature_res=config.feature_res,
                                  p_keep=config.p_keep)
    timing_filepath = output_folder / 'elapsed_seconds.txt'
    mask_filepath = output_folder / 'masks_first_ten.npy'
    predictions_filepath = output_folder / 'predictions.npy'
    saliency_filepath = output_folder / 'saliency.npy'
    explainer_neutral_value_filepath = output_folder / 'explainer_neutral_value.txt'
    statistics_filepath = output_folder / 'statistics.txt'

    if (
        timing_filepath.exists()
        and mask_filepath.exists()
        and predictions_filepath.exists()
        and saliency_filepath.exists()
        and statistics_filepath.exists()
        and explainer_neutral_value_filepath.exists()
    ):
        saliency = np.load(saliency_filepath)
        explainer_neutral_value = float(np.loadtxt(explainer_neutral_value_filepath))
    else:
        saliency, explainer_neutral_value = explainer.explain_image_distance(model, input_arr, embedded_reference)

        elapsed_time = time.time() - start_time
        with open(timing_filepath, 'w') as fh:
            fh.write(str(elapsed_time))
        np.save(mask_filepath, explainer.masks[:10])
        np.save(predictions_filepath, explainer.predictions)
        np.save(saliency_filepath, saliency)
        with open(statistics_filepath, 'w') as fh:
            fh.write(explainer.statistics)
        np.savetxt(explainer_neutral_value_filepath, [explainer_neutral_value])

    return saliency, explainer_neutral_value


def run_and_analyse_explainer(case_name, config: Config, embedded_reference, input_arr, input_image, model, output_folder: Path,
                              preprocess_function=None, analyse=True) -> tuple[numpy.typing.NDArray, float]:
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

    saliency, explainer_neutral_value = run_explainer(case_name, config, embedded_reference, input_arr, model, output_folder, preprocess_function=preprocess_function)

    if analyse:
        central_value = explainer_neutral_value if config.manual_central_value is None else config.manual_central_value
        fig, ax = plt.subplots(1, 1)
        plot_saliency_map_on_image(input_image, saliency[0], ax=ax,
                                title=f'{case_name} {config.mask_selection_range_min} - {config.mask_selection_range_min}',
                                add_value_limits_to_title=True, vmin=saliency[0].min(), vmax=saliency[0].max(),
                                central_value=central_value)
        plt.savefig(output_folder / 'saliency.png')
    return saliency, explainer_neutral_value


def log_git_versions(output_folder):
    explainable_embedding_sha = git.Repo(search_parent_directories=True).head.object.hexsha
    #dianna_sha = git.Repo(dianna.__path__[0], search_parent_directories=True).head.object.hexsha
    with open(output_folder / 'git_commits.txt', 'w') as fh:
        fh.write(f'explainable_embedding: {explainable_embedding_sha}')
        #fh.write(f'dianna: {dianna_sha}')


def run_benchmark(config, run_uid=None, image_image_cases=slice(None), image_caption_cases=slice(None)):
    if run_uid is None:
        run_uid = int(time.time())

    output_folder = Path('output') / f'{config.experiment_name}_{run_uid}'
    output_folder.mkdir(exist_ok=True, parents=True)
    config.to_yaml_file(output_folder / 'config.yml')

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
    for imagenet_case in imagenet_cases[image_image_cases]:
        case_folder = output_folder / 'image_vs_image' / imagenet_case.name
        case_folder.mkdir(exist_ok=True, parents=True)
        # we do this in a separate process because otherwise we can't get Tensorflow to give back the GPU memory for torch later on
        # process_eval = multiprocessing.Process(target=run_image_vs_image_experiment, args=(imagenet_case, config, case_folder))
        # process_eval.start()
        # process_eval.join()
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
    for image_captioning_case in image_captioning_cases[image_caption_cases]:
        case_folder = output_folder / 'image_captioning' / image_captioning_case.name
        case_folder.mkdir(exist_ok=True, parents=True)
        # we do this in a separate process because otherwise we can't get Tensorflow to give back the GPU memory for torch later on (or vice versa? anyway, this works, hopefully)
        # process_eval = multiprocessing.Process(target=run_image_captioning_experiment, args=(image_captioning_case, config, case_folder))
        # process_eval.start()
        # process_eval.join()
        run_image_captioning_experiment(image_captioning_case, config, case_folder)

    # something with molecules?


if __name__ == '__main__':
    # import dataclasses
    #run_benchmark(dataclasses.replace(test_config, number_of_masks=500))

    for run_config in runs_20240227:
        run_benchmark(run_config, image_image_cases=slice(0, 0))

    # for run_config in runs_20240227_moar_features_moar_masks:
    #     run_benchmark(run_config, image_image_cases=slice(0, 0), image_caption_cases=slice(1, 4, 2))
