import pathlib

import matplotlib.pyplot as plt

from benchmarking.distance_benchmark import run_image_vs_image_experiment, imagenet_cases, log_git_versions, plot_saliency_map_on_image, run_image_captioning_experiment, image_captioning_cases
from benchmarking.Config import original_config_options as config


image_size = 224
half_image_size = image_size / 2

fancy_figure_kwargs = {
    # much fun with DPI, column width and font size (and font type of course!)
    # ... once we know these things
    'alpha': 0.7
}

base_output_folder = pathlib.Path('paper_figures')


def make_image_vs_image_figure():
    fig, ax = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")

    for ix, case in enumerate(imagenet_cases):
        output_folder = base_output_folder / f'{config.experiment_name}'
        output_folder.mkdir(exist_ok=True, parents=True)
        config.to_yaml_file(output_folder / 'config.yml')

        log_git_versions(output_folder)

        case_folder = output_folder / f'image_vs_image_{case.name}'
        case_folder.mkdir(exist_ok=True, parents=True)
        saliency, central_value, input_image = run_image_vs_image_experiment(case, config, case_folder, analyse=False)

        ax_ix = ax.flatten()[ix]

        plot_saliency_map_on_image(input_image, saliency[0], ax=ax_ix,
                                   title="", add_value_limits_to_title=False,
                                   vmin=saliency[0].min(), vmax=saliency[0].max(),
                                   central_value=central_value, **fancy_figure_kwargs)
    
        ax_ix.text(half_image_size, image_size + 20, case.name,
                    horizontalalignment='center', verticalalignment='center')

    fig.savefig(base_output_folder / 'item-pair_image_vs_image_gallery.pdf')


def make_image_vs_caption_figure():
    fig, ax = plt.subplots(2, 5, figsize=(12, 4), layout="constrained")

    for ix, case in enumerate(image_captioning_cases):
        output_folder = base_output_folder / f'{config.experiment_name}'
        output_folder.mkdir(exist_ok=True, parents=True)
        config.to_yaml_file(output_folder / 'config.yml')

        log_git_versions(output_folder)

        case_folder = output_folder / f'image_vs_caption_{case.name}'
        case_folder.mkdir(exist_ok=True, parents=True)
        saliency, central_value, input_image = run_image_captioning_experiment(case, config, case_folder, analyse=False)

        ax_ix = ax.flatten()[ix]

        plot_saliency_map_on_image(input_image, saliency[0], ax=ax_ix,
                                   title="", add_value_limits_to_title=False,
                                   vmin=saliency[0].min(), vmax=saliency[0].max(),
                                   central_value=central_value, **fancy_figure_kwargs)
    
        ax_ix.text(half_image_size, image_size + 20, case.name,
                    horizontalalignment='center', verticalalignment='center')

    fig.savefig(base_output_folder / 'item-pair_image_vs_caption_gallery.pdf')



if __name__ == '__main__':
    make_image_vs_image_figure()
    make_image_vs_caption_figure()
