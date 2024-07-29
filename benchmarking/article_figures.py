import dataclasses
from itertools import product
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from distance_benchmark import run_image_vs_image_experiment, ImageVsImageCase, log_git_versions, plot_saliency_map_on_image, run_image_captioning_experiment, ImageCaptioningCase
from Config import original_config_options, Config


image_size = 224
half_image_size = image_size / 2

fancy_figure_kwargs = {
    # much fun with DPI, column width and font size (and font type of course!)
    # ... once we know these things
    'alpha': 0.7
}

base_output_folder = pathlib.Path('paper_figures')


number_of_masks_configs = {(n_mask, seed): dataclasses.replace(original_config_options,
                                               experiment_name=f'n_masks_{n_mask}_seed_{seed}',
                                               random_seed=seed,
                                               number_of_masks=n_mask,
                                               ) for n_mask, seed in
                           product([100, 500, 2000], (1, 2, 3))}


def make_number_of_masks_figure():
    fig, ax = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")

    for ix, config in enumerate(number_of_masks_configs.values()):
        output_folder = base_output_folder / f'{config.experiment_name}'
        output_folder.mkdir(exist_ok=True, parents=True)
        config.to_yaml_file(output_folder / 'config.yml')

        log_git_versions(output_folder)

        imagenet_case = ImageVsImageCase(name='bee_vs_fly',
                                        input_image_file_name='bee.jpg',
                                        reference_image_file_name='fly.jpg')
        case_folder = output_folder / f'image_vs_image_{imagenet_case.name}'
        case_folder.mkdir(exist_ok=True, parents=True)
        saliency, central_value, input_image = run_image_vs_image_experiment(imagenet_case, config, case_folder, analyse=False)

        plot_saliency_map_on_image(input_image, saliency[0], ax=ax.flatten()[ix],
                                   title="", add_value_limits_to_title=False,
                                   vmin=saliency[0].min(), vmax=saliency[0].max(),
                                   central_value=central_value, **fancy_figure_kwargs)
    ax[2, 0].text(half_image_size, image_size + 20, 'random seed 1',
                  horizontalalignment='center', verticalalignment='center')
    ax[2, 1].text(half_image_size, image_size + 20, 'random seed 2',
                  horizontalalignment='center', verticalalignment='center')
    ax[2, 2].text(half_image_size, image_size + 20, 'random seed 3',
                  horizontalalignment='center', verticalalignment='center')

    ax[0, 0].text(-10, half_image_size, '100 masks',
                  horizontalalignment='center', verticalalignment='center', rotation=90)
    ax[1, 0].text(-10, half_image_size, '500 masks',
                  horizontalalignment='center', verticalalignment='center', rotation=90)
    ax[2, 0].text(-10, half_image_size, '2000 masks',
                  horizontalalignment='center', verticalalignment='center', rotation=90)

    fig.savefig(base_output_folder / 'number_of_masks_convergence.pdf')


p_keep_configs = [dataclasses.replace(original_config_options,
                                      experiment_name=f'p_keep_{p_keep}',
                                      p_keep=p_keep
                                      ) for p_keep in
                  [0.1, 0.3, 0.5, 0.7, 0.9]]


def make_p_keep_figures():
    image_captioning_cases = [ImageCaptioningCase(name='bee image wrt a bee',
                                                  input_image_file_name='bee.jpg',
                                                  caption="a bee"),
                              ImageCaptioningCase(name='car_vs_bicycle',
                                                  input_image_file_name='car2.png',
                                                  caption='a bicycle')
    ]

    for case in image_captioning_cases:
        fig, ax = plt.subplots(1, 5, figsize=(12, 2), layout="constrained")

        for ix, config in enumerate(p_keep_configs):
            output_folder = base_output_folder / f'{config.experiment_name}'
            output_folder.mkdir(exist_ok=True, parents=True)
            config.to_yaml_file(output_folder / 'config.yml')

            log_git_versions(output_folder)

            case_folder = output_folder / f'image_captioning_{case.name}'
            case_folder.mkdir(exist_ok=True, parents=True)
            saliency, central_value, input_image = run_image_captioning_experiment(case, config, case_folder, analyse=False)

            ax_ix = ax.flatten()[ix]
            print(saliency[0].min(), saliency[0].max())
            plot_saliency_map_on_image(input_image, saliency[0], ax=ax_ix,
                                       title="", add_value_limits_to_title=False,
                                       vmin=saliency[0].min(), vmax=saliency[0].max(),
                                       central_value=central_value, **fancy_figure_kwargs)
            if ix == 0:
                ax_ix.text(half_image_size, image_size + 20, f'$p_\\mathrm{{keep}}$: {config.p_keep}',
                        horizontalalignment='center', verticalalignment='center')
            else:
                ax_ix.text(half_image_size, image_size + 20, f'{config.p_keep}',
                        horizontalalignment='center', verticalalignment='center')

        fig.savefig(base_output_folder / f'p_keep_{case.name}.pdf')

if __name__ == '__main__':
    # make_number_of_masks_figure()
    make_p_keep_figures()
