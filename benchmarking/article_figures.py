import dataclasses
from itertools import product
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from distance_benchmark import run_image_vs_image_experiment, ImageVsImageCase, log_git_versions, plot_saliency_map_on_image
from Config import original_config_options, Config


number_of_masks_configs = {(n_mask, seed): dataclasses.replace(original_config_options,
                                               experiment_name=f'n_masks_{n_mask}_seed_{seed}',
                                               random_seed=seed,
                                               number_of_masks=n_mask,
                                               ) for n_mask, seed in
                           product([100, 500, 2000], (1, 2, 3))}


def make_number_of_masks_figure():
    fancy_figure_kwargs = {
        # much fun with DPI, column width and font size (and font type of course!)
        # ... once we know these things
    }
    fig, ax = plt.subplots(3, 3, figsize=(10, 10), layout="constrained")

    for key, config in number_of_masks_configs.items():
        output_folder = pathlib.Path('paper_figures') / f'{config.experiment_name}'
        output_folder.mkdir(exist_ok=True, parents=True)
        config.to_yaml_file(output_folder / 'config.yml')

        log_git_versions(output_folder)

        imagenet_case = ImageVsImageCase(name='bee_vs_fly',
                                        input_image_file_name='bee.jpg',
                                        reference_image_file_name='fly.jpg')
        case_folder = output_folder / f'image_vs_image_{imagenet_case.name}'
        case_folder.mkdir(exist_ok=True, parents=True)
        saliency, central_value, input_image = run_image_vs_image_experiment(imagenet_case, config, case_folder, analyse=False)


    # TODO: DOE DAADWERKELIJK ALLES OP 1 IMAGE!

    plot_saliency_map_on_image(input_image, saliency[0], ax=ax,
                               title="", add_value_limits_to_title=False,
                               vmin=saliency[0].min(), vmax=saliency[0].max(),
                               central_value=central_value, **fancy_figure_kwargs)
    fig.savefig(output_folder / 'number_of_masks_convergence.pdf')


if __name__ == '__main__':
    make_number_of_masks_figure()
