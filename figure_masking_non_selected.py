import dataclasses
from itertools import product
import pathlib

import matplotlib.pyplot as plt

from benchmarking.distance_benchmark import run_image_vs_image_experiment, ImageVsImageCase, log_git_versions, plot_saliency_map_on_image, run_image_captioning_experiment, ImageCaptioningCase
from benchmarking.Config import original_config_options, Config


imagenet_case = ImageVsImageCase(name='bee_vs_fly',
                                 input_image_file_name='bee.jpg',
                                 reference_image_file_name='fly.jpg')

image_size = 224
half_image_size = image_size / 2

fancy_figure_kwargs = {
    # much fun with DPI, column width and font size (and font type of course!)
    # ... once we know these things
    'alpha': 0.7
}

base_output_folder = pathlib.Path('paper_figures')


mask_nonselect_configs = [dataclasses.replace(original_config_options,
                                              experiment_name=f'mask_non_selection_neg{neg_min}-{neg_max}_pos{pos_min}-{pos_max}',
                                              mask_selection_range_max=pos_max,
                                              mask_selection_range_min=pos_min,
                                              mask_selection_negative_range_max=neg_max,
                                              mask_selection_negative_range_min=neg_min,
                                              ) for pos_min, pos_max, neg_min, neg_max in
                          [
                              (0.4, 0.5, 0.5, 0.6),
                              (0.3, 0.4, 0.6, 0.7),
                              (0.2, 0.3, 0.7, 0.8),
                              (0.1, 0.2, 0.8, 0.9),
                              (0.0, 0.1, 0.9, 1.0)
                          ]]


def make_figure():
    fig, ax = plt.subplots(1, 5, figsize=(12, 2), layout="constrained")

    for ix, config in enumerate(mask_nonselect_configs):
        output_folder = base_output_folder / f'{config.experiment_name}'
        output_folder.mkdir(exist_ok=True, parents=True)
        config.to_yaml_file(output_folder / 'config.yml')

        log_git_versions(output_folder)

        case_folder = output_folder / f'image_vs_image_{imagenet_case.name}'
        case_folder.mkdir(exist_ok=True, parents=True)
        saliency, central_value, input_image = run_image_vs_image_experiment(imagenet_case, config, case_folder, analyse=False)

        plot_saliency_map_on_image(input_image, saliency[0], ax=ax.flatten()[ix],
                                   title="", add_value_limits_to_title=False,
                                   vmin=saliency[0].min(), vmax=saliency[0].max(),
                                   central_value=central_value, **fancy_figure_kwargs)
    
    ax[0].text(half_image_size, image_size + 20, 'discarded: 50-40% (both sides)',
               horizontalalignment='center', verticalalignment='center')
    ax[1].text(half_image_size, image_size + 20, '40-30%',
               horizontalalignment='center', verticalalignment='center')
    ax[2].text(half_image_size, image_size + 20, '30-20%',
               horizontalalignment='center', verticalalignment='center')
    ax[3].text(half_image_size, image_size + 20, '20-10%',
               horizontalalignment='center', verticalalignment='center')
    ax[4].text(half_image_size, image_size + 20, 'included: top 10%',
               horizontalalignment='center', verticalalignment='center')

    fig.savefig(base_output_folder / 'masking_non_selected.pdf')



if __name__ == '__main__':
    make_figure()
