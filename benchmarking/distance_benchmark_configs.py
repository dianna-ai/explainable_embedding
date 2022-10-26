import dataclasses
from itertools import product

from benchmarking.Config import original_config_options

furthest_masks_config_options = dataclasses.replace(original_config_options,
                                                    mask_selection_range_min=.8,
                                                    mask_selection_range_max=1,
                                                    )

number_of_masks_configs = [dataclasses.replace(original_config_options,
                                               experiment_name='n_masks_sweep_',
                                               n_masks=n_mask,
                                               ) for n_mask, seed in
                           product([10, 50, 100, 200, 500, 1000, 2000, 5000], range(20))]

p_keep_configs = [dataclasses.replace() for p_keep in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]]
mask_selection_configs = [dataclasses.replace() for n_masks in some_range]
mask_weighting_configs = [dataclasses.replace() for n_masks in some_range]
