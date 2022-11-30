import dataclasses
from itertools import product

from Config import original_config_options

furthest_masks_config_options = dataclasses.replace(original_config_options,
                                                    mask_selection_range_min=.8,
                                                    mask_selection_range_max=1,
                                                    )

number_of_masks_configs = [dataclasses.replace(original_config_options,
                                               experiment_name=f'n_masks_sweep_{n_mask}_seed_{seed}',
                                               random_seed=seed,
                                               number_of_masks=n_mask,
                                               ) for n_mask, seed in
                           product([10, 50, 100, 200, 500, 1000, 2000, 5000], range(20))]

p_keep_configs = [dataclasses.replace(original_config_options,
                                      experiment_name=f'p_keep_sweep_{p_keep}',
                                      p_keep=p_keep
                                      ) for p_keep in
                  [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]]

mask_threshold_configs = [dataclasses.replace(original_config_options,
                                              experiment_name=f'mask_selection_threshold_sweep_neg{neg_min}-{neg_max}_pos{pos_min}-{pos_max}',
                                              mask_selection_range_max=pos_max,
                                              mask_selection_range_min=pos_min,
                                              mask_selection_negative_range_max=neg_max,
                                              mask_selection_negative_range_min=neg_min,
                                              ) for neg_min, neg_max, pos_min, pos_max in
                          [
                              (0.0, 0.05, 0.95, 1.0),
                              (0.0, 0.1, 0.9, 1.0),
                              (0.0, 0.15, 0.85, 1.0),
                              (0.0, 0.2, 0.8, 1.0),
                              (0.0, 0.25, 0.75, 1.0),
                              (0.0, 0.3, 0.7, 1.0),
                              (0.0, 0.4, 0.6, 1.0),
                              (0.0, 0.5, 0.5, 1.0),
                          ]]

mask_one_sided_configs = [dataclasses.replace(original_config_options,
                                              experiment_name=f'mask_selection_one_sided_neg{neg_min}-{neg_max}_pos{pos_min}-{pos_max}',
                                              mask_selection_range_max=pos_max,
                                              mask_selection_range_min=pos_min,
                                              mask_selection_negative_range_max=neg_max,
                                              mask_selection_negative_range_min=neg_min,
                                              manual_central_value=None,
                                              ) for neg_min, neg_max, pos_min, pos_max in
                          [
                              (0.0, 0.0, 0.0, 1.0),
                              (0.0, 0.0, 0.9, 1.0),
                              (0.0, 0.0, 0.8, 1.0),
                              (0.0, 0.0, 0.7, 1.0),
                              (0.0, 0.0, 0.6, 1.0),
                              (0.0, 0.0, 0.5, 1.0),
                              (0.0, 0.1, 0.0, 0.0),
                              (0.0, 0.2, 0.0, 0.0),
                              (0.0, 0.3, 0.0, 0.0),
                              (0.0, 0.4, 0.0, 0.0),
                              (0.0, 0.5, 0.0, 0.0),
                          ]]

mask_nonselect_configs = [dataclasses.replace(original_config_options,
                                              experiment_name=f'mask_non_selection_neg{neg_min}-{neg_max}_pos{pos_min}-{pos_max}',
                                              mask_selection_range_max=pos_max,
                                              mask_selection_range_min=pos_min,
                                              mask_selection_negative_range_max=neg_max,
                                              mask_selection_negative_range_min=neg_min,
                                              ) for neg_min, neg_max, pos_min, pos_max in
                          [
                              (0.2, 0.5, 0.5, 0.8),
                              (0.1, 0.5, 0.5, 0.9),
                          ]]

feature_res_configs = [dataclasses.replace(original_config_options,
                                           experiment_name=f'feature_res_sweep_{feature_res}',
                                           feature_res=feature_res
                                           ) for feature_res in
                       [1, 2, 4, 6, 8, 12, 16, 24, 32, 64, 128]]

runs_20221109 = number_of_masks_configs + p_keep_configs + mask_threshold_configs + mask_one_sided_configs + mask_nonselect_configs + feature_res_configs

test_config = dataclasses.replace(original_config_options, number_of_masks=10, experiment_name='test')

runs_20221130 = p_keep_configs + feature_res_configs
