def run_image_vs_image_experiment(case, config):
    match case:
        case 'bee_vs_fly':
            input_image_path = 'data/images/bee.jpg'
            reference_image_path = 'data/images/fly.jpg'
            model = ImageNetModel()







def run_image_captioning_experiment(case, config):
    pass


def run_benchmark(config):
    imagenet_cases = ['bee_vs_fly', 'labradoodles', 'dogcar_vs_car', 'dogcar_vs_dog', 'flower_vs_car', 'car_vs_bike']
    for imagenet_case in imagenet_cases:
        run_image_vs_image_experiment(imagenet_case, config)

    image_captioning_cases = ['foo', 'bar']
    for case in image_captioning_cases:
        run_image_captioning_experiment(case, config)

    #something with molecules?

config_options = {
    'mask_selection_method': '',  # range, random, all
    'mask_selection_percentage_min': 0,  # 0-100
    'mask_selection_percentage_max': 100, # 0-100
    'mask_distance_power': 1,  # 0.5, 1, 2 ... 100
    # Normalize weights to [0, 1] before taking dot product with masks.
    'weight_range_normalization': False,  # True, False
    'distance_metric': 'cosine',  # cosine, euclidian, manhatten?
    'number_of_masks': 1000,  # auto, [1, -> ]
    'p-keep': 0.5,  # None (auto), 0 - 1
    'feature_res': 8,  # [1, ->]
}
