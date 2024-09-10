import pandas as pd


def read_mprt_data(path, layers):
    json = pd.read_json(path)
    return [json[0][0][layer][0] for layer in layers]


with open('layer_numbers.txt', 'r') as f:
    layers = [line.split(' ')[1].strip('\n') for line in f.readlines()]
results = pd.DataFrame({
    'Top down': read_mprt_data('attribution_maps_1000_bee_bee2_top_down.json', layers),
    'Bottom up': read_mprt_data('attribution_maps_1000_bee_bee2_bottom_up.json', layers),
    'Independent': read_mprt_data('attribution_maps_1000_bee_bee2_independent.json', layers),
}, index=[i + 1 for i, _ in enumerate(layers)])

results.round(2).to_csv('results.csv')
results.round(2).to_latex('results.tex')
