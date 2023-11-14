import argparse
import glob
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import List, Iterable

from Config import Config


@dataclass
class Image:
    path: Path
    experiment: str
    group: str
    domain: str
    case: str
    config: Config


groups = [
    'p_keep_sweep',
    'n_masks_sweep',
    'mask_selection_threshold',
    'mask_selection_one_sided',
    'mask_non_selection',
    'feature_res_sweep',
]


def get_group_name(experiment):
    for group in groups:
        if group in experiment:
            return group
    raise ValueError(f'no known group for {experiment}')


def to_image(entry):
    path = Path(entry)
    experiment = str(path.parent.parent.parent.name)
    return Image(path=path,
                 experiment=experiment,
                 group=get_group_name(experiment),
                 domain=(str(path.parent.parent.name)),
                 case=(str(path.parent.name)),
                 config=(Config.load(path.parent.parent.parent / 'config.yml')),
                 )


def main(base_path=Path('/Users/pbos/SURFdrive/explainable_embeddings_SHARED/output')):
    entries = base_path.glob('*/*/*/*.png')
    exclude = base_path.glob('EXCLUDE/*/*/*.png')
    entries = list(set(entries) - set(exclude))
    ims = [to_image(entry) for entry in sorted(entries)]
    # pprint(ims)
    create_html(ims)


def make_very_nice_img(image: Image, caption_parameters: Iterable[str]):
    """Makes an HTML figure. Yes, yes!"""
    caption_lines = []
    # only keep the lines in caption_parameters:
    for line in image.config.to_yaml().replace('-', '_').splitlines():
        if line.split(':')[0] in caption_parameters:
            caption_lines.append(line)
    caption = '<br/>'.join(caption_lines)
    return f"""
                         <figure>
                  <img src="{image.path}" style="width:100%">
                  <figcaption>{caption}</figcaption>
                </figure> 
                        """


def create_html(ims):
    header = """
<!doctype html>

<html lang="en">
<head>
  <meta charset="utf-8">

  <title>Explainable embeddings Analysis Platform U -- version 1.1.2</title>

  <style>
    figure {
        float: left;
        margin: 0em;
        margin-bottom: 55px;
    }

    img {
        width: 300px;
        height: 480px;
        object-fit: cover;
        margin-left: -55px;
        margin-right: -85px;
        margin-top: -55px;
        margin-bottom: -55px;
    }

    div {
        overflow: hidden;
    }
</style>
</head>

<body>
"""
    footer = """
</body>
</html>
"""
    content = ''
    cases = sorted(set([im.case for im in ims]))
    domains = sorted(set([im.domain for im in ims]))
    sorted_ims = sorted(ims, key=lambda x: x.config.experiment_name)
    toc = '<ol>'

    special_groups = ['n_masks_sweep']
    normal_groups = [group for group in groups if group not in special_groups]

    for group in normal_groups:
        toc += f'<li><a href="#{group}">{group}</a><ol>'

        content += f'<h1 id="{group}">{group}</h1>'
        for domain in domains:
            toc += f'<li><a href="#{group}-{domain}">{domain}</a></li>'
            content += f'<h2 id="{group}-{domain}">{domain}</h2>'
            for case in cases:
                case_images = [im for im in sorted_ims
                               if im.group == group and im.case == case and im.domain == domain]
                if (len(case_images)) > 0:
                    # we only output the parameters that differ per image in this case:
                    caption_parameters = case_images[0].config ^ case_images[1].config
                    deez_images = [make_very_nice_img(im, caption_parameters) for im in case_images]
                    content += f'<h3>{case}</h3>'
                    content += '<div>' + ''.join(deez_images) + '</div><hr>'

        toc += '</ol></li>'

    for group in special_groups:
        if group == 'n_masks_sweep':
            t, c = create_html_n_masks_sweep(domains, cases, ims)
            toc += t
            content += c
        else:
            raise Exception(f"group {group} not implemented in a super special way")

    # and finally end the toc
    toc += '</ol>'

    with open('image_gallery.html', 'w') as f:
        f.write(header + toc + content + footer)


def create_html_n_masks_sweep(domains, cases, ims: List[Image]):
    # especially for n_masks_sweep we do another layout
    group = 'n_masks_sweep'
    toc = f'<li><a href="#{group}">{group}</a><ol>'

    sorted_ims = sorted(ims, key=lambda x: x.config.experiment_name)

    content = f'<h1 id="{group}">{group}</h1>'
    for domain in domains:
        toc += f'<li><a href="#{group}-{domain}">{domain}</a></li>'
        content += f'<h2 id="{group}-{domain}">{domain}</h2>'
        for case in cases:
            case_images = {(im.config.random_seed, im.config.number_of_masks): im
                           for im in sorted_ims
                           if im.group == group and im.case == case and im.domain == domain}
            if (len(case_images)) > 0:
                keys = list(case_images.keys())
                # we only output the parameters that differ per image in this case:
                caption_parameters = case_images[keys[0]].config ^ case_images[keys[1]].config
                deez_images = {key: make_very_nice_img(im, caption_parameters) for key, im in case_images.items()}
                seed_list = sorted(set([im.config.random_seed for im in ims]))
                n_masks_list = sorted(set([im.config.number_of_masks for im in ims]))

                content += f'<h3>{case}</h3><table>'

                for n_masks in n_masks_list:
                    content += "<tr>"
                    for seed in seed_list:
                        content += f"<td>{deez_images[(seed, n_masks)]}</td>"
                    content += "</tr>"

                content += "</table>"

    toc += '</ol></li>'
    return toc, content


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to create an image gallery for the output.")
    parser.add_argument("base_path", type=Path, help="The base path to be processed; the root of the output folder.")

    args = parser.parse_args()
    return args.base_path


if __name__ == '__main__':
    base_path = parse_arguments()
    main(base_path=base_path)
