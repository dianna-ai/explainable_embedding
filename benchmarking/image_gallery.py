import glob
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import List

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


def main():
    entries = glob.glob('/Users/pbos/SURFdrive/explainable_embeddings_SHARED/output/*/*/*/*.png')
    ims = [to_image(entry) for entry in sorted(entries)]
    # pprint(ims)
    create_html(ims)


def make_very_nice_img(image: Image):
    """Makes an HTML figure. Yes, yes!"""
    caption = image.config.to_yaml().replace('\n', '<br/>')
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
        margin-right: 0em;
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
                deez_images = [make_very_nice_img(im) for im in sorted(ims, key=lambda x: x.config.experiment_name)
                               if im.group == group and im.case == case and im.domain == domain]
                if (len(deez_images)) > 0:
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

    content = f'<h1 id="{group}">{group}</h1>'
    for domain in domains:
        toc += f'<li><a href="#{group}-{domain}">{domain}</a></li>'
        content += f'<h2 id="{group}-{domain}">{domain}</h2>'
        for case in cases:
            deez_images = {(im.config.random_seed, im.config.number_of_masks): make_very_nice_img(im)
                           for im in sorted(ims, key=lambda x: x.config.experiment_name)
                           if im.group == group and im.case == case and im.domain == domain}
            if (len(deez_images)) > 0:
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


main()
