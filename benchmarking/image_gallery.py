import glob
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

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

  <title>Explainable embeddings Analysis Platform -- version 1.0.1 -- life is a jurny</title>

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

    for group in groups:
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

    toc += '</ol>'

    with open('image_gallery.html', 'w') as f:
        f.write(header + toc + content + footer)


main()
