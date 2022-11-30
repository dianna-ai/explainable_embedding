import glob
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint

from benchmarking.Config import Config


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
    entries = glob.glob('./output_das6/*/*/*/*.png')
    ims = [to_image(entry) for entry in entries]
    pprint(ims)
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
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <title>A Basic HTML5 Template</title>
  <meta name="description" content="A simple HTML5 Template for new projects.">
  <meta name="author" content="SitePoint">

  <meta property="og:title" content="A Basic HTML5 Template">
  <meta property="og:type" content="website">
  <meta property="og:url" content="https://www.sitepoint.com/a-basic-html5-template/">
  <meta property="og:description" content="A simple HTML5 Template for new projects.">
  <meta property="og:image" content="image.png">

  <link rel="icon" href="/favicon.ico">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="apple-touch-icon" href="/apple-touch-icon.png">

  <link rel="stylesheet" href="css/styles.css?figure_template=1.0">
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-EVSTQN3/azprG1Anm3QDgpJLIm9Nao0Yz1ztcQTwFspd3yD65VohhpuuCOmLASjC" crossorigin="anonymous">
  <style>
    figure {
        float: left;
        margin-right: 0em;
    }
    
    div {
        overflow: hidden;
    }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
</head>

<body>"""
    footer = """
</body>
</html>
"""
    content = ''
    cases = set([im.case for im in ims])
    domains = set([im.domain for im in ims])

    for group in groups:
        content += f'<h1>{group}</h1>'
        for domain in domains:
            content += f'<h2>{domain}</h2>'
            for case in cases:
                content += '<div>' + ''.join([make_very_nice_img(im) for im in ims if
                                              im.group == group and im.case == case and im.domain == domain]) + '</div>'
    with open('image_gallery.html', 'w') as f:
        f.write(header + content + footer)


main()
