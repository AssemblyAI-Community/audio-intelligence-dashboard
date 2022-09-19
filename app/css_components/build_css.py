# Strings together css files in this folder and exports to `../styles.css`

import os

css_filepaths = [f for f in os.listdir() if f.endswith(".css")]

css_filepaths.remove('file.css')
css_filepaths.insert(0, 'file.css')

css = ""
for filepath in css_filepaths:
    with open(filepath, 'r') as file:
        css += file.read()

with open("../styles.css", 'w') as f:
    f.write(css)