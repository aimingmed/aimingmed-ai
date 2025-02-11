# Introduction

[`MkDocs`](https://www.mkdocs.org/) is a static site generator that's geared towards building project documentation. Documentation source files are written in `Markdown`, and configured with a single `YAML` configuration file.

# Installation

`MkDocs` is a Python library and there are also some extras recommended plugins to be installed.

1. [`mkdocs-material`](https://pypi.org/project/mkdocs-material/) - Theme
2. [`mkdocstrings`](https://pypi.org/project/mkdocstrings/) - Auto generate documentation from docstrings
3. [`mkdocs-jupyter`](https://pypi.org/project/mkdocs-jupyter/) - Embed Jupyter notebook in the documentation

=== "Poetry"

    ```bash
    poetry add -G doc mkdocs mkdocs-material "mkdocstrings[python]" mkdocs-jupyter
    ```

=== "Conda"

    ```bash
    conda install mkdocs mkdocs-material mkdocstrings mkdocs-jupyter
    ```

# Workflow

1. Install mkdocs and the plugins as above

2. Run `mkdocs new .` in the project root folder. It will create `mkdocs.yml` and `docs` folder

3. Edit `mkdocs.yml` to configure the site

4. Write the documentation in `docs` folder

# References:

- [MkDocs docs](https://www.mkdocs.org/)
