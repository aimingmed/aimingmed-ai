# Visual Studio Code setup

1.  Recommended vscode extensions for Python Development

    - [`Python`](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
    - [`Black`](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
    - [`Ruff`](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)
    - [`Jupyter`](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
    - [`autoDocstring`](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
    - [`Even Better TOML`](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)
    - [`YAML`](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
    - [`Remote Development`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)
    - [`Docker`](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)
    - [`GitLens`](https://marketplace.visualstudio.com/items?itemName=eamodio.gitlens)

2.  Enable strict typehints in the setting

    ```json
    {
      "python.analysis.typeCheckingMode": "strict",
      "python.analysis.inlayHints.pytestParameters": true,
      "python.analysis.inlayHints.functionReturnTypes": true
    }
    ```

3.  Format using `Black` and Lint using `Ruff` on save

    !!! warning

        Set all these settings to false for existing projects that never use `Black` or `Ruff` before.

    ```json
    {
      "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
          "source.organizeImports": true,
          "source.fixAll": true
        }
      },
      "python.formatting.provider": "none",
      "notebook.formatOnSave.enabled": true
    }
    ```

4.  Test using `pytest`

    ```json
    {
      "python.testing.pytestEnabled": true,
      "python.testing.pytestArgs": ["tests"]
    }
    ```

5.  Use `google` documentation style

    ```json
    {
      "autoDocstring.docstringFormat": "google"
    }
    ```

6.  Guides from the internet

    - [Python in Visual Studio](https://code.visualstudio.com/docs/languages/python)

# Virtual Environment Management

Pyenv and Poetry is typically used for managing Python package, while Conda is more popular in Data Science project.

Both features virtual environment creation, but Poetry offers finer details on installing packages for production or development, publishing the Python package.

1.  **Both** [`Pyenv`](https://github.com/pyenv/pyenv) **AND** [`Poetry`](https://github.com/python-poetry/poetry/)

    !!! note "Pyenv Installation"
    Pyenv installation may requires executing more than one line in terminal, you also need to set your shell profile (`.zshrc` or `.bash_profile`).

        Read carefully [here](https://github.com/pyenv/pyenv#installation)

    !!! note "Poetry Installation"

        If you have old Poetry version (<1.2), you need to uninstall it and install the latest version.

2.  **Either** [Conda `Miniforge` **OR** `Mambaforge`](https://github.com/conda-forge/miniforge)

3.  [Docker](https://www.docker.com/)

# Gitignore

Download gitignore template from [gitignore.io](https://www.toptal.com/developers/gitignore/).

[`Python` and `visualstudiocode`](https://www.toptal.com/developers/gitignore/api/python,visualstudiocode) are recommended arguments for starter.

## Alternative Way

You may create command line function `gi` to download gitignore file. Follow the guide in [here](https://docs.gitignore.io/install/command-line). Then, you can the command in the project root folder.

```bash
gi python,visualstudiocode >> .gitignore
```
