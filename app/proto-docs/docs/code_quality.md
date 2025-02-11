# Formatting

`Black` is recommended to format the Python files. It is a highly opinionated formatter that follows PEP8. 

## Typical Workflow

1. Add [`black`](https://pypi.org/project/black/) python library as dev dependency in the virtual environment. Pin down the version if necessary.
    
    === "Poetry"

        ```bash
        poetry add -G dev black
        ```

    === "Conda"

        ```bash
        conda install black
        ```

2. Install `Black` vscode extension as specified in [Computer Setup](./0_computer_setup.md)

3. Enable format on save in your editor

## Additional settings
`Black` can be customized through `pyproject.toml`

`Black` settings are explained in [here](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html).

# Linting

`Ruff` is recommended to lint the Python files. 

## Typical Workflow

1. Add [`ruff`](https://pypi.org/project/ruff/) python library as dev dependency in the virtual environment. Pin down the version if necessary.

    === "Poetry"

        ```bash
        poetry add -G dev ruff
        ```

    === "Conda"

        ```bash
        conda install ruff
        ```

2. Install `Ruff` vscode extension as specified in [Computer Setup](./0_computer_setup.md)

## Additional setting

`Ruff` can be customized through `pyproject.toml`

`Ruff` settings are explained in [here](https://beta.ruff.rs/docs/configuration/).


# Type checking

`Pyright` is recommended to type check all python files in the project.

## Typical Worklow

1. Add [`pyright`](https://pypi.org/project/pyright/) python library as dev dependency in the virtual environment. Pin down the version if necessary.

    === "Poetry"

        ```bash
        poetry add -G dev pyright
        ```

    === "Conda"

        ```bash
        conda install pyright
        ```

2. Install `Python` vscode extension as specified in [Computer Setup](./0_computer_setup.md)


## Additional setting

`Pyright` can be customized through `pyproject.toml`

`Pyright` settings are explained in [here](https://github.com/microsoft/pyright/blob/main/docs/configuration.md)