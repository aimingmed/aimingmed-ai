# Pyenv and Poetry

Pyenv is meant to manage multiple Python versions installed in the computer. It allows you to download different Python versions. You can set a global Python version and a different local Python version for a specific project, so when you enter that directory, the Python version is automatically switched to the one you set.

Poetry is meant to manage virtual environment of a Python project. Poetry does not have a way to download Python specific version needed in a project, so it requires Pyenv to provide one.

## Typical Workflow

You need to replace `<project name>` and `<Python version>` based on your needs. For example, you may set `<project name>` to `my-project` and `<Python version>` to `3.11.3`.

### New Project

1. Decide which Python version should be used for the new project.

2. Ensure you have the Python version needed in the computer. You may check it through `pyenv versions`

    If it is not installed yet, you may run `pyenv install <Python version>` in the terminal.

    !!! note

        You can also check installable Python list by running `pyenv install --list`
    

3. Create the new project folder by running `poetry new --src <project name>` and change directory to the project folder `cd <project name>`

4. Now set the Python version locally `pyenv local <Python version>`

5. Ensure Python version requirement in `pyproject.toml` satisfies Python version specified in the previous step

6. Set Poetry to create the virtual environment and prefer active Python in the path
    ```
    poetry config virtualenvs.in-project true --local 
    poetry config virtualenvs.prefer-active-python true --local
    ```

    !!! note

        `in-project` is needed to ensure the virtual environment folder `.venv` is created in the project root folder. This is useful when you want find this folder easily during docker multi-stage build.

        `prefer-active-python` is needed to ensure Poetry uses the Python version specified in the previous step.

        You can also set the configuration globally by replacing `--local` with `--global`. This is useful when you want to use the same configuration for all projects.

7. Create the Python virtual environment by running `poetry install`

8. Activate the virtual environment by running `poetry shell`

9. You can specify third-party dependencies with `poetry add` ([doc here](https://python-poetry.org/docs/cli/#add)) in the terminal or directly edit `pyproject.toml`

### Convert Existing Project

!!! tip

    It is usually unnecessary to convert existing Python virtual environment manager to Poetry if you already have one. However, if you want to convert it, you can follow the steps below.

It follows the same steps above, except in creating project folder since you already existing one. So instead of running `poetry new --src <project name>`, run `poetry init --no-interaction --python <Python version>`.

- References:

    - [Pyenv docs](https://github.com/pyenv/pyenv)
    - [Poetry docs](https://python-poetry.org/docs/)

# Conda or Mamba

## Typical Workflow

You need to replace `<project name>`, `<Python version>`, and `<environment name>` based on your needs. For example, you may set `<project name>` to `my-project`, `<Python version>` to `3.11.3`, and `<environment name>` to `my-env`.

If `mamba` is installed, replace `conda` command with `mamba` for faster third-party packages installation.

### New Project

1. Create new folder `mkdir <project name> && cd <project name>`

2. Specify the Python virtual environment requirements by creating  `environment.yml` file in project root folder

    ```yaml
    name: <environment name>
    channels:
        - conda-forge
    dependencies:
        - python=<Python version>
        - pandas=2.0
        - ...
    ```
3. Create Python virtual environment and install third party dependencies by running `conda env create -f environment.yml`

4. Activate the environment by running `conda activate <environment name>`

5. You can specify third-party dependencies with `conda install` ([doc here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html)) in the terminal or directly edit `environment.yml`

- References:

    - [Conda docs](https://docs.conda.io/projects/conda/en/latest/user-guide/index.html)

### Convert Existing Project

!!! tip

    It is usually unnecessary to convert existing Python virtual environment manager to Conda if you already have one. However, if you want to convert it, you can follow the steps below.

It follows the same steps above, except in creating project folder since you already existing one. You may freeze the existing dependencies through `pip freeze > requirements.txt` and create `environment.yml` file with the following content:

```yaml
name: <environment name>
channels:
    - conda-forge
dependencies:
    - python=<Python version>
    - pip
    - pip:
        - -r requirements.txt
```