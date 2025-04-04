# Build an ML Pipeline for OSCAR object identification

## Table of contents

- [Preliminary steps](#preliminary-steps)
  - [Create environment](#create-environment)
  - [Get API key for Weights and Biases](#get-api-key-for-weights-and-biases)
  - [Cookie cutter](#cookie-cutter)
  - [The configuration](#the-configuration)
  - [Running the entire pipeline or just a selection of steps](#running-the-entire-pipeline-or-just-a-selection-of-steps)
  - [Test the model perfomance on the test samples](#test-the-model-perfomance-on-the-test-samples)
  - [Test the production test samples based on production models](#test-production-samples-based-on-production-models)
  - [Wandb public workspace URL for this project](#wandb-public-workspace-url-for-this-project)

## Preliminary steps

### Create environment

Make sure to have pipenv installed and ready and you are at the app/mlops directory, then create a new environment:

```bash
# you may remove Pipfile.lock
> pipenv install
> pipenv shell
```

### Get API key for Weights and Biases

Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to
[https://wandb.ai/authorize](https://wandb.ai/authorize) and click on the + icon (copy to clipboard),
then paste your key into this command:

```bash
> wandb login [your API key]
```

You should see a message similar to:

```
wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc
```

### Cookie cutter

In order to make your job a little easier, you are provided a cookie cutter template that you can use to create
stubs for new pipeline components. It is not required that you use this, but it might save you from a bit of
boilerplate code. Just run the cookiecutter and enter the required information, and a new component
will be created including the `python_env.yml` file, the `MLproject` file as well as the script. You can then modify these
as needed, instead of starting from scratch.
For example:

```bash
> cookiecutter cookie-mlflow-step -o src

step_name [step_name]: basic_cleaning
script_name [run.py]: run.py
job_type [my_step]: basic_cleaning
short_description [My step]: This steps cleans the data
long_description [An example of a step using MLflow and Weights & Biases]: Performs basic cleaning on the data and save the results in Weights & Biases
parameters [parameter1,parameter2]: parameter1,parameter2,parameter3
```

This will create a step called `basic_cleaning` under the directory `src` with the following structure:

```bash
> ls src/basic_cleaning/
python_env.yml  MLproject  run.py
```

You can now modify the script (`run.py`), the virtualenv environment (`python_env.yml`) and the project definition
(`MLproject`) as you please.

The script `run.py` will receive the input parameters `parameter1`, `parameter2`,
`parameter3` and it will be called like:

```bash
> mlflow run src/step_name -P parameter1=1 -P parameter2=2 -P parameter3="test"
```

### The configuration

As usual, the parameters controlling the pipeline are defined in the `config.yaml` file defined in
the root of the starter kit. We will use Hydra to manage this configuration file.
Open this file and get familiar with its content. Remember: this file is only read by the `main.py` script
(i.e., the pipeline) and its content is
available with the `go` function in `main.py` as the `config` dictionary. For example,
the name of the project is contained in the `project_name` key under the `main` section in
the configuration file. It can be accessed from the `go` function as
`config["main"]["project_name"]`.

NOTE: do NOT hardcode any parameter when writing the pipeline. All the parameters should be
accessed from the configuration file.

NOTE: Make sure you have the .env file located at the llmops/src/chain_of_thought (it contains the API keys for the LLM chat models)

### Running the entire pipeline or just a selection of steps

In order to run the pipeline when you are developing, you need to be in the root of the starter kit,
then you can execute as usual:

```bash
# not recommended for now -- still in development stage
> cd app/llmops
> pipenv shell
> mlflow run .
```

This will run the entire pipeline. Please use the following to run working full pipeline for the project.
You may configure all settings for both training, testing, and production testing at the app/mlops/config.yaml.
Check all the `_steps` list you can run at app/llmops/main.py

```bash
> cd app/llmops
> pipenv shell
> mlflow run . -P steps=get_documents,etl_chromdb_pdf,chain_of_thought
> mlflow run . -P steps=chain_of_thought


```

When developing or troubleshooting, it is useful to be able to run one step at a time. Say you want to run only
the `chain_of_thought` step. The `main.py` is written so that the steps are defined at the top of the file, in the
`_steps` list, and can be selected by using the `steps` parameter on the command line:

```bash
> mlflow run . -P steps=chain_of_thought
```

If you want to run the `etl_chromdb_pdf` and the `chain_of_thought` steps, you can similarly do:

```bash
> mlflow run . -P steps=etl_chromdb_pdf,chain_of_thought
```

You can override any other parameter in the configuration file using the Hydra syntax, by
providing it as a `hydra_options` parameter. For example, say that we want to set the parameter
modeling -> product_to_train to Sale_MF and modeling-> stratify_by to Sale_MF:

```bash
> mlflow run . \
  -P steps=chain_of_thought \
  -P hydra_options="prompt_engineering.chat_model_provider='kimi' prompt_engineering.query='怎么治疗有kras的肺癌?'"
```

### Test the model perfomance on the test samples

First define the necessary parameters at the config.yaml at best_model_propensity and best_model_revenue sections, remember to set "prod" alias for the best_model_propensity and best_model_revenue to the best models you have trained before on wandb model output artifacts before you can run the testing on test samples to see the model performance on the test holdout samples. The test performance plots and test_result.csv are available at the wandb run log and output artifacts.

```bash
> mlflow run . -P steps=test_model
```

### Test the production test samples based on production models

First define the necessary parameters at the config.yaml at production.test_csv section, remember to set "prod" alias for the best_model_propensity and best_model_revenue to the best model you have trained before on wandb model output artifacts before you can run the testing on production.

```bash
> cd app/mlops
> pipenv shell
> mlflow run . \
  -P steps=chain_of_thought \
  -P hydra_options="prompt_engineering.query='怎么治疗有kras的肺癌?'"
# OR you can run the following to test the production samples
> mlflow run https://github.com/aimingmed/aimingmed-ai \
             -v v1.0.0 \
             -P steps=chain_of_thought \
             -P hydra_options="prompt_engineering.query='怎么治疗有kras的肺癌?'"
```

## Wandb public workspace URL for this project

Click the link below to see the wandb public workspace for this project. You can see the model training and testing results, as well as the production testing results.
https://wandb.ai/aimingmed/aimingmed-ai
