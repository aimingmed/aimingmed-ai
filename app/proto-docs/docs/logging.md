# Introduction

Logging is a very important part of any application. It allows you to see what is happening in your application, and is especially useful when debugging.

Logging is already built in `Python` though `logging` library, and we may add `rich` library to make it more beautiful.

=== "Poetry"

    ```bash
    poetry add rich
    ```

=== "Conda"

    ```bash
    conda install rich
    ```

There are several ways to write logging configuration. You can specify it with `yaml` file, `ini` file, `json` file, Python `dict` or simply as Python code.

If [rich handler](https://rich.readthedocs.io/en/stable/logging.html) is used, you can only write the configuration as Python code.

# References:

- [Logging for ML Systems](https://madewithml.com/courses/mlops/logging/)
- [Python Logging Guide – Best Practices and Hands-on Examples](https://coralogix.com/blog/python-logging-best-practices-tips/)
