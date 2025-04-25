import pytest
from importlib import import_module

def test_config_import():
    mod = import_module('config')
    assert mod is not None

def test_main_import():
    mod = import_module('main')
    assert mod is not None
