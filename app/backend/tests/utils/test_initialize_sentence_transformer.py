import pytest
from importlib import import_module

def test_initialize_sentence_transformer_import():
    mod = import_module('utils.initialize_sentence_transformer')
    assert mod is not None
