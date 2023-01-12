from importlib.machinery import SourceFileLoader
import pytest
import torch
import os.path

@pytest.mark.skipif(not os.path.exists("data/processed/dataset_test.pt"), reason="Data files not found")
def test_datasest_shape():
    datamaker = SourceFileLoader("mnist","src/data/make_dataset.py").load_module()
    dataset_train = datamaker.mnist(True)
    dataset_test = datamaker.mnist(False)
    assert len(dataset_train.data) == 25000
    assert len(dataset_test.data) == 5000

def test_datapoint_shape():
    datamaker = SourceFileLoader("mnist","src/data/make_dataset.py").load_module()
    dataset_train = datamaker.mnist(True)
    dataset_test = datamaker.mnist(False)
    assert dataset_train.data.shape == torch.Size([25000, 1, 28, 28])
    assert dataset_test.data.shape == torch.Size([5000, 1, 28, 28])

def test_labels():
    datamaker = SourceFileLoader("mnist","src/data/make_dataset.py").load_module()
    dataset_train = datamaker.mnist(True)
    dataset_test = datamaker.mnist(False)
    assert len(dataset_train.targets.unique()) == 10
    assert len(dataset_test.targets.unique()) == 10