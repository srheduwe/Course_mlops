from importlib.machinery import SourceFileLoader
import pytest
import torch
import sys

sys.path.append('src/models/')
from model import MyAwesomeModel

@pytest.mark.parametrize("test_input, expected", [(torch.rand(1,1,28,28),1), (torch.rand(2,1,28,28),2) ,(torch.rand(5,1,28,28),5)])

# test_data = torch.rand(1,1,28,28)

def test_load_model(test_input, expected):
    model = MyAwesomeModel()
    output = model(test_input)
    assert output.shape == torch.Size([expected,10])
