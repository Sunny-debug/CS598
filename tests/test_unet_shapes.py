import pytest

pytestmark = [pytest.mark.local, pytest.mark.slow, pytest.mark.requires_weights]

import torch
from app.models.unet_infer import UNetInfer
from models.unet import UNetSmall

def test_forward_shapes():
    m = UNetSmall(in_ch=3, base=16).eval()
    x = torch.randn(2,3,256,256)
    y = m(x)
    assert y.shape == (2,1,256,256)