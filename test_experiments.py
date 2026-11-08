import math
import torch
from torch import nn

from src import MLP, measure_gradients


def test_model_output_shape():
    model = MLP(input_dim=20, width=16, depth=4, activation="relu", init="he")
    output = model(torch.randn(10, 20))
    assert output.shape == (10, 1)
 
 
def test_gradients_are_finite():
    rows = measure_gradients(
        depth=4, activation="relu", init="he", seed=11,
        batch_size=16, input_dim=20, width=16
    )
    assert len(rows) == 5  # four hidden Linear layers plus output layer
    assert all(math.isfinite(row["weight_grad_l2"]) for row in rows)
 
 
def test_same_seed_repeats():
    a = measure_gradients(4, "relu", "he", seed=99, batch_size=16, width=16)
    b = measure_gradients(4, "relu", "he", seed=99, batch_size=16, width=16)
    assert [x["weight_grad_l2"] for x in a] == [x["weight_grad_l2"] for x in b]
