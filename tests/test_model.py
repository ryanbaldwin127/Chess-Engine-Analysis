"""Tests for chesszero/model.py and chesszero/config.py.

Run from the repo root: python -m pytest tests/test_model.py -v

Tests run against both towers. The "residual" ones are skipped until Munisa's
ResidualBlock and ResidualTower are written; when they all pass, the tower is done.
"""

import pytest
import torch
import torch.nn.functional as F

from chesszero.config import load_config, missing_settings
from chesszero.encoding import INPUT_PLANES, POLICY_SIZE
from chesszero.model import ChessNet, count_parameters

BLOCKS, FILTERS = 8, 128  # tentative size from CLAUDE.md


def build(tower, blocks=BLOCKS, filters=FILTERS):
    try:
        return ChessNet(blocks, filters, tower=tower)
    except NotImplementedError as e:
        pytest.skip(f"{tower} tower not written yet ({e})")


@pytest.fixture(params=["stand_in", "residual"])
def tower(request):
    return request.param


def random_batch(batch_size=4):
    return torch.randn(batch_size, INPUT_PLANES, 8, 8)


def test_output_shapes(tower):
    policy, value = build(tower)(random_batch(4))
    assert policy.shape == (4, POLICY_SIZE)
    assert value.shape == (4, 1)


def test_value_in_range(tower):
    model = build(tower)
    _, value = model(random_batch(16) * 10)  # large inputs push tanh toward its limits
    assert value.min() >= -1 and value.max() <= 1


def test_gradients_reach_input_layer(tower):
    """A broken shortcut or a detached tensor stops gradients before the first layer."""
    model = build(tower)
    policy, value = model(random_batch(4))
    (policy.sum() + value.sum()).backward()
    grad = model.input_block[0].weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_overfits_tiny_batch(tower):
    """The network should memorise 8 fixed examples. If it can't, something is wired wrong."""
    torch.manual_seed(0)
    model = build(tower, blocks=2, filters=32)
    x = random_batch(8)
    target_moves = torch.randint(0, POLICY_SIZE, (8,))
    target_values = torch.tensor([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0]).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def loss_fn():
        policy, value = model(x)
        return F.cross_entropy(policy, target_moves) + F.mse_loss(value, target_values)

    first = loss_fn().item()
    for _ in range(150):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_parameter_count():
    """About 2.4M in the tower plus about 0.2M in the heads at 8 blocks x 128 filters."""
    model = build("residual")
    assert 2.4e6 < count_parameters(model) < 2.7e6


def test_builds_from_config():
    cfg = load_config("configs/condition_25.yaml")
    assert not missing_settings(cfg["network"])
    model = ChessNet.from_config(cfg, tower="stand_in")
    policy, _ = model(random_batch(2))
    assert policy.shape == (2, POLICY_SIZE)


def test_condition_overrides_base():
    cfg = load_config("configs/condition_75.yaml")
    assert cfg["condition"] == 75
    assert cfg["network"]["filters"] == load_config()["network"]["filters"]
