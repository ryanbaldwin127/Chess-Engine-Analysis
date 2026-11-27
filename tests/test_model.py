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
from chesszero.model import ChessNet, count_parameters, masked_log_softmax, policy_value_loss

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
    target_values = torch.tensor([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
    mask = torch.ones(8, POLICY_SIZE, dtype=torch.bool)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def loss_fn():
        policy, value = model(x)
        return policy_value_loss(policy, value, target_moves, target_values, mask)[0]

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


# --- Loss ------------------------------------------------------------------

def random_mask(batch_size, legal_per_row=30, seed=0):
    """A mask with legal_per_row legal slots per row, like a typical position."""
    g = torch.Generator().manual_seed(seed)
    mask = torch.zeros(batch_size, POLICY_SIZE, dtype=torch.bool)
    for row in mask:
        row[torch.randperm(POLICY_SIZE, generator=g)[:legal_per_row]] = True
    return mask


def legal_target(mask, seed=0):
    """A one-hot target index and a visit-count-like distribution, both on legal moves."""
    g = torch.Generator().manual_seed(seed)
    index = torch.stack([row.nonzero()[torch.randint(len(row.nonzero()), (1,), generator=g)].squeeze() for row in mask])
    dist = torch.rand(mask.shape, generator=g) * mask
    return index, dist / dist.sum(dim=1, keepdim=True)


def loss_inputs(batch_size=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch_size, POLICY_SIZE, generator=g, requires_grad=True)
    value = torch.rand(batch_size, 1, generator=g) * 2 - 1
    z = torch.tensor([1.0, -1.0, 0.0] * (batch_size // 3))
    mask = random_mask(batch_size, seed=seed)
    return logits, value, z, mask


def test_one_hot_loss_matches_masked_cross_entropy():
    logits, value, z, mask = loss_inputs()
    index, _ = legal_target(mask)
    total, policy_loss, value_loss = policy_value_loss(logits, value, index, z, mask)
    expected = F.cross_entropy(logits.masked_fill(~mask, float("-inf")), index)
    assert torch.allclose(policy_loss, expected)
    assert torch.allclose(value_loss, F.mse_loss(value.squeeze(1), z))
    assert torch.allclose(total, policy_loss + value_loss)


def test_distribution_target_is_finite():
    """F.cross_entropy gives NaN here (0 target x -inf log-prob); ours must not."""
    logits, value, z, mask = loss_inputs()
    _, dist = legal_target(mask)
    assert torch.isnan(F.cross_entropy(logits.masked_fill(~mask, float("-inf")), dist))
    total, policy_loss, _ = policy_value_loss(logits, value, dist, z, mask)
    assert torch.isfinite(total)
    total.backward()
    assert torch.isfinite(logits.grad).all()


def test_distribution_loss_matches_manual_sum_over_legal_moves():
    logits, value, z, mask = loss_inputs()
    _, dist = legal_target(mask)
    _, policy_loss, _ = policy_value_loss(logits, value, dist, z, mask)
    expected = torch.stack([
        -(dist[i, m] * torch.log_softmax(logits[i, m], dim=0)).sum() for i, m in enumerate(mask)
    ]).mean()
    assert torch.allclose(policy_loss, expected)


def test_one_hot_distribution_equals_index_target():
    logits, value, z, mask = loss_inputs()
    index, _ = legal_target(mask)
    one_hot = F.one_hot(index, POLICY_SIZE).float()
    a = policy_value_loss(logits, value, index, z, mask)[1]
    b = policy_value_loss(logits, value, one_hot, z, mask)[1]
    assert torch.allclose(a, b)


def test_uniform_logits_give_log_of_legal_count():
    """With equal logits, every legal move has probability 1/30."""
    _, value, z, mask = loss_inputs()
    index, _ = legal_target(mask)
    logits = torch.zeros(mask.shape)
    _, policy_loss, _ = policy_value_loss(logits, value, index, z, mask)
    assert torch.allclose(policy_loss, torch.log(torch.tensor(30.0)))


@pytest.mark.parametrize("target_kind", ["index", "distribution"])
def test_illegal_logits_have_no_effect(target_kind):
    """Changing illegal slots' logits doesn't change the loss, and they get no gradient."""
    logits, value, z, mask = loss_inputs()
    index, dist = legal_target(mask)
    target = index if target_kind == "index" else dist
    total, _, _ = policy_value_loss(logits, value, target, z, mask)
    total.backward()
    assert torch.all(logits.grad[~mask] == 0)
    assert logits.grad[mask].abs().sum() > 0

    shifted = logits.detach().clone()
    shifted[~mask] += 100.0
    assert torch.allclose(policy_value_loss(shifted, value, target, z, mask)[0], total)


def test_masked_probabilities_are_exactly_zero():
    logits, _, _, mask = loss_inputs()
    probs = masked_log_softmax(logits, mask).exp()
    assert torch.all(probs[~mask] == 0)
    assert torch.allclose(probs.sum(dim=1), torch.ones(len(mask)))


def test_bf16_network_outputs():
    """Under bf16 autocast the loss is still computed, and returned, in float32."""
    logits, value, z, mask = loss_inputs()
    _, dist = legal_target(mask)
    total, policy_loss, value_loss = policy_value_loss(logits.bfloat16(), value.bfloat16(), dist, z, mask)
    assert total.dtype == torch.float32 and torch.isfinite(total)
    reference = policy_value_loss(logits, value, dist, z, mask)[1]
    assert torch.allclose(policy_loss, reference, rtol=0.05)


def test_value_weight():
    logits, value, z, mask = loss_inputs()
    index, _ = legal_target(mask)
    total, policy_loss, value_loss = policy_value_loss(logits, value, index, z, mask, value_weight=0.25)
    assert torch.allclose(total, policy_loss + 0.25 * value_loss)
