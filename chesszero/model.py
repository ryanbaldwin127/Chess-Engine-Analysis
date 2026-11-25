"""ResNet with policy and value heads.

Shapes (B = batch size):
    input   (B, INPUT_PLANES, 8, 8)   from chesszero.encoding.board_to_planes
    policy  (B, POLICY_SIZE)          raw logits, unmasked; the loss and the MCTS
                                      apply the legal-move mask
    value   (B, 1)                    in [-1, 1], from the side to move's view

Layout: input block -> tower of residual blocks -> policy head and value head.
The tower follows AlphaGo Zero's design [AlphaGoZero], built from residual
blocks [ResNet]. The policy head is convolutional (see docs/decisions.md #22):
it outputs the 73 AlphaZero move planes on the 8x8 board directly [AlphaZero].

Loss: policy_value_loss (masked policy cross-entropy + value MSE), shared by the
supervised and self-play learners. L2 regularization comes from the optimizer's
weight_decay.

Munisa owns ResidualBlock and ResidualTower. Until they're written, pass
tower="stand_in" to get a plain convolution stack with the same input and
output shapes, so everything else can be built and tested.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from chesszero.encoding import INPUT_PLANES, NUM_MOVE_PLANES, POLICY_SIZE


def conv_bn(in_channels, out_channels, kernel_size):
    """Convolution (no bias, 'same' padding) followed by batch norm."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
    )


# ---------------------------------------------------------------------------
# Tower: Munisa
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """One residual block [ResNet], as in AlphaGo Zero [AlphaGoZero]:

        x -> conv3x3 -> batch norm -> ReLU -> conv3x3 -> batch norm -> (+ x) -> ReLU

    Input and output shape: (B, filters, 8, 8). The "+ x" shortcut is what lets
    many blocks be stacked and still train; tests/test_model.py checks that
    gradients reach the input layer.

    Optional later (docs/decisions.md #14): a squeeze-and-excitation gate [SE]
    before the shortcut addition.
    """

    def __init__(self, filters: int):
        super().__init__()
        raise NotImplementedError("Munisa: ResidualBlock")

    def forward(self, x):
        raise NotImplementedError


class ResidualTower(nn.Module):
    """A stack of `blocks` ResidualBlocks. Input and output shape: (B, filters, 8, 8)."""

    def __init__(self, blocks: int, filters: int):
        super().__init__()
        raise NotImplementedError("Munisa: ResidualTower")

    def forward(self, x):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Stand-in tower and heads
# ---------------------------------------------------------------------------

class StandInTower(nn.Module):
    """Temporary tower: two plain conv layers, no shortcuts. For testing the rest of the code only."""

    def __init__(self, blocks: int, filters: int):
        super().__init__()
        self.layers = nn.Sequential(conv_bn(filters, filters, 3), nn.ReLU(),
                                    conv_bn(filters, filters, 3), nn.ReLU())

    def forward(self, x):
        return self.layers(x)


class PolicyHead(nn.Module):
    """conv3x3 -> BN -> ReLU -> conv1x1 to 73 move planes -> flatten to POLICY_SIZE logits.

    Flattening (B, 73, 8, 8) gives index = move_plane * 64 + square, which is the
    layout chesszero.encoding.move_to_index must follow.
    """

    def __init__(self, filters: int):
        super().__init__()
        self.conv = conv_bn(filters, filters, 3)
        self.out = nn.Conv2d(filters, NUM_MOVE_PLANES, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return self.out(x).flatten(1)


class ValueHead(nn.Module):
    """conv1x1 to 1 plane -> BN -> ReLU -> FC 256 -> ReLU -> FC 1 -> tanh [AlphaGoZero]."""

    def __init__(self, filters: int, hidden: int = 256):
        super().__init__()
        self.conv = conv_bn(filters, 1, 1)
        self.fc1 = nn.Linear(64, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.conv(x)).flatten(1)
        return torch.tanh(self.fc2(F.relu(self.fc1(x))))


class ChessNet(nn.Module):
    """Policy-value network. See the module docstring for shapes."""

    def __init__(self, residual_blocks: int, filters: int, tower: str = "residual"):
        super().__init__()
        self.input_block = conv_bn(INPUT_PLANES, filters, 3)
        if tower == "residual":
            self.tower = ResidualTower(residual_blocks, filters)
        elif tower == "stand_in":
            self.tower = StandInTower(residual_blocks, filters)
        else:
            raise ValueError(f"unknown tower {tower!r}")
        self.policy_head = PolicyHead(filters)
        self.value_head = ValueHead(filters)

    @classmethod
    def from_config(cls, cfg: dict, tower: str = "residual") -> "ChessNet":
        net = cfg["network"]
        return cls(net["residual_blocks"], net["filters"], tower=tower)

    def forward(self, x):
        x = F.relu(self.input_block(x))
        x = self.tower(x)
        return self.policy_head(x), self.value_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_log_softmax(policy_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-probabilities over the legal moves only; illegal slots get -inf.

    Computed in float32 even if the network runs in bf16 (docs/decisions.md #12),
    since the softmax over 4,672 slots is where low precision hurts most.
    """
    return F.log_softmax(policy_logits.float().masked_fill(~mask, float("-inf")), dim=1)


def policy_value_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    mask: torch.Tensor,
    value_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AlphaZero's loss without the L2 term: policy cross-entropy + value MSE.

    policy_logits  (B, POLICY_SIZE)  raw network output
    value          (B, 1) or (B,)    network output, side to move's view
    policy_target  (B,) int64        the move played (supervised, one-hot), or
                   (B, POLICY_SIZE)  a distribution over moves, e.g. normalized
                                     MCTS visit counts (self-play)
    value_target   (B,)              game result z, side to move's view
    mask           (B, POLICY_SIZE)  bool, True at legal moves (legal_move_mask)

    Returns (total, policy_loss, value_loss), each a scalar averaged over the
    batch; log the parts with .item().

    Illegal moves get probability exactly 0. For distribution targets, the masked
    terms are zeroed before summing: target 0 times log-probability -inf would
    otherwise be NaN, and F.cross_entropy does return NaN there. Target mass on
    an illegal move is ignored, so targets must come from legal moves only; a
    one-hot target on an illegal move gives an infinite loss.

    L2 regularization is not added here: pass weight_decay to the optimizer.
    """
    log_probs = masked_log_softmax(policy_logits, mask)
    if policy_target.dtype in (torch.int64, torch.int32):
        policy_loss = -log_probs.gather(1, policy_target.long().unsqueeze(1)).mean()
    else:
        policy_loss = -(policy_target.float() * log_probs.masked_fill(~mask, 0.0)).sum(dim=1).mean()
    value_loss = F.mse_loss(value.float().reshape(-1), value_target.float().reshape(-1))
    return policy_loss + value_weight * value_loss, policy_loss, value_loss


assert POLICY_SIZE == NUM_MOVE_PLANES * 64
