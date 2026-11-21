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


assert POLICY_SIZE == NUM_MOVE_PLANES * 64
