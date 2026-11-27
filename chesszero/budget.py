"""Network-evaluation counter: the experiment's budget and phase switch.

The budget B is counted in network evaluations (docs/decisions.md #1):

    n_eval = mcts_evals + TRAINING_COST * (supervised_samples + selfplay_samples)

    mcts_evals   positions actually sent through the network during self-play
                 search, as reported in each game file (cache hits, terminal
                 positions and batch padding are free)
    samples      training samples the learner actually trained on, in either phase

A run trains on human games until n_eval >= p * B, then plays itself until
n_eval >= B. Only the learner holds a Budget, and it is saved in the learner's
checkpoint. Self-play evaluations are counted when the learner ingests a game
file, not when a worker generates it: games that never reach the learner
(a worker killed mid-batch, files arriving after the run ends) never affected
the model, so they are recorded as discarded instead (#25). Validation passes
and evaluation matches don't change the model and are outside B.

Checkpoints fall at multiples of checkpoint_every, the same n_eval values in
every condition, so Elo-vs-compute curves line up. B must be a multiple of
4 * checkpoint_every, so every condition's switch point (0, 0.25, 0.5, 0.75 or
1 x B) is a checkpoint, which is also what branching from one supervised run
needs (#24).

Typical use in the learner:

    budget = Budget.from_config(cfg)            # or Budget.from_state_dict(ckpt["budget"])
    while budget.phase() == SUPERVISED:
        ... train on a batch of n samples ...
        budget.add_training_samples(n)
        if budget.checkpoint_due():
            save(weights, optimizer, budget.state_dict(), record=budget.record())
"""

import numbers
import time
from datetime import datetime, timezone

TRAINING_COST = 3  # forward + backward pass ~ 3 forward passes

SUPERVISED = "supervised"
SELFPLAY = "selfplay"
DONE = "done"

_COUNTERS = (
    "mcts_evals",
    "supervised_samples",
    "selfplay_samples",
    "games",
    "plies",
    "discarded_evals",
    "discarded_games",
)


class Budget:
    """Counts network evaluations for one run and decides its phase."""

    def __init__(
        self,
        total: int,
        supervised_fraction: float,
        condition: int | None = None,
        checkpoint_every: int | None = None,
        train_samples_per_position: float | None = None,
    ):
        if total <= 0:
            raise ValueError("total budget must be positive")
        if not 0 <= supervised_fraction <= 1:
            raise ValueError("supervised_fraction must be in [0, 1]")
        if checkpoint_every is not None and (checkpoint_every <= 0 or total % (4 * checkpoint_every)):
            raise ValueError(
                f"total ({total}) must be a positive multiple of 4 * checkpoint_every ({checkpoint_every}), "
                "so every condition's switch point falls on a checkpoint"
            )
        self.total = int(total)
        self.supervised_fraction = float(supervised_fraction)
        self.condition = condition
        self.checkpoint_every = checkpoint_every
        self.train_samples_per_position = train_samples_per_position
        for name in _COUNTERS:
            setattr(self, name, 0)
        self.next_checkpoint_at = checkpoint_every
        self.last_checkpoint_target = None
        self.branched_at = None  # n_eval when this run branched from a supervised trunk

    @classmethod
    def from_config(cls, cfg: dict) -> "Budget":
        """Build from load_config output. Raises if the budget settings are still null."""
        b = cfg["budget"]
        if b.get("network_evaluations") is None:
            raise ValueError("budget.network_evaluations is still null; set it from the pilot")
        return cls(
            total=b["network_evaluations"],
            supervised_fraction=cfg["condition"] / 100,
            condition=cfg["condition"],
            checkpoint_every=b.get("checkpoint_every"),
            train_samples_per_position=b.get("train_samples_per_position"),
        )

    # --- Totals and phase -----------------------------------------------------

    @property
    def training_samples(self) -> int:
        return self.supervised_samples + self.selfplay_samples

    @property
    def n_eval(self) -> int:
        return self.mcts_evals + TRAINING_COST * self.training_samples

    @property
    def supervised_target(self) -> float:
        return self.supervised_fraction * self.total

    def phase(self) -> str:
        if self.n_eval >= self.total:
            return DONE
        if self.n_eval < self.supervised_target:
            return SUPERVISED
        return SELFPLAY

    def fraction(self) -> float:
        """n_eval / B, the input to the learning-rate schedule (#19)."""
        return self.n_eval / self.total

    # --- Counting -------------------------------------------------------------

    def add_training_samples(self, n: int) -> None:
        """Count n samples the learner just trained on, in the current phase.

        The batch that crosses a threshold is counted in full, so n_eval can end
        slightly past p * B or B; the record keeps the real total.
        """
        _check_count(n)
        phase = self.phase()
        if phase == DONE:
            raise RuntimeError("training after the budget is spent")
        if phase == SUPERVISED:
            self.supervised_samples += int(n)
        else:
            self.selfplay_samples += int(n)

    def add_selfplay_games(self, evals: int, games: int, plies: int) -> None:
        """Count a game file the learner just ingested.

        Raises during the supervised phase: phases run in order, so self-play
        data arriving then is a bug, not something to count. Once the budget is
        spent, record late files with add_discarded instead.
        """
        for n in (evals, games, plies):
            _check_count(n)
        phase = self.phase()
        if phase == SUPERVISED:
            raise RuntimeError("self-play games ingested during the supervised phase")
        if phase == DONE:
            raise RuntimeError("self-play games ingested after the budget is spent; use add_discarded")
        self.mcts_evals += int(evals)
        self.games += int(games)
        self.plies += int(plies)

    def add_discarded(self, evals: int, games: int) -> None:
        """Record self-play work that never reached training. Not part of n_eval."""
        _check_count(evals)
        _check_count(games)
        self.discarded_evals += int(evals)
        self.discarded_games += int(games)

    # --- Flow control between self-play workers and the learner ---------------

    def train_to_data_ratio(self) -> float | None:
        """Self-play samples trained per self-play position ingested so far."""
        return self.selfplay_samples / self.plies if self.plies else None

    def selfplay_samples_allowed(self) -> int:
        """How many more self-play samples the learner may train on right now.

        Keeps samples trained <= train_samples_per_position * positions ingested,
        so the learner waits for data when it's ahead. Without this, the ratio
        would depend on how many machines happened to be free, and conditions
        run on different numbers of machines would train differently (#25).
        """
        if self.train_samples_per_position is None:
            raise ValueError("train_samples_per_position is not set")
        return max(0, int(self.train_samples_per_position * self.plies) - self.selfplay_samples)

    # --- Checkpoints ----------------------------------------------------------

    def checkpoint_due(self) -> bool:
        """True once each time n_eval crosses a multiple of checkpoint_every.

        If one batch crosses several multiples, it fires once and the record's
        checkpoint_target is the highest one crossed.
        """
        if self.checkpoint_every is None or self.n_eval < self.next_checkpoint_at:
            return False
        self.last_checkpoint_target = (self.n_eval // self.checkpoint_every) * self.checkpoint_every
        self.next_checkpoint_at = self.last_checkpoint_target + self.checkpoint_every
        return True

    def record(self, now: float | None = None) -> dict:
        """Per-checkpoint metadata (CLAUDE.md: n_eval split, games, plies, time, condition, phase)."""
        now = time.time() if now is None else now
        return {
            "condition": self.condition,
            "phase": self.phase(),
            "n_eval": self.n_eval,
            "fraction": self.fraction(),
            "checkpoint_target": self.last_checkpoint_target,
            "mcts_evals": self.mcts_evals,
            "training_samples": self.training_samples,
            "supervised_samples": self.supervised_samples,
            "selfplay_samples": self.selfplay_samples,
            "games": self.games,
            "plies": self.plies,
            "discarded_evals": self.discarded_evals,
            "discarded_games": self.discarded_games,
            "train_to_data_ratio": self.train_to_data_ratio(),
            "branched_at": self.branched_at,
            "timestamp": now,
            "time_utc": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        }

    # --- Saving, resuming and branching ----------------------------------------

    def state_dict(self) -> dict:
        """Everything needed to resume exactly; plain values, safe for JSON or torch.save."""
        state = {
            "total": self.total,
            "supervised_fraction": self.supervised_fraction,
            "condition": self.condition,
            "checkpoint_every": self.checkpoint_every,
            "train_samples_per_position": self.train_samples_per_position,
            "next_checkpoint_at": self.next_checkpoint_at,
            "last_checkpoint_target": self.last_checkpoint_target,
            "branched_at": self.branched_at,
        }
        state.update({name: getattr(self, name) for name in _COUNTERS})
        return state

    @classmethod
    def from_state_dict(cls, state: dict) -> "Budget":
        budget = cls(
            state["total"],
            state["supervised_fraction"],
            state["condition"],
            state["checkpoint_every"],
            state["train_samples_per_position"],
        )
        for name in _COUNTERS:
            setattr(budget, name, state[name])
        budget.next_checkpoint_at = state["next_checkpoint_at"]
        budget.last_checkpoint_target = state["last_checkpoint_target"]
        budget.branched_at = state["branched_at"]
        return budget

    @classmethod
    def branch(cls, trunk_state: dict, supervised_fraction: float, condition: int) -> "Budget":
        """Start a condition's self-play from a supervised trunk's checkpoint (#24).

        trunk_state must be the trunk's checkpoint taken where it crossed this
        condition's switch point p * B: still supervised-only, past p * B, and
        before the next checkpoint. The branch keeps the trunk's counts, so its
        n_eval starts at about p * B, and records where it branched.
        """
        budget = cls.from_state_dict(trunk_state)
        if trunk_state["supervised_fraction"] < supervised_fraction:
            raise ValueError("a trunk can only branch conditions with a smaller supervised fraction")
        if budget.mcts_evals or budget.selfplay_samples:
            raise ValueError("trunk state already contains self-play")
        target = supervised_fraction * budget.total
        if budget.n_eval < target:
            raise ValueError(f"trunk is at n_eval {budget.n_eval}, before the switch point {target:.0f}")
        if budget.checkpoint_every is not None and budget.n_eval >= target + budget.checkpoint_every:
            raise ValueError(
                f"trunk is at n_eval {budget.n_eval}, past the checkpoint after the switch point {target:.0f}; "
                "use the checkpoint taken at the switch point"
            )
        budget.supervised_fraction = float(supervised_fraction)
        budget.condition = condition
        budget.branched_at = budget.n_eval
        return budget


def _check_count(n: int) -> None:
    # numbers.Integral accepts numpy integers, e.g. counts read back from game files.
    if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 0:
        raise ValueError(f"counts must be non-negative integers, got {n!r}")
