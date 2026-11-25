"""Tests for chesszero/budget.py."""

import json

import numpy as np
import pytest

from chesszero.budget import DONE, SELFPLAY, SUPERVISED, TRAINING_COST, Budget
from chesszero.config import load_config

B = 1_000_000
EVERY = B // 100
BATCH = 1024  # supervised batch: 3,072 evals, so batches don't land exactly on thresholds


def run_supervised(budget: Budget, until: float | None = None) -> list[dict]:
    """Train supervised batches until the phase changes (or n_eval >= until); return checkpoint records."""
    records = []
    while budget.phase() == SUPERVISED and (until is None or budget.n_eval < until):
        budget.add_training_samples(BATCH)
        if budget.checkpoint_due():
            records.append(budget.record(now=0.0))
    return records


def run_selfplay(budget: Budget) -> list[dict]:
    """Alternate ingesting a game file and training on it until the budget is spent."""
    records = []
    while budget.phase() != DONE:
        budget.add_selfplay_games(evals=100 * 800, games=10, plies=800)
        if budget.phase() != DONE:
            budget.add_training_samples(800)
        if budget.checkpoint_due():
            records.append(budget.record(now=0.0))
    return records


def test_n_eval_counts_mcts_evals_and_three_per_sample():
    budget = Budget(B, 0.5)
    budget.add_training_samples(1000)
    assert budget.n_eval == 3000
    run_supervised(budget)
    budget.add_selfplay_games(evals=5000, games=2, plies=150)
    budget.add_training_samples(100)
    assert budget.n_eval == budget.mcts_evals + TRAINING_COST * budget.training_samples
    assert budget.mcts_evals == 5000 and budget.selfplay_samples == 100


@pytest.mark.parametrize("p", [0, 0.25, 0.5, 0.75, 1])
def test_phases_run_in_order(p):
    budget = Budget(B, p, checkpoint_every=EVERY)
    assert budget.phase() == (SUPERVISED if p > 0 else SELFPLAY)
    run_supervised(budget)
    supervised_evals = budget.n_eval
    assert supervised_evals >= p * B
    assert supervised_evals < p * B + TRAINING_COST * BATCH  # overshoot is at most one batch
    run_selfplay(budget)
    assert budget.phase() == DONE
    assert budget.n_eval >= B
    assert TRAINING_COST * budget.supervised_samples == supervised_evals
    if p == 1:
        assert budget.mcts_evals == 0 and budget.selfplay_samples == 0


def test_selfplay_games_during_supervised_phase_raise():
    budget = Budget(B, 0.25)
    with pytest.raises(RuntimeError, match="supervised phase"):
        budget.add_selfplay_games(evals=100, games=1, plies=1)


def test_nothing_counts_after_the_budget_is_spent():
    budget = Budget(B, 0)
    run_selfplay(budget)
    with pytest.raises(RuntimeError):
        budget.add_training_samples(1)
    with pytest.raises(RuntimeError, match="add_discarded"):
        budget.add_selfplay_games(evals=100, games=1, plies=1)
    before = budget.n_eval
    budget.add_discarded(evals=12345, games=3)
    assert budget.n_eval == before
    assert budget.record()["discarded_evals"] == 12345


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_bad_counts_raise(bad):
    with pytest.raises(ValueError):
        Budget(B, 0.5).add_training_samples(bad)


def test_numpy_counts_are_accepted():
    budget = Budget(B, 0)
    budget.add_selfplay_games(evals=np.int64(500), games=np.int32(1), plies=np.int64(60))
    assert budget.n_eval == 500
    json.dumps(budget.state_dict())  # still plain Python ints


def test_checkpoints_fire_once_per_interval_at_aligned_targets():
    budget = Budget(B, 0.25, checkpoint_every=EVERY)
    records = run_supervised(budget) + run_selfplay(budget)
    targets = [r["checkpoint_target"] for r in records]
    assert targets == sorted(set(targets))  # once each, increasing
    assert all(t % EVERY == 0 for t in targets)
    assert 0.25 * B in targets  # the switch point is a checkpoint
    assert all(r["n_eval"] >= r["checkpoint_target"] for r in records)


def test_checkpoint_doesnt_refire_without_progress():
    """The `count % N == 0` bug from Cam's project: firing again while the count is unchanged."""
    budget = Budget(B, 0.5, checkpoint_every=EVERY)
    budget.add_training_samples(EVERY // TRAINING_COST + 1)
    assert budget.checkpoint_due()
    assert not budget.checkpoint_due()


def test_one_batch_crossing_several_intervals_fires_once():
    budget = Budget(B, 0.5, checkpoint_every=EVERY)
    budget.add_training_samples(EVERY)  # 3 intervals at once
    assert budget.checkpoint_due()
    assert budget.record()["checkpoint_target"] == 3 * EVERY
    assert not budget.checkpoint_due()


def test_checkpoint_interval_must_align_with_switch_points():
    with pytest.raises(ValueError, match="multiple of 4"):
        Budget(B, 0.25, checkpoint_every=B // 50 + 1)
    Budget(B, 0.25, checkpoint_every=B // 20)  # 0.25 * B = 5 intervals


def test_resume_continues_exactly():
    straight = Budget(B, 0.5, condition=50, checkpoint_every=EVERY)
    run_supervised(straight, until=0.3 * B)
    state = json.loads(json.dumps(straight.state_dict()))  # survives a JSON round trip
    resumed = Budget.from_state_dict(state)
    assert resumed.state_dict() == straight.state_dict()

    a = run_supervised(straight) + run_selfplay(straight)
    b = run_supervised(resumed) + run_selfplay(resumed)
    assert a == b
    assert resumed.state_dict() == straight.state_dict()


def test_branch_matches_an_independent_run():
    """Branching the 25% condition from the trunk gives the same counts as running it alone (#24)."""
    trunk = Budget(B, 1.0, condition=100, checkpoint_every=EVERY)
    run_supervised(trunk, until=0.25 * B)
    branched = Budget.branch(trunk.state_dict(), 0.25, condition=25)

    alone = Budget(B, 0.25, condition=25, checkpoint_every=EVERY)
    run_supervised(alone)

    assert branched.phase() == SELFPLAY
    assert branched.branched_at == branched.n_eval == alone.n_eval
    expected = alone.state_dict() | {"branched_at": branched.branched_at}
    assert branched.state_dict() == expected

    # The trunk carries on to the next branch point unaffected.
    run_supervised(trunk, until=0.5 * B)
    assert Budget.branch(trunk.state_dict(), 0.5, condition=50).phase() == SELFPLAY


def test_branch_rejects_the_wrong_checkpoint():
    trunk = Budget(B, 1.0, checkpoint_every=EVERY)
    run_supervised(trunk, until=0.2 * B)
    with pytest.raises(ValueError, match="before the switch point"):
        Budget.branch(trunk.state_dict(), 0.25, condition=25)
    run_supervised(trunk, until=0.3 * B)
    with pytest.raises(ValueError, match="past the checkpoint"):
        Budget.branch(trunk.state_dict(), 0.25, condition=25)

    selfplay_run = Budget(B, 0.25, checkpoint_every=EVERY)
    run_supervised(selfplay_run)
    selfplay_run.add_selfplay_games(evals=10, games=1, plies=5)
    with pytest.raises(ValueError, match="self-play"):
        Budget.branch(selfplay_run.state_dict(), 0.25, condition=25)


def test_learner_waits_for_data_at_the_fixed_ratio():
    budget = Budget(B, 0, train_samples_per_position=2.0)
    assert budget.selfplay_samples_allowed() == 0  # no data yet
    budget.add_selfplay_games(evals=1000, games=1, plies=100)
    assert budget.selfplay_samples_allowed() == 200
    budget.add_training_samples(150)
    assert budget.selfplay_samples_allowed() == 50
    budget.add_training_samples(50)
    assert budget.selfplay_samples_allowed() == 0
    assert budget.train_to_data_ratio() == 2.0


def test_record_has_the_checkpoint_fields():
    budget = Budget(B, 0.5, condition=50, checkpoint_every=EVERY)
    run_supervised(budget, until=0.1 * B)
    record = budget.record(now=1_764_000_000.0)
    for key in ("condition", "phase", "n_eval", "mcts_evals", "training_samples", "games", "plies", "timestamp"):
        assert key in record
    assert record["time_utc"].startswith("2025-11-24")
    json.dumps(record)


def test_from_config():
    cfg = load_config("configs/condition_25.yaml")
    with pytest.raises(ValueError, match="network_evaluations"):
        Budget.from_config(cfg)  # still null until the pilot
    cfg["budget"] |= {"network_evaluations": B, "checkpoint_every": EVERY, "train_samples_per_position": 1.0}
    budget = Budget.from_config(cfg)
    assert budget.supervised_target == 0.25 * B and budget.condition == 25
