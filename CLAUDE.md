# CLAUDE.md — CS435 Chess Engine Project

Context for Claude Code sessions on this repo. Read this before writing code. The full design lives in the team docs in `docs/` (listed at the bottom). This file is the short version plus the conventions code must follow.

## The project in one paragraph

We train AlphaZero-style chess models (small ResNet + MCTS) and test **how much of a fixed compute budget should go to supervised learning on human Lichess games before switching to self-play**. Five conditions spend 0/25/50/75/100% of the same budget on human games, then the rest on self-play. The goal is **not** a strong engine. Our models will be weak; we compare them to each other and to baselines. The course is CS435 (Intro to Big Data), so distributed processing (Spark on HDFS, self-play spread across GPU machines) is part of the grade. The deadline is roughly the first week of December 2026.

## Repo layout and ownership

`docs/file-structure.md` describes every file. Ask the owner before restructuring their area.

| Area | Files | Owner(s) |
|---|---|---|
| Board and move encoding, legal-move mask | `chesszero/encoding.py` | Ryan, Munisa |
| Players, game loop | `chesszero/players.py`, `chesszero/game.py` | Kylie |
| Network and loss | `chesszero/model.py` | Ryan (loss with Munisa) |
| MCTS | `chesszero/mcts.py` | Josh |
| Budget counter | `chesszero/budget.py` | Ryan |
| Parquet dataset | `chesszero/dataset.py` | Ryan, Munisa |
| Shared-folder file helpers | `chesszero/storage.py` | Unassigned |
| Data pipeline | `pipeline/chunk_pgn.py`, `filter_games.py`, `make_rows.py` | Munisa |
| Opening book | `pipeline/opening_book.py` | Kylie |
| Training and phase switch | `training/train_supervised.py`, `selfplay_worker.py`, `train_selfplay.py`, `run_condition.py` | Ryan (supervised with Munisa) |
| Launch scripts | `training/launch_workers.sh`, `training/scan_gpus.sh` | Unassigned (based on scripts from Cam, a friend outside the team) |
| Evaluation | `evaluation/sanity_checks.py`, `move_analysis.py`, `mate_puzzles.py`, `opening_analysis.py` | Kylie |
| Matches and Elo | `evaluation/match_runner.py` | Munisa |
| Pilot, scaling test, plots | `experiments/pilot.py`, `experiments/scaling_test.py`, `analysis/plots.py` | Josh (pilot with Munisa) |
| Main runs × 5 conditions | `configs/condition_*.yaml` | Ryan |

`configs/base.yaml` holds the shared settings and `configs/condition_<p>.yaml` overrides them per condition. Values that are still `null` are not decided yet.

## Current status (Nov 19, 2026)

- **On `main`:** stub files only (module docstrings), plus README, configs, `requirements.txt`, `training/scan_gpus.sh` and `docs/`.
- **Branch `chunk-pgn` (Munisa, not merged):** `pipeline/chunk_pgn.py` and `pipeline/filter_games.py` are written.
- **Data on the shared folder:** the October 2026 Lichess month (checksum verified) is split into 954 chunks of 100,000 games each (about 95.4M games) in `data/chunks_2026-10/`. The filter has **not** been run on it yet. Its rules don't match the spec below; see `docs/decisions.md`.
- **MCTS:** Josh has an implementation that isn't pushed yet.
- Everything else is still to build. The **chess core** (`encoding.py`, `players.py`, `game.py`) comes first because everything imports it.

## Experiment rules

These definitions are what make the five conditions comparable. They can change if we find a flaw, but record every change and its reasoning in `docs/decisions.md`.

- **Budget unit = network evaluations**, not wall time and not games. Count:
  - **Self-play:** every position actually sent through the network during MCTS. Count positions, not batches. Terminal positions and cache hits are free.
  - **Training:** training samples × 3 (forward + backward ≈ 3 forward passes). This applies to both the supervised and self-play phases. Count the samples the learner actually trains on, not positions generated, because the replay window resamples positions.
  - Rough guide: a supervised sample ≈ 3 evaluations; a self-play position ≈ S + 3 (S = simulations per move, about 100), so about 35× more.
- **Phases run in order within a run:** supervised until `n_eval >= p * B`, then self-play until `n_eval >= B`. No self-play during the supervised phase.
- **Same everywhere:** network size, MCTS settings, regularization (L2 weight decay, no dropout), evaluation set.
- **Value target = game result z in both phases.** Do **not** blend in the MCTS root value. It would make the phases train toward different targets.
- **No Stockfish labels in training, ever.** That would be distillation and would contaminate the comparison. Stockfish is only used in evaluation.
- **No varied/human opening positions in self-play training.** It injects human knowledge into the 0% condition. Varied openings are fine (and required) for evaluation matches.
- Record per checkpoint: `n_eval` (split into MCTS evals and training samples), games, moves (plies), timestamp, condition, phase. Games and moves are for readability; timestamps are for checking the wall-time forecast.

## Architecture

One learner per condition trains the network. Everything else produces data for it or reads its checkpoints. All hand-offs are files.

- **Human rows:** built by Spark on our own clusters' HDFS (Parquet: game ID, ply, FEN, UCI move, z, both ratings). Each person's cluster is private to them, and HDFS is **not** reachable from the fish machines. Finished outputs go to the shared folder, which is the only place the whole team and the fish machines can all read.
- **Shared folder:** `/s/bach/a/class/cs435/cs435a/chess-zero` (department network drive, seen by every lab machine, group `cs435a`). It currently has:
  - `data/`: Lichess downloads and `chunks_2026-10/`
  - `engines/stockfish-19/`: Stockfish 19 with its version and checksum notes
  - `checkpoints/`, `results/` (empty)
  - Planned per-condition layout: `checkpoints/<condition>/{weights,inbox,checkpoints,logs}`.
  - A default ACL keeps everything group-writable (the department requires `g+rwx`). Files moved in with `mv` keep their old permissions; use `cp` or `chmod -R g+rwX`.
  - Spark jobs read it with `file:///s/bach/...` paths. A plain `/s/bach/...` path is read as HDFS.
- **Self-play workers:** stateless. Load the latest weights from the folder, play many games at once with batched network calls, write finished games to the inbox. They can start or stop on any machine at any time. Playing with slightly old weights is fine.
- **Learner:** the only stateful process. It resumes from its checkpoint. DDP is optional. It's probably unnecessary for a ~2.4M-parameter net, but we may run one small DDP side experiment for the distributed-training learning goal.
- **No Ray, no cluster manager.** Processes run in tmux sessions launched over SSH by `training/launch_workers.sh`, adapted from Cam's scripts for his Ultimate Tic-Tac-Toe (UTTT) project, which ran on the same machines. Each process gets a detached tmux session on its own host, so it survives a dropped SSH connection.

### File hand-off rules

- Write files under a temporary name **in the same directory**, then rename. Never let a reader see a half-written file.
- Save games in batches (tens to a few hundred per file), not one file per game.
- Workers load a small **weights-only** file. Optimizer state and any replay buffer go in separate, less frequent checkpoint files. (Cam's project had a ~3 GB checkpoint that every worker reloaded.)
- **Never swallow exceptions when loading weights.** A silent `except: return False` leaves workers playing with stale weights while the metrics look fine.
- Expect the network drive to show a new file a few seconds late. Code must tolerate that rather than assume immediate visibility.

## Model and search

- **Input:** 8×8 planes. 12 piece planes (type × colour), plus side to move, castling rights, en passant and the 50-move counter. **Always encoded from the side to move's view** (own vs opponent pieces, board flipped for Black). No move history (a known simplification).
- **Output:** policy over 4,672 move slots (AlphaZero encoding); value in [-1, 1].
- **Legal-move mask:** set illegal slots to -inf before the softmax, in **both** the training loss and the MCTS priors. Legal moves come from python-chess `board.legal_moves`.
- **Size (tentative):** 8 residual blocks × 128 channels, about 2.4M parameters, about 10 MB.
- **Policy head:** don't bottleneck it to 2 channels before the 4,672-way FC layer. Use a wider first conv (e.g. 32) or a convolutional head.
- **Loss:** masked policy cross-entropy + value MSE + L2 weight decay.
  - Policy target: one-hot human move (supervised) or normalized MCTS visit counts (self-play).
  - Value target: z from the side to move's view (+1 win, 0 draw, -1 loss).
- **MCTS:** PUCT, about 100 simulations per move (to tune in the pilot).
  - Batch leaf evaluations across many games per worker (e.g. ~128 games in lockstep, one leaf per game per step).
  - **Unvisited children (FPU):** use the parent's value, not 0. Check the sign/perspective.
  - **Dirichlet noise:** root only, self-play only, once per move (not per simulation). Alpha ≈ 0.3, weight 0.25. Never during evaluation.
  - **Temperature:** sample from visit counts for the first ~30 plies, then pick the most-visited move.
  - Use `board.push` / `board.pop`, not `copy(stack=False)` (that drops the repetition history).
  - **Don't share statistics between transposed positions.** Repetition and the 50-move rule make identical boards have different values. If you add Zobrist keys (`chess.polyglot.zobrist_hash`), use them for lookups or dedup, not to merge tree nodes.
- **Game loop:** `board.push(move)` until `board.is_game_over(claim_draw=True)`, plus a move cap of about 300 plies.

## Data pipeline (Spark)

1. Download one month of `.pgn.zst` to `data/` on the shared folder. `pipeline/chunk_pgn.py` splits it at `[Event` lines into zstd-compressed chunks (100,000 games each) without writing a decompressed copy.
2. `pipeline/filter_games.py` applies the header filters and reports the count dropped by each rule. Spec:
   - No BOT titles.
   - `Termination` = Normal (drops time forfeits, abandoned games, rules infractions and unterminated games).
   - Base time control ≥ 300 s; drop `TimeControl` "-".
   - Both players rated ≥ 1000 (Lichess).
   - The code on `chunk-pgn` currently differs (rating ≥ 1200, base time > 300 s, keeps abandoned games). Settle it and record the result in `docs/decisions.md`.
3. `pipeline/make_rows.py` expands each game to one row per position.
   - Skip positions where the player to move had under 30 seconds left (from the `[%clk ...]` comments). The value target stays the game result. See `docs/decisions.md` #10.
   - Which time controls to include is decided after the pilot; default to base ≥ 600 s. See `docs/decisions.md` #11.
4. **Split train/val by hash of game ID (90/10), never by position.** Verify zero shared IDs.
5. Global shuffle, then write Parquet shards (every shard should mix thousands of games).
6. Encode FEN → planes at load time in the PyTorch DataLoader (`chesszero/dataset.py`). If the GPU sits idle, add DataLoader workers or pre-encode.

## Evaluation

- **Headline result:** Elo vs compute for every condition, using checkpoints throughout training.
- **Elo pool:** all checkpoints + random player + material-count MCTS baseline + Stockfish at several levels.
- **Match rules:**
  - Start from a fixed set of varied openings.
  - Play each opening twice with colours swapped.
  - Noise and temperature off.
  - BayesElo/Ordo with confidence intervals.
  - Greedy MCTS is deterministic. Without varied openings, repeated games are identical.
- **Move quality:** Stockfish at a fixed node/depth limit (not "skill level"); report average centipawn loss.
- **Stockfish usage:** one engine per CPU core, Threads = 1, one engine per Spark partition (not per row), node limits for reproducibility. Use the shared copy of Stockfish 19 at `engines/stockfish-19/stockfish` (path in `configs/base.yaml` as `paths.stockfish`), driven through python-chess's `chess.engine`. It isn't a pip package, so it's not in `requirements.txt`. Our models will be far weaker than Stockfish's built-in skill levels, so build weaker opponents with node limits (e.g. 1, 10, 100, 1000 nodes).
- **Fixed test positions:** decided in advance and identical for every model.

## Required tests (write these before trusting training)

Tests live in `tests/` (`test_encoding.py`, `test_mcts.py`, `test_game.py`). Run them with `python -m pytest tests` from the repo root.

- **Move encoding round-trip:** every legal move in many random positions maps to a slot and back exactly.
- **Mask support:** policy support equals the set of legal moves.
- **Perspective and sign:** in a position where one side is clearly winning, these all agree with "good for the side to move":
  - network value
  - terminal value
  - backed-up Q
  - root value
  - stored z
- **Environment:** random vs random over thousands of games, where every game ends; Stockfish levels rank correctly in the match runner; material-count MCTS finds mate-in-1/2.
- No board augmentation (rotations and reflections are invalid in chess). If anyone adds left-right mirroring, it's only valid without castling rights, and the policy must be remapped. Test it by re-encoding.

## Engineering gotchas (from Cam's UTTT project, all real bugs)

- `torch.compile`: keep a handle on the raw module. Load weights into it and save its `state_dict`. Compiled modules add an `_orig_mod.` key prefix, and compiling twice can return a non-module.
- **Periodic triggers:** use `if count >= next_at: next_at += N`, not `if count % N == 0` (it fires repeatedly while `count` doesn't change).
- Change the LR per training iteration, not per gradient step, if using a schedule.
- Save anything the trainer adjusts at runtime in the checkpoint, so resume doesn't reset it.
- Training-loss metrics can improve while strength doesn't, and they can hide data bugs. **Only matches measure strength.**

## Infrastructure

- **Machines:** the GPU lab "fish machines", each with one NVIDIA RTX 4000 Ada (20 GB): anchovy, barracuda, blowfish, bonito, bullhead, char, cod, dorado, eel, flounder, grouper, halibut, herring, mackerel, marlin, perch, pollock, sardine, shark, sole, swordfish, tarpon, tuna, turbot, wahoo. (brill reported no GPU.) They're shared with other users.
  - Run `training/scan_gpus.sh` to see which are free (least-used first), and check `who` before claiming one.
  - Pin GPUs with `CUDA_VISIBLE_DEVICES`.
  - Keep the team sign-up note of which machines are in use.
  - Working benchmark: about 5 machines; scaling test at 1/2/5/10 (up to 20).
- **Environment:** the course Python 3.13 (`/usr/local/python-env/py313/bin/python3`), which already has PyTorch with CUDA. Install the rest once per person with `/usr/local/python/3.13.5/bin/python3 -m pip install --user -r requirements.txt` (home directories are shared across machines). **Don't use conda:** an active conda environment shadows this Python and breaks `torch` imports. `pyspark` is pinned to the clusters' Spark version (4.1.1).
- **Storage:** about 150 GB project space; about 72 GB is used by the October download and its chunks. Delete raw downloads once they're no longer needed. Keep full optimizer state only for recent checkpoints.

## Planning the budget

The pilot measures, per machine type: **R_sp** (self-play evals/sec per worker) and **R_sup** (supervised samples/sec).

- Across the five conditions, self-play totals about 2.5·B evaluations. Supervised training also totals about 2.5·B evaluations, which is 2.5·B / 3 samples. So total time ≈ 2.5·B / (R_sp × machines) + 2.5·B / (3 · R_sup). The self-play term dominates.
- **Work backward from the deadline:** B ≈ (usable machine-hours × 3600 × R_sp) / 2.5, after reserving about 30% for debugging, reruns and evaluation matches.
- If that B gives too few games to learn anything, shrink the network or reduce simulations per move early.

## Open questions

- Budget B and machine count (after the pilot and scaling test).
- Final filter thresholds (see Data pipeline).
- Whether Spark on our clusters can read the shared folder directly, or whether chunks must be uploaded to HDFS first.
- Whether the human training shards fit the shared-folder quota.
- Self-play sample file format, opening book format, evaluation results format.
- Whether one month of games is enough after filtering (a spot check of two chunks kept about 10.5%, so roughly 10M games).
- Optional: DDP side run, Maia/Lc0 opponents, noise on/off side experiment.

## How to work in this repo

- We're implementing now. Small details can change freely, and larger design changes are fine when they fix a real flaw. **Record every decision that affects the experiment or the write-up in `docs/decisions.md`**, with the reasoning and the alternatives considered.
- Keep changes small and separately testable. When a change could affect playing strength, validate it with a fixed-simulation match against the previous version, not just loss curves.
- Prefer python-chess for all rules. We write the encoding, MCTS, network and training loop ourselves (reference code like alpha-zero-general is for checking, not copying).
- Work on a branch and merge to `main` through a pull request.

## Team docs

Copies in this repo, under `docs/`:

- `decisions.md`: decisions made during implementation and why
- `proposal.pdf`: the project proposal
- `method-reference-v1.pdf`: team reference (full design, decisions, FAQ, references). Live version: https://claude.ai/artifact/4AxnwYDL4dEJUeXdf9TtCP
- `diagrams/`: system overview, component dependencies, and the conditions chart. Live system diagram: https://claude.ai/artifact/JdrusZQru6PpRvB3xnf39q
- `file-structure.md`: what each file in the repo is for

Note: the method reference still says hand-offs go through HDFS. That's outdated, because the fish machines can't reach it. Use the shared folder as described above.
