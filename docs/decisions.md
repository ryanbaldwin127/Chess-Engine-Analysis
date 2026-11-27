# Decisions

A running record of project decisions and the reasoning behind them, for the final
write-up. Add an entry whenever a decision affects the experiment, the data, or how
results should be read. Don't rewrite old entries: if a decision changes, add a new
entry and mark the old one as superseded.

Each entry has a date, a status (decided, open or superseded), the decision, why we
made it, and the alternatives we considered. Sources are cited by key, like [Maia],
from [references.md](references.md).

## Index

| # | Decision | Status |
|---|---|---|
| 1 | [Measure compute in network evaluations](#1-measure-compute-in-network-evaluations) | Decided |
| 2 | [Train the value head on the game result in both phases](#2-train-the-value-head-on-the-game-result-in-both-phases) | Decided |
| 3 | [No Stockfish labels in training](#3-no-stockfish-labels-in-training) | Decided |
| 4 | [Self-play starts from the standard opening position](#4-self-play-starts-from-the-standard-opening-position) | Decided |
| 5 | [Split train and validation by game, not by position](#5-split-train-and-validation-by-game-not-by-position) | Decided |
| 6 | [Run GPU work on the fish machines](#6-run-gpu-work-on-the-fish-machines) | Decided |
| 7 | [Hand off files through the shared folder, not HDFS](#7-hand-off-files-through-the-shared-folder-not-hdfs) | Decided |
| 8 | [Use the October 2026 Lichess month, split into 100,000-game chunks](#8-use-the-october-2026-lichess-month-split-into-100000-game-chunks) | Decided |
| 9 | [Game filter thresholds](#9-game-filter-thresholds) | Open |
| 10 | [Drop positions played with under 30 seconds on the clock](#10-drop-positions-played-with-under-30-seconds-on-the-clock) | Decided |
| 11 | [Which time controls go into the training rows](#11-which-time-controls-go-into-the-training-rows) | Open (after the pilot) |
| 12 | [bf16 mixed precision on the GPU](#12-bf16-mixed-precision-on-the-gpu) | Open |
| 13 | [Repetition-count input planes](#13-repetition-count-input-planes) | Decided |
| 14 | [Squeeze-and-excitation residual blocks](#14-squeeze-and-excitation-residual-blocks) | Open |
| 15 | [Training batch size](#15-training-batch-size) | Open |
| 16 | [Screen design choices on a small network](#16-screen-design-choices-on-a-small-network) | Open |
| 17 | [Self-play draw rate and the value signal](#17-self-play-draw-rate-and-the-value-signal) | Open |
| 18 | [Keep the simulation count fixed](#18-keep-the-simulation-count-fixed) | Open |
| 19 | [Learning-rate schedule](#19-learning-rate-schedule) | Open |
| 20 | [Starting value for unvisited moves in search](#20-starting-value-for-unvisited-moves-in-search) | Open |
| 21 | [Use Stockfish 19 from the official release for evaluation](#21-use-stockfish-19-from-the-official-release-for-evaluation) | Decided |
| 22 | [Use a convolutional policy head](#22-use-a-convolutional-policy-head) | Decided |
| 23 | [Supervised row format and how the dataset reads it](#23-supervised-row-format-and-how-the-dataset-reads-it) | Decided |
| 24 | [Branch the conditions from one supervised run](#24-branch-the-conditions-from-one-supervised-run) | Open |
| 25 | [Count self-play evaluations when the learner ingests them](#25-count-self-play-evaluations-when-the-learner-ingests-them) | Decided (ratio value open) |

---

## 1. Measure compute in network evaluations

- **Date:** planning (method reference v1, Nov 3, 2026)
- **Status:** Decided
- **Decision:** The budget B is counted in network evaluations. Self-play counts every position sent through the network during search. Training counts samples × 3, since a forward and backward pass costs about three forward passes.
- **Why:** The question is how to split a fixed amount of compute, so supervised training and self-play need one shared unit. Wall time depends on which shared machines we get and how busy they are. Game counts aren't comparable either, because a self-play position costs about 35 times more than a supervised sample.
- **Alternatives:** Wall-clock time (not reproducible on shared machines); number of games or gradient steps (ignores the very different cost of each phase).

## 2. Train the value head on the game result in both phases

- **Date:** planning (method reference v1, Nov 3, 2026)
- **Status:** Decided
- **Decision:** The value target is the final game result z (+1, 0, -1 from the side to move's view) in both the supervised and self-play phases.
- **Why:** If self-play blended in the MCTS root value while the supervised phase used z, the two phases would train toward different targets. Differences between conditions could then come from the target rather than from the data source.
- **Alternatives:** Blending z with the root value during self-play, as some AlphaZero reimplementations do.

## 3. No Stockfish labels in training

- **Date:** planning (method reference v1, Nov 3, 2026)
- **Status:** Decided
- **Decision:** Stockfish is used only in evaluation, never to label training data.
- **Why:** Training on Stockfish moves or scores would be distillation from a much stronger engine. That would swamp the human-versus-self-play comparison the project is measuring.
- **Alternatives:** None seriously considered.

## 4. Self-play starts from the standard opening position

- **Date:** planning (method reference v1, Nov 3, 2026)
- **Status:** Decided
- **Decision:** Self-play training games start from the normal starting position. Varied opening positions are used only in evaluation matches.
- **Why:** Opening positions drawn from human games would inject human knowledge into the 0% condition, which is supposed to learn only from self-play. Evaluation still needs varied openings, because greedy MCTS is deterministic and would otherwise repeat the same game.
- **Alternatives:** Starting self-play from a human opening book for diversity.

## 5. Split train and validation by game, not by position

- **Date:** planning (method reference v1, Nov 3, 2026)
- **Status:** Decided
- **Decision:** Rows are split 90/10 into train and validation by a hash of the game ID, and we verify that no game ID appears in both.
- **Why:** Positions from the same game are highly correlated. Splitting by position would leak near-copies of training positions into validation and make validation loss look better than it is.
- **Alternatives:** A random split by position.

## 6. Run GPU work on the fish machines

- **Date:** Nov 10, 2026
- **Status:** Decided
- **Decision:** Self-play and training run on the CS department's "fish machines" rather than on our individual clusters. Spark jobs still run on our own clusters.
- **Why:** The professor directed us to them, and they have better GPUs. A scan on Nov 10 found 25 with an NVIDIA RTX 4000 Ada (20 GB) and the course Python environment, all able to write to the shared folder.
- **Alternatives:** Running everything on our own clusters.

## 7. Hand off files through the shared folder, not HDFS

- **Date:** Nov 11, 2026
- **Status:** Decided. This supersedes the HDFS hand-offs in the method reference.
- **Decision:** All hand-offs between people and between machines go through the shared folder `/s/bach/a/class/cs435/cs435a/chess-zero`: data, Parquet shards, weights, checkpoints and self-play games. HDFS is used only inside a Spark job on one person's cluster.
- **Why:** Each team member's Hadoop cluster is private to them, and the fish machines can't reach HDFS. The shared folder is the only storage the whole team and every lab machine can read.
- **Alternatives:** One shared HDFS cluster (we don't have one); copying files between personal clusters (manual and error-prone).

## 8. Use the October 2026 Lichess month, split into 100,000-game chunks

- **Date:** Nov 16, 2026 (Munisa)
- **Status:** Decided
- **Decision:** The human data is the Lichess standard rated games file for October 2026 (31 GB compressed, about 95.4M games). `pipeline/chunk_pgn.py` splits it into 954 zstd-compressed chunks of 100,000 complete games each, stored in `data/chunks_2026-10/` on the shared folder.
- **Why:** Chunks give Spark many independent units of work, and splitting at `[Event` lines means no game is ever cut in half. Keeping the chunks compressed avoids writing a decompressed copy, which would be several times the 31 GB download. The download matched Lichess's published SHA-256 checksum, and spot checks of the first and last chunks found them complete.
- **Alternatives:** Several months of data (not needed if one month gives enough games after filtering; see #9); uploading the raw file to HDFS and letting Spark split it (a single compressed file can't be split in parallel).

## 9. Game filter thresholds

- **Date:** Nov 19, 2026
- **Status:** Open. Owner: Munisa.
- **Question:** The method reference and the code on the `chunk-pgn` branch disagree:

  | Rule | Method reference | `filter_games.py` |
  |---|---|---|
  | Bots | Drop BOT titles | Drop BOT titles |
  | Termination | Normal only | Drops time forfeits only |
  | Time control | Base ≥ 300 s | Base > 300 s |
  | Rating | Both ≥ 1000 | Both ≥ 1200 |

  The system diagram also says "> 300 s", so the team docs disagree with each other.

- **Evidence so far:**
  - On the first and last chunks, abandoned games are 0.4% of kept games in the first chunk and 8.1% in the last. All of them have fewer than 10 plies, and over half have no moves at all. Normal games under 10 plies are below 1%, mostly early resignations.
  - Games with a 5-minute base (300+0 and 300+3) are about 16% of all games.
  - Share of games kept under each option, measured on three chunks (first, middle, last; 295,628 games):

    | Option | Kept | Month estimate |
    |---|---|---|
    | Current code (drop time forfeits, base > 300 s, rating ≥ 1200) | 11.0% | ~10.5M |
    | Normal only, base > 300 s, rating ≥ 1200 | 10.7% | ~10.2M |
    | Normal only, base ≥ 300 s, rating ≥ 1200 | 20.8% | ~19.8M |
    | Normal only, base ≥ 300 s, rating ≥ 1000 (method reference) | 24.8% | ~23.7M |
    | Normal only, base ≥ 180 s, rating ≥ 1200 | 38.5% | ~36.7M |

- **Proposed:** Match the method reference: `Termination` = Normal, base ≥ 300 s, both ratings ≥ 1000.
  - **Normal only:** abandoned games carry almost no moves and would add noise to the policy target. Requiring "Normal" also drops unterminated and rules-infraction games in one check, and costs almost no good games (11.0% to 10.7%).
  - **Generous time and rating thresholds:** the filter output keeps each game's time control and both ratings. A stricter cut can be applied later in `make_rows.py` without re-filtering the month (see #11), while a looser one would mean re-running the filter.
  - **Not below 5 minutes:** 3-minute games would add volume we likely don't need, and they're mostly fast blitz with noisier moves.

## 10. Drop positions played with under 30 seconds on the clock

- **Date:** Nov 19, 2026
- **Status:** Decided. The 30-second cutoff is tentative (`data.min_clock_seconds`). Implemented in `pipeline/make_rows.py` (owner: Ryan, taken over from Munisa).
- **Implementation:** the player to move's clock is the time they had left after their previous move (Lichess's `[%clk]` comment, which includes the increment), or the base time for each side's first move. On a sample of two chunks (20,584 games after the filter and the 10-minute default), the rule removed 10,091 of about 1.36M positions, about 0.7%.
- **Decision:** When expanding games into training rows, skip any position where the player to move had less than 30 seconds left. The clock comes from the `[%clk h:mm:ss]` comment Lichess records after each move. The rest of the game is still used, and the value target for kept positions is still the final result.
- **Why:** Most of the quality loss in faster games happens in time scrambles, where players make rushed moves and blunders. Dropping those positions removes the worst moves from every time control, including 10-minute games, without throwing away whole games. It also lets us keep 5-minute games if we need the volume (see #11).
- **Alternatives:** Dropping whole fast games (loses the good early and middle-game moves too); no clock filter (keeps time-scramble blunders as policy targets).
- **Notes:**
  - Record how many positions this removes, for the write-up.
  - The value target is unchanged, so a game decided by a time-scramble blunder still labels its earlier positions with that result. That's a known source of noise in z that this rule doesn't fix.
  - Precedent: Maia [Maia], a model trained to predict human moves on Lichess games, discarded moves made with less than 30 seconds left, according to Microsoft Research's write-up of the project. Quote the paper's methods section for the exact rule in the write-up.

## 11. Which time controls go into the training rows

- **Date:** Nov 19, 2026
- **Status:** Open until the pilot sets the budget B. The default is implemented as `data.rows_min_base_seconds: 600` in `configs/base.yaml`, applied in `pipeline/make_rows.py`.
- **Question:** Should the supervised training rows include 5-minute games (300+0, 300+3), or only games with a base of 10 minutes or more?
- **Considerations:**
  - **For leaving 5-minute games out:** players have less time to find good moves, more games are decided by late blunders, and the value target z is noisier. Games of 10 minutes or more still give about 10M games, or roughly 650M positions at a median of about 60 plies.
  - **For including them:** they double the number of games (to about 20M). At 1200+ ratings they're still far stronger than our models are likely to become, and the supervised phase only needs to give a sensible starting policy. #10 already removes the worst time-scramble moves.
  - **Fairness:** every condition with human data uses the same rows, so this choice affects how much human data helps overall, not whether the conditions are comparable. Mention it in the write-up either way.
- **Plan:** Default to games of 10+ minutes. After the pilot, compare the number of supervised samples the 100% condition will use (B / 3) with the positions available. Add the 5-minute games only if the 10+ minute rows would have to be repeated many times. Either way, this is a column filter in `make_rows.py`, so the month doesn't need re-filtering.

---

Entries 12–20 come from reviewing Cam's notes on his Ultimate Tic-Tac-Toe AlphaZero project, which also trained on the fish machines. Each stays open until the code it affects is being written. The recommendation is a starting point for that discussion, not a decision.

## 12. bf16 mixed precision on the GPU

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit when writing `chesszero/model.py` and the training loop.
- **Question:** Run self-play inference and training under bf16 autocast?
- **Recommendation:** Adopt. The fish machines' RTX 4000 Ada GPUs support bf16, which speeds up both self-play and training. It's standard practice and doesn't affect the comparison between conditions, provided every condition uses it.
- **Check when implementing:** confirm the loss and value outputs stay stable, and keep the softmax over masked moves in float32 if -inf masking misbehaves in bf16.

## 13. Repetition-count input planes

- **Date:** Nov 19, 2026
- **Status:** Decided Nov 24, 2026. Implemented in `chesszero/encoding.py` (planes 12–13). Needs a column from `pipeline/make_rows.py` (owner: Munisa).
- **Question:** Add two input planes marking whether the current position has occurred once or twice before?
- **Decision:** Adopt. Plane 12 is all 1s if the position occurred at least once before in the game, and plane 13 if at least twice.
- **Why:** We encode no move history, so the network can't tell that a move would allow a threefold-repetition draw, even though search and the game result both depend on it. Two constant planes are cheap. AlphaZero's chess input included repetition counts.
- **Supervised rows need a repetition column.** A board rebuilt from a FEN has no move history, so without the column these planes would always be 0 in the supervised phase but not in self-play, and conditions entering self-play would meet inputs they never trained on. `make_rows.py` replays each game anyway, so it adds `repetition_count` per row: the number of earlier occurrences of the position, keyed on the first four FEN fields (pieces, side to move, castling, en passant), which is the same key python-chess's `is_repetition` compares. `dataset.py` passes it to `board_to_planes(board, repetition_count=...)`. `tests/test_encoding.py::test_fen_rebuild_matches_live_board` checks that this reproduces the live board's encoding.
- **Alternatives:** Full 8-position history as in AlphaZero (multiplies the input size; more than this project needs); no repetition information; keep the planes but accept that they're always 0 in supervised data.
- **Related:** the en passant plane (19) is set only when an en passant capture is legal. python-chess sets `board.ep_square` after every double pawn push, but `board.fen()` writes it only when the capture is legal, so encoding the raw square would make a live board and its rebuilt FEN differ.

## 14. Squeeze-and-excitation residual blocks

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit when writing `chesszero/model.py`, and again after the pilot.
- **Question:** Add a squeeze-and-excitation gate (global average pool, small MLP, sigmoid, channel scaling) to each residual block?
- **Recommendation:** Optional. It's cheap and gives each block whole-board context; Leela Chess Zero uses it. Try it if the pilot shows spare GPU time, and keep it only if a fixed-simulation match shows a gain.

## 15. Training batch size

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit in the pilot.
- **Question:** Which batch size should the learner use?
- **Recommendation:** Start at 512. Cam found 512 clearly better than 256 at large network sizes. Our network is smaller, so confirm it in the pilot by checking throughput and validation loss. Use the same batch size in both phases and every condition.

## 16. Screen design choices on a small network

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit when planning the pilot.
- **Question:** Test design choices (like #13 and #14) on a cheap, small network before committing to them in the main runs?
- **Recommendation:** Yes. In Cam's project, a small screening run predicted the large run's result, which made ideas cheap to test. Screening compute is outside the experiment's budget B and should be reported separately in the write-up.

## 17. Self-play draw rate and the value signal

- **Date:** Nov 19, 2026
- **Status:** Open. Measure in the pilot.
- **Problem:** Weak models often shuffle pieces until the 300-ply move cap or a repetition, so many self-play games may be draws. A draw gives z = 0, so if most games are draws, the value head gets almost no signal during self-play.
- **Plan:** Measure the self-play draw rate, and how games end (checkmate, repetition, 50-move rule, move cap), early in the pilot.
- **Ruled out:** Blending the MCTS root value into the value target, which Cam used. It conflicts with #2: blending only in self-play would make the two phases train toward different targets.
- **If the draw rate is high:** any fix must apply identically to every condition and get its own entry here. Options include a different move cap, or adjudicating capped games. Adjudicating by material would add a hand-written chess heuristic, which needs to be weighed against #3 and #4.

## 18. Keep the simulation count fixed

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit when writing `training/selfplay_worker.py`.
- **Question:** Raise simulations per move as training goes on, as Cam did (100 → 400 → 800 → 1600)?
- **Recommendation:** Keep it fixed (about 100, tuned in the pilot). The budget unit already counts the extra cost, but conditions start self-play at different points. A schedule tied to training progress would give each condition different search settings, which breaks "same MCTS settings everywhere".

## 19. Learning-rate schedule

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit when writing the training loop.
- **Question:** Which learning-rate schedule should the learner use?
- **Recommendation:** A simple schedule defined as a function of the budget spent (n_eval / B), identical across conditions. Cam's cosine schedule with decaying warm restarts has several settings that would each need tuning, and is harder to keep identical across five conditions.
- **Open sub-question:** whether the rate should reset when a run switches from supervised training to self-play.
- **Keep from Cam's project:** change the rate once per training iteration, not per gradient step. Stepping per gradient step caused accidental restarts.

## 20. Starting value for unvisited moves in search

- **Date:** Nov 19, 2026
- **Status:** Open. Revisit when reviewing Josh's MCTS.
- **Question:** What value should search assume for a move it hasn't visited yet (first-play urgency, FPU)?
- **Recommendation:** Use the parent node's network value, as in CLAUDE.md, and test that the parent and child values use the same player's perspective. Leela's version (parent value minus a penalty that grows with the prior probability of the moves already visited) suits chess better at scale. Try it only if a match shows search quality is the weak point.
- **Alternatives:** Assuming 0, a draw (spreads visits across moves the network already rates poorly).

---

## 21. Use Stockfish 19 from the official release for evaluation

- **Date:** Nov 20, 2026
- **Status:** Decided.
- **Decision:** Every evaluation uses one shared copy of Stockfish 19, at `engines/stockfish-19/stockfish` on the shared folder (`paths.stockfish` in `configs/base.yaml`). It's the official Linux x86-64 "universal" build, released Sep 5, 2026, tag `sf_19`, and is driven from Python through python-chess's `chess.engine`.
  - The download matched GitHub's published SHA-256 digest (`9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f` for the archive). `VERSION.txt` and `SHA256-binary.txt` in the same folder record the details.
  - On a fish machine (Intel i9-13900K) it searches about 1.1M nodes per second on one thread in Stockfish's built-in benchmark.
- **Why:**
  - One pinned version means every Elo and move-quality number is measured against the same engine.
  - The universal build picks the fastest code path for each CPU, so the same file works on every fish machine.
  - The PyPI `stockfish` package is only a Python wrapper and doesn't include the engine. We'd still need this download, and python-chess already provides the same functionality.
- **Alternatives:** The PyPI wrapper (doesn't include the engine); building from source (slower to set up, with no benefit here).
- **Related, still open:** which node limits make up the Stockfish ladder in the Elo pool. Our models will be far weaker than Stockfish's built-in skill levels, so node limits (e.g. 1, 10, 100, 1000) are the likely way to get weaker opponents. They're also reproducible, which skill levels aren't. Settle this when writing `evaluation/match_runner.py`.

## 22. Use a convolutional policy head

- **Date:** Nov 20, 2026
- **Status:** Decided. Implemented in `chesszero/model.py`.
- **Decision:** The policy head is conv3x3 → batch norm → ReLU → conv1x1 to 73 planes, flattened to 4,672 logits. There's no fully connected layer. The flattening order fixes the move index as `move_plane * 64 + square`, which `chesszero/encoding.py` documents and must follow.
- **Why:** CLAUDE.md asked for a policy head that isn't squeezed to 2 channels before a 4,672-way fully connected layer (from Cam's project [CamUTTT]). But widening that version to 32 channels makes the fully connected layer 32 × 64 × 4,672 ≈ 9.6M weights, almost four times the 2.4M-parameter tower, so the head would dominate the network's size and cost. A convolutional head outputs AlphaZero's 73 move planes on the 8×8 board directly [AlphaZero], with about 157,000 weights, and keeps every square's moves tied to that square, as Leela's convolutional heads do [Lc0]. The whole network is about 2.56M parameters at 8 blocks × 128 filters.
- **Alternatives:** AlphaGo Zero's 2-channel head with a fully connected layer [AlphaGoZero] (about 0.6M weights, but the bottleneck CLAUDE.md warns against); a 32-channel head with a fully connected layer (about 9.6M weights).

## 23. Supervised row format and how the dataset reads it

- **Date:** Nov 24, 2026
- **Status:** Decided. Implemented in `chesszero/dataset.py`, which documents the format. `pipeline/make_rows.py` must write it; `tests/test_dataset.py::game_to_rows` is a reference for one game.
- **Decision:** `make_rows.py` turns the filter's one-row-per-game output (`game_id, white_elo, black_elo, result, time_control, pgn`) into one row per position: `game_id, ply, fen, uci, z, repetition_count, white_elo, black_elo`. `fen` is the position before the move and `uci` the move played. `z` is the game result from the **side to move's** view (+1 win, 0 draw, -1 loss), so it flips sign every ply in a decisive game; the dataset passes it through unchanged. Train and validation go to separate directories of Parquet shards on the shared folder (#5, #7).
- **Reading:** a streaming `IterableDataset`, not an in-memory one. Each pass reads every row once: shards are shuffled by (seed, epoch) and dealt round-robin to DataLoader workers, and rows go through a shuffle buffer (Spark has already shuffled globally). FEN parsing, planes and the legal-move mask are computed at load time (about 200 µs per sample on one core, mostly python-chess parsing the FEN and generating legal moves), so a learner likely needs several DataLoader workers; measure in the pilot.
- **Resuming:** after a restart the learner starts a new pass with a new epoch number rather than skipping into the interrupted one. Some positions from the interrupted pass are seen again and others not at all, which only matters if the supervised phase is short relative to the data. The budget is unaffected, because it counts samples actually trained on (#1).
- **Why:** storing z from the side to move's view matches the value target (CLAUDE.md) and leaves no sign flip in the reader, where a missed flip for Black would silently train the value head on the wrong sign. At hundreds of millions of rows the data doesn't fit in memory, so it has to stream. Exact mid-pass resume with several workers and a shuffle buffer is fiddly and the gain is small.
- **Alternatives:** store the result from White's view and flip it in `dataset.py`; pre-encode planes and masks in Spark (removes the CPU cost but makes shards far larger, since a mask is 4,672 values per row; the fallback if the loader can't keep up); exact mid-pass resume by tracking each worker's position.

## 24. Branch the conditions from one supervised run

- **Date:** Nov 24, 2026
- **Status:** Open. Proposed; settle before the pilot's training runs, since it shapes `training/run_condition.py` and the learning-rate schedule (#19).
- **Question:** Train five separate runs, or one supervised run ("trunk") from 0 to B whose checkpoints at 0.25·B, 0.5·B and 0.75·B each continue as that condition's self-play phase?
- **Recommendation:** Branch. The trunk at B is the 100% condition; the 0% condition is self-play from random weights, as before.
- **Why it's equivalent to separate runs:** a separate 25% run with the same seed, data order and settings would be identical to the trunk up to 0.25·B, because nothing in it knows yet where it will stop. Branching skips recomputing that shared work rather than removing randomness the separate runs would have. This holds only if nothing in supervised training depends on p:
  - the learning-rate schedule is a function of n_eval / B, the same for every condition (#19's recommendation); a schedule that decays to the end of each run's own supervised phase would break the equivalence;
  - every condition reads the same rows in the same order (same dataset seed, #23);
  - no early stopping or other per-run choices.
- **Advantages:**
  - Supervised compute totals 1.0·B instead of 2.5·B (0 + 0.25 + 0.5 + 0.75 + 1).
  - Less noise in the comparison: conditions share the same supervised start, so differences between them come from the switch point, not from one run getting a luckier start (a paired design).
  - Elo-vs-compute curves coincide below each branch point, because the models really are the same there; the trunk's checkpoints serve every condition.
- **Disadvantages:**
  - The wall-time saving is modest: self-play dominates (CLAUDE.md, Planning the budget) and still totals 2.5·B; only the supervised term shrinks, from 2.5·B / (3·R_sup) to 1.0·B / (3·R_sup). The pilot's R_sup and R_sp give the actual hours.
  - One trunk is one draw of luck that every condition from 25% up inherits. With one run per condition we couldn't measure seed variance either way; a second trunk with another seed, if time allows, would.
  - The 75% branch can't start until the trunk reaches 0.75·B. Supervised training is fast relative to self-play, so the 0% and 25% branches would already be running.
- **Bookkeeping if adopted:** each branch starts from the trunk's full checkpoint (weights and optimizer state) and keeps its n_eval, so its count starts at p·B, not 0; checkpoint metadata records the branch point. The write-up states the branching and this equivalence argument, so readers don't take the conditions as independent replicates.
- **Alternatives:** five independent runs with the same seed (equivalent results, 1.5·B more supervised compute); five independent runs with different seeds (independent noise per condition, but with one run each it can't be separated from the effect of p).

## 25. Count self-play evaluations when the learner ingests them

- **Date:** Nov 24, 2026
- **Status:** Decided. Implemented in `chesszero/budget.py`. The value of `budget.train_samples_per_position` is open until the pilot.
- **Decision:**
  - **Counting at ingest:** self-play evaluations count toward B when the learner ingests a game file, not when a worker generates it. Each game file reports the evaluations used to produce it (positions actually sent through the network; cache hits, terminal positions and batch padding are free).
  - **Discarded work:** games that never reach the learner (a worker killed mid-batch, files arriving after the run hits B) are recorded as discarded evaluations and games, reported separately, and not counted in n_eval.
  - **Outside B:** validation passes and evaluation matches don't change the model, so they aren't counted.
  - **Phase guard:** self-play games ingested during the supervised phase raise an error rather than being counted, because phases run in order (#1).
  - **Fixed train-to-data ratio:** in self-play, the learner trains on at most `train_samples_per_position` samples per position ingested, and waits for data when it's ahead. The value is the same in every condition.
  - **Checkpoint intervals:** checkpoints fall at multiples of `budget.checkpoint_every`, the same n_eval values in every condition. B must be a multiple of 4 × `checkpoint_every`, so every switch point (0, 0.25, 0.5, 0.75, 1 × B) is a checkpoint, which branching (#24) needs.
- **Why:**
  - Only the learner keeps state, so counting there keeps the count in one process, and it counts exactly the data that affected the model. The learner's checkpoint must also record which game files it has ingested, so a restart doesn't count a file twice (exactly-once ingest); that belongs in `training/train_selfplay.py`.
  - Without a fixed ratio, the samples trained per fresh position would depend on how many fish machines happened to be free, so conditions run on different numbers of machines would learn differently for reasons unrelated to p.
- **Game file metadata:** besides the games themselves, each self-play file records:
  - the worker's host,
  - the weights version it played with,
  - evaluations used, games and plies,
  - start and end times.

  The learner logs when it ingested each file and the weights version current then. That gives the staleness distribution (how many weight versions behind the data was), ingest lag, and discarded work, which are the measurements that describe the self-play system as a distributed system (asynchronous actors and learner, bounded staleness, exactly-once ingest, backpressure).
- **Alternatives:**
  - Counting at generation, which would need workers to report counts separately and would count work that never affected the model.
  - A synchronous design where all workers play one generation with the same weights and wait at a barrier: no staleness or discarded work, but machines sit idle waiting for the slowest. (Correction, Nov 27, 2026: this entry first called this "AlphaGo Zero-style". AlphaGo Zero actually ran training, evaluation and self-play asynchronously, with an evaluator choosing the network that plays [AlphaGoZero]. The barrier-per-generation loop is the one in the alpha-zero-general reference code [AZGeneral]. See `docs/distributed-system.md`, section 7.)
  - Letting the learner train as fast as it can.
