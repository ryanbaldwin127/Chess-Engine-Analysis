# Decisions

A running record of project decisions and the reasoning behind them, for the final
write-up. Add an entry whenever a decision affects the experiment, the data, or how
results should be read. Don't rewrite old entries: if a decision changes, add a new
entry and mark the old one as superseded.

Each entry has a date, a status (decided, open or superseded), the decision, why we
made it, and the alternatives we considered.

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
- **Status:** Decided. The 30-second cutoff is tentative. Implemented in `pipeline/make_rows.py` (owner: Munisa).
- **Decision:** When expanding games into training rows, skip any position where the player to move had less than 30 seconds left. The clock comes from the `[%clk h:mm:ss]` comment Lichess records after each move. The rest of the game is still used, and the value target for kept positions is still the final result.
- **Why:** Most of the quality loss in faster games happens in time scrambles, where players make rushed moves and blunders. Dropping those positions removes the worst moves from every time control, including 10-minute games, without throwing away whole games. It also lets us keep 5-minute games if we need the volume (see #11).
- **Alternatives:** Dropping whole fast games (loses the good early and middle-game moves too); no clock filter (keeps time-scramble blunders as policy targets).
- **Notes:**
  - Record how many positions this removes, for the write-up.
  - The value target is unchanged, so a game decided by a time-scramble blunder still labels its earlier positions with that result. That's a known source of noise in z that this rule doesn't fix.
  - Maia (McIlroy-Young et al., 2020), a model trained to predict human moves on Lichess games, used a similar low-clock cutoff. Check the paper's exact rule before citing it.

## 11. Which time controls go into the training rows

- **Date:** Nov 19, 2026
- **Status:** Open until the pilot sets the budget B.
- **Question:** Should the supervised training rows include 5-minute games (300+0, 300+3), or only games with a base of 10 minutes or more?
- **Considerations:**
  - **For leaving 5-minute games out:** players have less time to find good moves, more games are decided by late blunders, and the value target z is noisier. Games of 10 minutes or more still give about 10M games, or roughly 650M positions at a median of about 60 plies.
  - **For including them:** they double the number of games (to about 20M). At 1200+ ratings they're still far stronger than our models are likely to become, and the supervised phase only needs to give a sensible starting policy. #10 already removes the worst time-scramble moves.
  - **Fairness:** every condition with human data uses the same rows, so this choice affects how much human data helps overall, not whether the conditions are comparable. Mention it in the write-up either way.
- **Plan:** Default to games of 10+ minutes. After the pilot, compare the number of supervised samples the 100% condition will use (B / 3) with the positions available. Add the 5-minute games only if the 10+ minute rows would have to be repeated many times. Either way, this is a column filter in `make_rows.py`, so the month doesn't need re-filtering.
