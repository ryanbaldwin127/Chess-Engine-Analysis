chess-zero/
├── README.md               # setup, how to run each stage, file-format notes
├── requirements.txt
├── .gitignore              # data/, checkpoints/, *.pgn, *.parquet — never commit these
├── configs/
│   ├── base.yaml           # network size, MCTS sims, budget, HDFS paths
│   └── condition_25.yaml   # one small file per condition (0/25/50/75/100)
│
├── chesszero/              # shared library — everyone imports this
│   ├── encoding.py         # board → planes, move ↔ index, legal-move mask
│   ├── model.py            # ResNet with policy + value heads
│   ├── mcts.py             # search (PUCT, Dirichlet noise, temperature, batching)
│   ├── players.py          # Player interface + Random, Stockfish, MCTS players
│   ├── game.py             # play two players, draw rules, move cap, save PGN
│   ├── dataset.py          # PyTorch Dataset streaming Parquet shards from the shared folder
│   ├── budget.py           # network-evaluation counter, phase switch, checkpoint timing
│   └── storage.py          # HDFS helpers: atomic write/rename, save/load checkpoints
│
├── pipeline/               # Spark jobs
│   ├── chunk_pgn.py        # split a month's PGN into chunks on HDFS
│   ├── filter_games.py     # header filters + counts after each filter
│   ├── make_rows.py        # games → rows, split by game ID, shuffle, write Parquet
│   └── opening_book.py     # (position, move) counts for the first 10–15 plies
│
├── training/
│   ├── train_supervised.py
│   ├── selfplay_worker.py
│   ├── train_selfplay.py
│   ├── run_condition.py    # runs one condition: supervised → switch → self-play → stop at budget
│   └── launch_workers.sh   # start workers on each machine with CUDA_VISIBLE_DEVICES
│
├── evaluation/
│   ├── sanity_checks.py
│   ├── match_runner.py     # round-robin games + Elo
│   ├── move_analysis.py    # Stockfish scoring, best/worst tables (Spark)
│   ├── mate_puzzles.py
│   └── opening_analysis.py
│
├── experiments/
│   ├── pilot.py            # measure throughput, set the budget
│   └── scaling_test.py     # 1, 2, 5, 10 machines
│
├── analysis/
│   ├── plots.py            # Elo vs compute, training curves, scaling curves
│   ├── data_stats.py       # summarize the human-game data into results/data_stats_<month>.json
│   └── pipeline_figures.ipynb  # data-pipeline figures and counts table -> figures/pipeline/
│
├── results/
│   └── data_stats_2026-10.json  # small summary the pipeline figures read
│
├── figures/                # figures for the write-up, one folder per section
│   └── pipeline/           # PDF + PNG, and pipeline_counts.tex
│
└── tests/
    ├── test_encoding.py    # input planes; round-trip every legal move; mask support
    ├── test_dataset.py     # every row once per pass; rows encode like the live game
    ├── test_budget.py      # phases in order, aligned checkpoints, exact resume and branching
    ├── test_mcts.py        # finds mate-in-1/2 with the stand-in network
    └── test_game.py