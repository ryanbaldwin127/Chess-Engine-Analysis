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
│   ├── dataset.py          # PyTorch Dataset reading Parquet shards from HDFS
│   ├── budget.py           # network-evaluation counter
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
│   └── plots.py            # Elo vs compute, training curves, scaling curves
│
└── tests/
    ├── test_encoding.py    # round-trip every legal move
    ├── test_mcts.py        # finds mate-in-1/2 with the stand-in network
    └── test_game.py