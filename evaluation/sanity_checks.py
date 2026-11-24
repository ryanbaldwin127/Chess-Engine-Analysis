"""Sanity checks: random vs random games all end, Stockfish ladder ranks correctly

Run from repo root on dept machine:
    python -m evaluation.sanity_checks
    python -m evaluation.sanity_checks --random-games 200 --openings 5   # quick run 
"""

import argparse
import collections
import json
import os
import random
import sys
import time
import chess

from chesszero.game import play_game
from chesszero.players import RandomPlayer, StockfishPlayer

SHARED_RESULTS = "/s/bach/a/class/cs435/cs435a/chess-zero/results"
LADDER_NODES = [1, 100, 1000, 10000]
WHITE_SCORE = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def random_games(n, seed=0):
    terminations = collections.Counter()
    results = collections.Counter()
    plies = []
    
    for i in range(n):
        r = play_game(RandomPlayer(seed=seed + 2 * i), RandomPlayer(seed=seed + 2 * i + 1))
        terminations[r.termination] += 1
        results[r.result] += 1
        plies.append(r.plies)

    draws = results["1/2-1/2"]
    
    return {
        "games": n,
        "results": dict(results),
        "terminations" : dict(terminations),
        "draw_share": draws / n,
        "avg_plies": sum(plies) / n,
        "max_plies": max(plies),
        "passed": draws / n > 0.5,
    }
    

def make_openings(n, plies=4, seed=0):
    rng = random.Random(seed)
    fens, seen = [], set()
    
    while len(fens) < n:
        board = chess.Board()

        for _ in range(plies):
            board.push(rng.choice(list(board.legal_moves)))
        
        fen = board.fen()
        
        if not board.is_game_over() and fen not in seen:
            seen.add(fen)
            fens.append(fen)
    return fens


def head_to_head(a, b, openings):
    wins = draws = losses = 0
    
    for fen in openings:
        for a_is_white in (True,False):
            white, black = (a,b) if a_is_white else (b, a)
            r = play_game(white, black, start_fen=fen)
            score = WHITE_SCORE[r.result] if a_is_white else 1 - WHITE_SCORE[r.result]

            if score == 1:
                wins += 1
            elif score == 0.5:
                draws += 1
            else:
                losses += 1

    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games
    
    return {
        "player": a.name, "opponent": b.name,
        "wins": wins, "draws": draws, "losses": losses,
        "score": score,
        "passed": score > 0.5,
    }
    

def stockfish_ladder(openings, nodes_list=LADDER_NODES, seed=0):
    players = [RandomPlayer(seed=seed)]
    try: 
        for nodes in nodes_list:
            players.append(StockfishPlayer(nodes=nodes))
        return [head_to_head(stronger, weaker, openings)
                for weaker, stronger in zip(players, players[1:])]
    
    finally:
        for p in players:
            p.close()


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for game runner and Stockfish")
    
    parser.add_argument("--random-games", type=int, default=2000)
    parser.add_argument("--openings", type=int, default=20,
                        help="openings per ladder pairing (each played twice)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(SHARED_RESULTS, "sanity_checks.json"))
    
    args = parser.parse_args()
    start = time.time()

    print(f"Random vs random ({args.random_games} games):")
    rr = random_games(args.random_games, args.seed)
    print(f" results: {rr['results']}")
    print(f" terminations: {rr['terminations']}")
    print(f" draw share {rr['draw_share']:.1%}, avg plies {rr['avg_plies']:.0f}, "
          f"max plies {rr['max_plies']} -> {'PASS' if rr['passed'] else 'FAIL'}")

    print(f"\nStockfish ladder ({args.openings} openings x 2 colors per pairings):")

    ladder = stockfish_ladder(make_openings(args.openings, seed=args.seed), seed=args.seed)

    for row in ladder:
        print(f" {row['player']:>18} vs {row['opponent']:<18} "
              f"W{row['wins']:>3} D{row['draws']:>3} L{row['losses']:>3} "
              f"score {row['score']:.2f} -> {'PASS' if row['passed'] else 'FAIL'}")
        
    all_passed = rr["passed"] and all(row["passed"] for row in ladder)

    summary = {
        "settings": vars(args),
        "random_vs_random": rr,
        "stockfish_ladder": ladder,
        "all_passed": all_passed,
        "seconds": round(time.time() - start, 1),
    }
    save_json(summary, args.out)
    print(f"\n{'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'} "
          f"({summary['seconds']}s). Saved to {args.out}")
    
    return 0 if all_passed else 1


if __name__ =="__main__":
    sys.exit(main())