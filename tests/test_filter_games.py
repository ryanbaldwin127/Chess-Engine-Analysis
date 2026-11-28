"""Tests for pipeline/filter_games.py's per-game rules (no Spark needed).

Run from the repo root: python -m pytest tests/test_filter_games.py -v
"""

import zstandard

from pipeline.filter_games import check_rules, parse_headers, split_games, to_row

MIN_BASE, MIN_RATING = 300, 1000


def make_game(site="https://lichess.org/abcd1234", white_elo="1500", black_elo="1600",
              time_control="600+5", termination="Normal", result="1-0", white_title=None):
    headers = [
        '[Event "Rated Rapid game"]',
        f'[Site "{site}"]',
        f'[Result "{result}"]',
        f'[WhiteElo "{white_elo}"]',
        f'[BlackElo "{black_elo}"]',
        f'[TimeControl "{time_control}"]',
        f'[Termination "{termination}"]',
    ]
    if white_title:
        headers.append(f'[WhiteTitle "{white_title}"]')
    return "\n".join(headers) + f"\n\n1. e4 {{ [%clk 0:10:00] }} 1... e5 {{ [%clk 0:10:00] }} {result}\n"


def rule(game):
    return check_rules(parse_headers(game), MIN_BASE, MIN_RATING)


def test_kept_game():
    assert rule(make_game()) == "kept"


def test_bot():
    assert rule(make_game(white_title="BOT")) == "bot"


def test_termination_must_be_normal():
    for termination in ("Time forfeit", "Abandoned", "Rules infraction", "Unterminated"):
        assert rule(make_game(termination=termination)) == "not_normal_termination"


def test_time_control():
    assert rule(make_game(time_control="300+0")) == "kept"        # 5-minute games are kept (#9)
    assert rule(make_game(time_control="300+3")) == "kept"
    assert rule(make_game(time_control="180+2")) == "short_time"
    assert rule(make_game(time_control="-")) == "short_time"      # correspondence


def test_rating():
    assert rule(make_game(white_elo="1000", black_elo="1000")) == "kept"
    assert rule(make_game(black_elo="999")) == "low_rating"
    assert rule(make_game(white_elo="?")) == "low_rating"


def test_rules_apply_in_order():
    """A game failing several rules is counted under the first one."""
    assert rule(make_game(white_title="BOT", termination="Abandoned", time_control="60+0")) == "bot"
    assert rule(make_game(termination="Abandoned", time_control="60+0")) == "not_normal_termination"


def test_bad_header():
    assert check_rules(parse_headers('[Event "x"\n\n1. e4 *\n'), MIN_BASE, MIN_RATING) == "bad_header"


def test_to_row():
    game = make_game()
    assert to_row(game, parse_headers(game)) == ("abcd1234", 1500, 1600, "1-0", "600+5", game)


def test_split_games_round_trip():
    games = [make_game(site=f"https://lichess.org/g{i}") for i in range(3)]
    chunk = zstandard.ZstdCompressor().compress("\n".join(games).encode("utf-8"))
    split = split_games(chunk)
    assert len(split) == 3
    assert [to_row(g, parse_headers(g))[0] for g in split] == ["g0", "g1", "g2"]
