"""Die vorgerechneten Messreihen (data/negotiation_grid.json): Aufbau, Zählungen und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import vhd_constants as C
import vhd_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/negotiation_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe"):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-6, abs=1e-12), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Rechnung oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_grid_has_every_cell_exactly_once(grid):
    assert [(r["d1"], r["d2"]) for r in grid["settle"]] == list(C.E1_PAIRS)
    assert sorted(grid["rounds"]) == sorted(f"{r1}|{r2}" for r1, r2 in C.E3_RATES) and all([r["dt"] for r in rows] == list(C.E3_DTS) for rows in grid["rounds"].values())
    assert [(r["c_first"], r["c_second"]) for r in grid["costs"]] == list(C.E4_CASES)
    assert sorted((r["a"], r["k"]) for r in grid["scale"]) == sorted((a, k) for a in C.E6_AS for k in C.E6_KS)
    assert sorted((r["game"], r["rule"], r["eta"]) for r in grid["learn"]) == sorted(ev.learn_cells())
    assert len(grid["fixed"]) == len(C.E2_PAIRS3) * 4 * 2 and len(grid["rule_gap"]) == 12 and len(grid["curved"]) == 6 and len(grid["nash_gap_short"]) == 9


def test_grid_meta(grid):
    assert grid["meta"]["learn_seeds"] == list(C.GRID_LEARN_SEEDS) and grid["meta"]["learn_rounds"] == C.GRID_LEARN_ROUNDS
    assert all(c["seeds"] == list(C.GRID_LEARN_SEEDS) and c["rounds"] == C.GRID_LEARN_ROUNDS and len(c["shares"]) == 4 for c in grid["learn"])


def test_the_cell_counts_of_the_bound_measurement(grid):
    assert grid["bound3"]["cells"] == len(C.E2_PAIRS3) * len(C.E2_GS) * 2 * len(C.E2_TS) == 60
    assert grid["bound_more_a"]["cells"] == len(C.E2_PAIRS_MORE) * 2 * 4 * 2 == 80 and grid["bound_more_b"]["cells"] == len(C.E2_PAIRS_MORE) * 3 * 2 * 2 == 60
    total = sum(grid[k]["cells"] for k in ("bound3", "bound_more_a", "bound_more_b"))
    ok = sum(grid[k]["ok"] for k in ("bound3", "bound_more_a", "bound_more_b"))
    assert (total, ok) == (200, 196)


def test_the_four_exceptions_are_unconverged_horizons(grid):
    exceptions = grid["bound_more_a"]["exceptions"]
    assert len(exceptions) == 4 and {(e[0], e[1], e[2], e[4]) for e in exceptions} == {(0.99, 0.99, 10000, 400), (0.99, 0.99, 10000, 401)}
    # T = 400 ist bei δ = 0,99 noch nicht konvergiert: der Fehler schon im stetigen Spiel liegt über der Schranke-Größenordnung (1/((1 − δ²)·G) = 0,5·10⁻³)
    shares = ev.R.share_at_horizons(0.99, 0.99, 401)
    limit = ev.R.rubinstein_share(0.99, 0.99)
    assert abs(shares[399] - limit) > 5e-3 and abs(shares[400] - limit) > 5e-3
    assert grid["bound3"]["exceptions"] == [] and grid["bound_more_b"]["exceptions"] == []


def test_the_headline_numbers_of_the_experiments(grid):
    settle = {(r["d1"], r["d2"]): r for r in grid["settle"]}
    assert [settle[p]["settled"] for p in C.E1_PAIRS] == [126, 1307, 99, 36] and [round(settle[p]["predicted"]) for p in C.E1_PAIRS] == [131, 1375, 101, 35]
    assert grid["outside_scan"] == {"cells": 2400, "diff": 0} and max(grid["outside_check"].values()) < 3e-5
    fixed = {(r["d1"], r["d2"], r["g"], r["rule"]): r for r in grid["fixed"]}
    assert [fixed[(0.9, 0.9, g, "A")]["count"] for g in (10, 20, 50, 100)] == [9, 2, 7, 6] and [fixed[(0.9, 0.9, g, "R")]["count"] for g in (10, 20, 50, 100)] == [9, 1, 9, 4]
    assert all(abs(r["gap"] * r["g"] - 1) < 1e-9 for r in grid["rule_gap"])


def test_learning_cells_never_hit_the_subgame_perfect_share_and_stay_near_nash(grid):
    for c in grid["learn"]:
        assert all(abs(x - c["spe"]) > 1e-9 for x in c["shares"]), c
        assert max(c["gaps"]) <= ev.NASH_GAP_LIMIT, c
