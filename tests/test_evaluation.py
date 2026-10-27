"""Auswertung: Kennzahlen je Reiter an Handwerten und gegen die Kernroutinen; Formatierung."""

from fractions import Fraction as F

import pytest

import vhd_constants as C
import vhd_evaluation as ev
import vhd_rubinstein as R
from vhd_scenario import (
    Settings,
    holder,
    ordered_costs,
    ordered_patience,
    other,
    to_player1,
)


def test_formatting_uses_the_german_decimal_comma():
    assert ev.de(0.12345) == "0,123" and ev.de(-0.0001, 3) == "0,000" and ev.de(2.0, 1) == "2,0"
    assert ev.pct(0.5263) == "52,6 %" and ev.pct(1.0) == "100,0 %" and ev.pct(0.12345, 2) == "12,35 %"
    assert ev.pp(0.0262) == "+2,6 Prozentpunkte" and ev.pp(-0.1) == "-10,0 Prozentpunkte"
    assert ev.sci(2.17e-5) == "2,2·10⁻⁵" and ev.sci(0) == "0" and ev.sci(9.99e-3) == "1,0·10⁻²" and ev.sci(5e-24) == "5,0·10⁻²⁴"


def test_roles_follow_the_first_bidder():
    s1, s2 = Settings(d1=0.8, d2=0.9, first=1, c1=0.1, c2=0.2), Settings(d1=0.8, d2=0.9, first=2, c1=0.1, c2=0.2)
    assert ordered_patience(s1) == (0.8, 0.9) and ordered_patience(s2) == (0.9, 0.8) and ordered_costs(s1) == (0.1, 0.2) and ordered_costs(s2) == (0.2, 0.1)
    assert to_player1(1, 0.7) == 0.7 and to_player1(2, 0.7) == pytest.approx(0.3)
    assert (holder(1), holder(2), other(1), other(2)) == (2, 1, 2, 1)


def test_analyse_on_the_default():
    a = ev.analyse(Settings())
    assert a.x1 == pytest.approx(10 / 19, abs=1e-8) and a.x2 == pytest.approx(9 / 19, abs=1e-8) and a.limit_x1 == pytest.approx(10 / 19)
    assert a.other_order_x1 == pytest.approx(9 / 19, abs=1e-8) and a.other_order_limit_x1 == pytest.approx(9 / 19)
    assert a.values[0] == a.share_first and len(a.values) == ev.ROUNDS_SHOWN and a.error < 1e-8


def test_analyse_with_few_rounds_shows_only_those():
    a = ev.analyse(Settings(t=3, d1=0.5, d2=0.5))
    assert a.values == (0.75, 0.5, 1.0) and a.x1 == 0.75 and a.error == pytest.approx(abs(0.75 - 2 / 3))


def test_patience_curve_and_slope():
    curve = ev.patience_curve(0.9)
    assert len(curve["d1"]) == 100 and curve["d1"][0] == 0.0 and curve["d1"][-1] == 0.99
    assert curve["verlader_first"][0] == pytest.approx(0.1) and curve["verlader_first"][90] == pytest.approx(10 / 19)
    assert curve["spediteur_first"][90] == pytest.approx(1 - 10 / 19)
    for d1 in (0.0, 0.5, 0.9, 0.99):
        h = 1e-6
        numeric = (R.rubinstein_share(d1 + h, 0.9) - R.rubinstein_share(d1 - h, 0.9)) / (2 * h)
        assert ev.patience_slope(d1, 0.9) == pytest.approx(numeric, rel=1e-5)
    assert [round(ev.patience_slope(d, 0.9), 2) for d in (0.5, 0.9, 0.99)] == [0.30, 2.49, 7.58]


def test_patience_step_goes_down_at_the_upper_limit():
    d_old, x_old, d_new, x_new = ev.patience_step(Settings(d1=0.9))
    assert (d_old, d_new) == (0.9, 0.91) and x_new > x_old
    d_old, x_old, d_new, x_new = ev.patience_step(Settings(d1=C.D_MAX))
    assert d_new == pytest.approx(C.D_MAX - 0.01) and x_new < x_old


def test_patience_rows_at_the_hook_values():
    rows = {r["d1"]: r for r in ev.patience_rows(0.9)}
    assert [round(rows[d]["first"], 4) for d in (0.0, 0.5, 0.8, 0.9, 0.99)] == [0.1, 0.1818, 0.3571, 0.5263, 0.9174]


def test_horizon_curve_matches_the_single_runs():
    s = Settings(d1=0.8, d2=0.95, first=2)
    curve = ev.horizon_curve(s, 12)
    assert curve["x1"] == pytest.approx([R.share_of_player1(0.8, 0.95, T, first=2) for T in range(1, 13)])
    assert curve["limit"] == pytest.approx(1 - R.rubinstein_share(0.95, 0.8))


def test_settle_row_on_the_first_pair():
    row = ev.settle_row(0.9, 0.9)
    assert row["settled"] == 126 and round(row["predicted"]) == 131 and row["limit"] == pytest.approx(10 / 19)
    assert row["x10"] == pytest.approx(0.3428008, abs=1e-6) and row["x11"] == pytest.approx(0.6914793, abs=1e-6) and row["err500"] < 1e-20


def test_error_curves_floor_and_shape():
    c = ev.error_curves(((0.9, 0.9),), 300)
    errs = c["series"]["(0,90; 0,90)"]
    assert len(errs) == 300 == len(c["T"]) and min(errs) >= 1e-12 and errs[0] == pytest.approx(1 - 10 / 19)


def test_short_rounds_rows_against_the_hand_values():
    rows = {r["dt"]: r for r in ev.short_rounds_rows(1.0, 2.0)}
    assert rows[1.0]["x_first"] == pytest.approx(0.909969, abs=1e-6) and rows[1.0]["x_second"] == pytest.approx(0.334759, abs=1e-6)
    assert rows[1.0]["limit"] == pytest.approx(2 / 3) and rows[1.0]["edge"] == pytest.approx(rows[1.0]["x_first"] - rows[1.0]["x_second"])
    assert ev.short_rounds_row(Settings(dt=1 / 24))["x_first"] == pytest.approx(0.6805, abs=1e-4)


def test_costs_curve_and_case_row():
    c = ev.costs_curve(Settings(c1=0.05, c2=0.10), 5)
    assert c["costs"] == pytest.approx([1.0, 0.10, 1.0, 0.15, 1.0]) and c["discount"] is not None and c["discount"][0] == 1.0
    row = ev.costs_case_row(0.05, 0.10)
    assert (row["pred_first"], row["t1000_first"], row["t1001_first"]) == (1, 1, 1) and row["discount"] == pytest.approx(0.6897, abs=1e-4)
    assert ev.costs_case_row(0.2, 0.2)["pred_first"] is None


def test_costs_curve_with_the_spediteur_first_converts_to_the_verlader():
    c = ev.costs_curve(Settings(c1=0.05, c2=0.10, first=2), 4)
    assert c["costs"] == pytest.approx([1 - R.costs_backward(0.10, 0.05, T)[0] for T in range(1, 5)])


def test_outside_state_and_curve():
    below = ev.outside_state(Settings(o=0.4))
    above = ev.outside_state(Settings(o=0.6))
    assert not below["counts"] and above["counts"] and above["with"] == pytest.approx(0.6) and above["without"] == pytest.approx(9 / 19, abs=1e-8)
    assert below["with"] == pytest.approx(below["without"]) and above["threshold"] == pytest.approx(0.9 * 0.1 / 0.19) and above["holder"] == 2
    c = ev.outside_curve(Settings())
    assert c["A"][60] == pytest.approx(0.6) and c["nash"][60] == pytest.approx(0.8) and c["C"][60] == pytest.approx(max(0.9 * 0.6, 9 / 19))
    assert ev.outside_state(Settings(first=2, o=0.6))["holder"] == 1


def test_outside_state_in_variant_c():
    st = ev.outside_state(Settings(o=0.6, variant="C"))
    assert st["payoff"] == pytest.approx(0.54) and st["counts"] and st["with"] == pytest.approx(0.54)


def test_outside_scan_and_check_on_a_small_grid():
    assert ev.outside_scan([(0.9, 0.9)], (0.2, 0.6), range(1, 11)) == {"cells": 20, "diff": 0}
    assert max(ev.outside_check([(0.9, 0.9)], (0.2, 0.6)).values()) < 1e-9


def test_nash_vs_exit_rows_at_the_hook_values():
    rows = {r["o"]: r for r in ev.nash_vs_exit_rows()}
    assert rows[0.2]["nash"] == pytest.approx(0.6) and rows[0.2]["exit"] == pytest.approx(0.5, abs=1e-4) and rows[0.6]["exit"] == pytest.approx(0.6, abs=1e-4)


def test_solutions_and_scale_rows():
    sol = ev.solutions_for(Settings(curv=0.5, scale=1.0, threat=0.0))
    assert sol["nash"] == pytest.approx(2 / 3) and sol["half"] == 0.5
    rows = ev.scale_rows(0.5, 0.1)
    assert [r["k"] for r in rows] == [1.0, 10.0, 0.1] and rows[0]["nash"] == pytest.approx(rows[1]["nash"]) and rows[0]["egalitarian"] != pytest.approx(rows[1]["egalitarian"], abs=0.1)


def test_curved_rubinstein_row_and_short_round_gap():
    row = ev.curved_rubinstein_row(0.9, 0.9, 0.5)
    assert row["rubinstein"] == pytest.approx(0.19 / 0.271) and row["nash"] == pytest.approx(2 / 3) and row["effective_d2"] == pytest.approx(0.81)
    assert ev.short_round_nash_gap(1.0, 0.001) == pytest.approx(0.00025, rel=1e-3)


def test_grid_state_and_curve():
    s = Settings(g=10, d1=0.9, d2=0.9, t=200)
    st = ev.grid_state(s)
    assert st["share_first"] == pytest.approx(0.1) and st["points"] == 9 and (st["points_lo"], st["points_hi"]) == (pytest.approx(0.1), pytest.approx(0.9))
    assert st["error_times_g"] == pytest.approx((0.1 - 10 / 19) * 10) and st["ratio"] <= 1.0 and st["bound"] == pytest.approx(1 / (0.19 * 10))
    curve = ev.grid_curve(s, 6)
    assert curve["grid"][:2] == pytest.approx([1.0, 0.1]) and curve["continuous"][:2] == pytest.approx([1.0, 0.1]) and curve["limit"] == pytest.approx(10 / 19)


def test_grid_state_other_rule_and_first_bidder():
    st = ev.grid_state(Settings(g=10, t=1, rule="R"))
    assert st["share_first"] == pytest.approx(0.9) and st["other_rule_share"] == 1.0


def test_bound_cells_fixed_row_and_rule_gap():
    b = ev.bound_cells([(0.9, 0.9)], (10, 100), (600, 601))
    assert b["cells"] == 8 and b["ok"] == 8 and b["exceptions"] == []
    f = ev.fixed_row(0.9, 0.9, 20, False)
    assert (f["count"], f["lo"], f["hi"], f["rule"]) == (1, pytest.approx(0.5), pytest.approx(0.5), "R")
    assert ev.rule_gap(0.9, 0.9, 10, range(1, 20)) == pytest.approx(0.1)


def test_learn_once_and_cell():
    run = ev.learn_once("ultimatum", "hedge", 2.0, 3000, 1)
    assert run["spe"] == 1.0 and run["pure_nash"] == 13 and run["result"].gap <= ev.NASH_GAP_LIMIT
    cell = ev.learn_cell("zwei_runden", "rm", 0.0, seeds=(1, 2), rounds=1000)
    assert cell["spe"] == pytest.approx(0.1) and len(cell["shares"]) == len(cell["gaps"]) == 2 and cell["rounds"] == 1000
    assert len(ev.learn_cells()) == 8 and ("zwei_runden", "rm", 0.0) in ev.learn_cells()


def test_find_matches_floats_with_tolerance():
    cells = [{"a": 0.1, "b": "x"}, {"a": 0.2, "b": "y"}]
    assert ev.find(cells, a=0.2)["b"] == "y" and ev.find(cells, a=0.3) is None and ev.find(cells, b="x")["a"] == 0.1


def test_load_grid_returns_none_for_a_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "GRID_PATH", tmp_path / "gibt-es-nicht.json")
    assert ev.load_grid() is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    monkeypatch.setattr(ev, "GRID_PATH", bad)
    assert ev.load_grid() is None


def test_closed_form_of_the_distance_to_the_formula():
    """Abstand zur Formel in geschlossener Form (📐-Block): T = 2m: −(1 − d₂) q^m/(1 − q); T = 2m + 1: +d₂ (1 − d₁) q^m/(1 − q), q = d₁d₂."""
    for d1, d2 in ((F(9, 10), F(9, 10)), (F(1, 2), F(4, 5)), (F(19, 20), F(3, 10))):
        q = d1 * d2
        limit = R.rubinstein_share(d1, d2)
        for T in range(1, 31):
            m = T // 2
            expected = d2 * (1 - d1) * q ** m / (1 - q) if T % 2 else -(1 - d2) * q ** m / (1 - q)
            assert R.backward_T(d1, d2, T)[0] - limit == expected, (d1, d2, T)


def test_closed_form_of_the_distance_for_random_pairs():
    """Dieselbe geschlossene Form für 60 Zufallspaare (Brüche, Geduld 0 bis 0,98) und T = 1 bis 14: ungerade Horizonte nicht unter, gerade nicht über dem Grenzwert."""
    from vhd_rng import SplitMix64
    rng = SplitMix64(17)
    for _ in range(60):
        d1, d2 = F(rng.below(99), 100), F(rng.below(99), 100)
        q, limit = d1 * d2, R.rubinstein_share(d1, d2)
        for T in range(1, 15):
            diff = R.backward_T(d1, d2, T)[0] - limit
            m = T // 2
            assert diff == (d2 * (1 - d1) * q ** m / (1 - q) if T % 2 else -(1 - d2) * q ** m / (1 - q))
            assert diff >= 0 if T % 2 else diff <= 0
