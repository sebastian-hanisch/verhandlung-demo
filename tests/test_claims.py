"""Jede Zahl der README steht hier und wird aus den vorgerechneten Messreihen (data/negotiation_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4, §11a).

Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`, `_p`) gegen die README geprüft; die Zeilen-Bauer (`patience_table` usw.) und `numbers()` dienen auch dazu, die README-Tabellen und
-Zahlen zu erzeugen (tools/build_readme.py liest sie). Alles bis auf die Lernläufe ist deterministisch; die Lernläufe stehen nur als Bänder über vier Zufalls-Seeds. Jede Zahl nennt ihre Bedingung (Geduld,
Horizont, Raster, Gleichstandsregel, Variante, Lernrate) in der README direkt daneben; die Tests unten prüfen die Zahl unter genau dieser Bedingung."""

from fractions import Fraction as F
from pathlib import Path

import pytest

import vhd_constants as C
import vhd_evaluation as ev
import vhd_outside as O
import vhd_raster as Ra
import vhd_rubinstein as R
import vhd_solutions as S
from vhd_scenario import Settings

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _p(v, d=1):
    return f"{100 * v:.{d}f}".replace(".", ",") + " %"


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def patience_table() -> list[str]:
    return [f"| {_c(r['d1'], 2)} | {_p(r['first'], 1)} | {_p(r['second'], 1)} | 50,0 % | {_c(r['slope'], 2)} |" for r in ev.patience_rows(0.9)]


def settle_table(grid) -> list[str]:
    return [f"| {_c(r['d1'], 2)}; {_c(r['d2'], 2)} | {_p(r['limit'], 2)} | {_p(r['x10'], 2)} | {_p(r['x11'], 2)} | {r['settled']} | {round(r['predicted'])} |" for r in grid["settle"]]


def bound_table(grid) -> list[str]:
    rows = []
    for name, key in (("3 Paare, G = 10 bis 10 000, beide Regeln, T = 600 und 601", "bound3"), ("5 weitere Paare, G = 1 000 und 10 000, T = 400, 401, 2 000, 2 001", "bound_more_a"), ("5 weitere Paare, G = 10, 50, 100, T = 400, 401", "bound_more_b")):
        b = grid[key]
        rows.append(f"| {name} | {b['cells']} | {b['ok']} | {_c(b['worst'], 2)} |")
    return rows


def fixed_table(grid) -> list[str]:
    cells = {(r["g"], r["rule"]): r for r in grid["fixed"] if (r["d1"], r["d2"]) == (0.9, 0.9)}
    return [f"| {g} | {cells[(g, 'A')]['count']} ({_p(cells[(g, 'A')]['lo'], 0)} bis {_p(cells[(g, 'A')]['hi'], 0)}) | {cells[(g, 'R')]['count']} ({_p(cells[(g, 'R')]['lo'], 0)} bis {_p(cells[(g, 'R')]['hi'], 0)}) |"
            for g in (10, 20, 50, 100)]


def rounds_table(grid) -> list[str]:
    rows = []
    for key, rs in grid["rounds"].items():
        r1, r2 = (float(x) for x in key.split("|"))
        for r in rs:
            if r["dt"] in (0.1, 0.01, 0.001, 0.0001):
                rows.append(f"| {_c(r1, 1)}; {_c(r2, 1)} | {_c(r['dt'], 4)} | {_p(r['x_first'], 3)} | {_p(r['limit'], 3)} | {_c(r['rate'], 6)} | {_c(r['rate_hand'], 6)} |")
    return rows


def costs_table(grid) -> list[str]:
    def first(r):
        return "nicht eindeutig" if r["pred_first"] is None else _p(r["pred_first"], 1)
    return [f"| {_c(r['c_first'], 2)}; {_c(r['c_second'], 2)} | {first(r)} | {_p(r['t1000_first'], 1)} | {_p(r['t1001_first'], 1)} | {'–' if r['discount'] is None else _p(r['discount'], 1)} |" for r in grid["costs"]]


def outside_table() -> list[str]:
    rows = []
    for o in (0.4, 0.5, 0.6, 0.8, 0.95):
        a = 1 - O.outside_backward(0.9, 0.9, o, 1001, "A")[0]
        c = 1 - O.outside_backward(0.9, 0.9, o, 1001, "C")[0]
        rows.append(f"| {_c(o, 2)} | {_p(a, 1)} | {_p(c, 1)} |")
    return rows


def nash_exit_table(grid) -> list[str]:
    return [f"| {_c(r['o'], 2)} | {_p(r['nash'], 1)} | {_p(r['exit'], 1)} |" for r in grid["nash_vs_exit"]]


def scale_table(grid) -> list[str]:
    rows = []
    for a in C.E6_AS:
        for r in sorted((r for r in grid["scale"] if abs(r["a"] - a) < 1e-9), key=lambda r: r["k"]):
            rows.append(f"| {_c(a, 2)} | {_c(r['k'], 1)} | {_p(r['nash'], 1)} | {_p(r['ks'], 1)} | {_p(r['egalitarian'], 1)} | {_p(r['utilitarian'], 1)} |")
    return rows


def curved_table(grid) -> list[str]:
    return [f"| {_c(r['a'], 2)} | {_c(r['d1'], 2)} | {_c(r['effective_d2'], 4)} | {_p(r['rubinstein'], 2)} | {_p(r['nash'], 2)} |" for r in grid["curved"]]


def nash_gap_table(grid) -> list[str]:
    return [f"| {_c(r['a'], 2)} | {_c(r['dt'], 3)} | {_c(r['gap'], 5)} |" for r in grid["nash_gap_short"]]


def learn_table(grid) -> list[str]:
    return [f"| {C.GAMES[c['game']]} | {C.LEARN_RULES[c['rule']]} | {_c(c['eta'], 1) if c['rule'] == 'hedge' else '–'} | {_p(min(c['shares']), 1)} bis {_p(max(c['shares']), 1)} | {_c(max(c['gaps']), 3)} | {_p(c['spe'], 0)} |" for c in grid["learn"]]


def numbers(grid) -> dict:
    """Die Zahlen im Fließtext der README (Name -> Zeichenkette in der Rundung der App)."""
    d = Settings()
    a = ev.analyse(d)
    hook = Settings(d1=0.91)
    ah = ev.analyse(hook)
    settle = {(r["d1"], r["d2"]): r for r in grid["settle"]}
    fixed = {(r["g"], r["rule"]): r for r in grid["fixed"] if (r["d1"], r["d2"]) == (0.9, 0.9)}
    total = sum(grid[k]["cells"] for k in ("bound3", "bound_more_a", "bound_more_b"))
    ok = sum(grid[k]["ok"] for k in ("bound3", "bound_more_a", "bound_more_b"))
    rounds = {k: {r["dt"]: r for r in v} for k, v in grid["rounds"].items()}
    costs = {(r["c_first"], r["c_second"]): r for r in grid["costs"]}
    learn = {(c["game"], c["rule"], c["eta"]): c for c in grid["learn"]}
    hour = ev.short_rounds_row(Settings(dt=1 / 24))
    minute = ev.short_rounds_row(Settings(dt=1 / 1440))
    day = ev.short_rounds_row(Settings(dt=1.0))
    risk = ev.solutions_for(Settings(curv=0.5))
    lin = ev.solutions_for(Settings())
    thr_a = O.outside_threshold(0.9, 0.9, "A")
    thr_c = O.outside_threshold(0.9, 0.9, "C")
    sol_by_k = {r["k"]: r for r in ev.scale_rows(0.5, 0.1)}
    slopes = {d1: ev.patience_slope(d1, 0.9) for d1 in (0.5, 0.9, 0.99)}
    lshares = [x for c in grid["learn"] for x in c["shares"]]
    return {
        "x1_default": _p(a.x1, 1), "x2_default": _p(a.x2, 1), "x1_hook": _p(ah.x1, 1), "hook_pp": _c(100 * (ah.x1 - a.x1), 2),
        "slope_05": _c(slopes[0.5], 2), "slope_09": _c(slopes[0.9], 2), "slope_099": _c(slopes[0.99], 2),
        "x1_t10": _p(settle[(0.9, 0.9)]["x10"], 2), "x1_t11": _p(settle[(0.9, 0.9)]["x11"], 2), "settled_09": str(settle[(0.9, 0.9)]["settled"]), "predicted_09": str(round(settle[(0.9, 0.9)]["predicted"])),
        "settled_099": str(settle[(0.99, 0.99)]["settled"]), "predicted_099": str(round(settle[(0.99, 0.99)]["predicted"])),
        "bound_total": str(total), "bound_ok": str(ok), "bound_exc": str(total - ok), "bound_worst3": _c(grid["bound3"]["worst"], 2),
        "fp10": str(fixed[(10, "A")]["count"]), "fp50a": str(fixed[(50, "A")]["count"]), "fp50r": str(fixed[(50, "R")]["count"]), "fp20r": str(fixed[(20, "R")]["count"]),
        "ult_a": _p(Ra.grid_share_first("9/10", "9/10", 1, 10, True), 1), "ult_r10": _p(Ra.grid_share_first("9/10", "9/10", 1, 10, False), 1), "ult_r100": _p(Ra.grid_share_first("9/10", "9/10", 1, 100, False), 1),
        "trap": _p(ev.grid_state(Settings(g=10))["share_first"], 1), "trap_t41": _p(float(Ra.grid_share_first("9/10", "9/10", 41, 10, True)), 1), "grid_default": _p(ev.grid_state(d)["share_first"], 1),
        "day_first": _p(day["x_first"], 2), "day_second": _p(day["x_second"], 2), "hour_first": _p(hour["x_first"], 2), "minute_first": _p(minute["x_first"], 2), "limit_12": _p(hour["limit"], 2),
        "edge_1_01": _c(rounds["1.0|1.0"][0.1]["edge"], 4), "edge_1_0001": _c(rounds["1.0|1.0"][0.001]["edge"], 6),
        "costs_a": _p(costs[(0.05, 0.1)]["discount"], 1), "costs_b": _p(costs[(0.1, 0.05)]["discount"], 1), "costs_c": _p(costs[(0.02, 0.3)]["discount"], 1),
        "costs_eq_even": _p(costs[(0.2, 0.2)]["t1000_first"], 1), "costs_eq_odd": _p(costs[(0.2, 0.2)]["t1001_first"], 1),
        "thr_a": _c(thr_a, 4), "thr_c": _c(thr_c, 4), "c_06": _p(1 - O.outside_backward(0.9, 0.9, 0.6, 1001, "C")[0], 1), "scan_cells": str(grid["outside_scan"]["cells"]), "scan_diff": str(grid["outside_scan"]["diff"]),
        "out_dev": ev.sci(max(grid["outside_check"].values())),
        "risk_nash": _p(risk["nash"], 1), "lin_nash": _p(lin["nash"], 1), "nash_a05": _p(S.nash(0.5, 1.0, 0.0), 1), "ks_a05_d01": _p(sol_by_k[1.0]["ks"], 1), "egal_a05_d01": _p(sol_by_k[1.0]["egalitarian"], 1),
        "egal_k10": _p(sol_by_k[10.0]["egalitarian"], 1), "egal_k01": _p(sol_by_k[0.1]["egalitarian"], 1),
        "curved_a05": _p(S.rubinstein_utility(0.9, 0.9, 0.5), 2),
        "learn_min": _p(min(lshares), 1), "learn_runs": str(len(lshares)), "learn_ult_rm": f"{_p(min(learn[('ultimatum', 'rm', 0.0)]['shares']), 0)} bis {_p(max(learn[('ultimatum', 'rm', 0.0)]['shares']), 0)}",
        "learn_two_h2": f"{_p(min(learn[('zwei_runden', 'hedge', 2.0)]['shares']), 1)} bis {_p(max(learn[('zwei_runden', 'hedge', 2.0)]['shares']), 1)}",
        "learn_two_h8": f"{_p(min(learn[('zwei_runden', 'hedge', 8.0)]['shares']), 0)} bis {_p(max(learn[('zwei_runden', 'hedge', 8.0)]['shares']), 0)}",
        "learn_two_rm": f"{_p(min(learn[('zwei_runden', 'rm', 0.0)]['shares']), 0)} bis {_p(max(learn[('zwei_runden', 'rm', 0.0)]['shares']), 0)}",
        "learn_maxgap": _c(max(g for c in grid["learn"] for g in c["gaps"]), 3), "learn_rounds": f"{grid['meta']['learn_rounds']:,}".replace(",", " "),
    }


# --- README gegen die Zeilen-Bauer ------------------------------------------------------------------------------------------------------------------

def _in_readme(lines):
    missing = [line for line in lines if line not in README]
    assert not missing, "fehlt in der README: " + " || ".join(missing[:3])


def test_readme_tables_equal_the_rows_of_the_app(grid):
    for rows in (patience_table(), settle_table(grid), bound_table(grid), fixed_table(grid), rounds_table(grid), costs_table(grid), outside_table(), nash_exit_table(grid), scale_table(grid), curved_table(grid),
                 nash_gap_table(grid), learn_table(grid)):
        assert rows
        _in_readme(rows)


def test_readme_numbers_appear_in_the_readme(grid):
    missing = [f"{k} = {v}" for k, v in numbers(grid).items() if v not in README]
    assert not missing, "fehlt in der README: " + "; ".join(missing)


# --- Die Zahlen selbst, unter ihrer Bedingung -----------------------------------------------------------------------------------------------

def test_h1_the_formula_and_the_default(grid):
    n = numbers(grid)
    assert n["x1_default"] == "52,6 %" and n["x2_default"] == "47,4 %" and R.rubinstein_share(F(9, 10), F(9, 10)) == F(10, 19)
    assert (n["settled_09"], n["predicted_09"], n["settled_099"], n["predicted_099"]) == ("126", "131", "1307", "1375")
    assert (n["x1_t10"], n["x1_t11"]) == ("34,28 %", "69,15 %")


def test_h1_hook_and_slopes(grid):
    n = numbers(grid)
    assert (n["x1_hook"], n["hook_pp"]) == ("55,2 %", "2,62") and (n["slope_05"], n["slope_09"], n["slope_099"]) == ("0,30", "2,49", "7,58")
    # Der Wert der Geduld wächst: Steigung 0,30 < 2,49 < 7,58 bei δ₂ = 0,9 (nicht für andere δ₂ belegt)
    assert ev.patience_slope(0.5, 0.9) < ev.patience_slope(0.9, 0.9) < ev.patience_slope(0.99, 0.9)
    assert [R.rubinstein_share(d, 0.9) for d in (0.0, 0.5, 0.8, 0.9, 0.99)] == pytest.approx([0.1, 2 / 11, 0.35714285714, 10 / 19, 0.917431192])


def test_h2_ultimatum_and_parity():
    assert R.backward_T(0.9, 0.9, 1) == [1] and R.backward_T(0.9, 0.9, 2)[0] == pytest.approx(0.1) and R.backward_T(0.9, 0.9, 3)[0] == pytest.approx(0.91)
    assert R.backward_T(0.9, 0.9, 4)[0] == pytest.approx(0.181)


def test_h3_the_grid(grid):
    n = numbers(grid)
    assert (n["bound_total"], n["bound_ok"], n["bound_exc"], n["bound_worst3"]) == ("200", "196", "4", "0,82")
    assert (n["fp10"], n["fp50a"], n["fp50r"], n["fp20r"]) == ("9", "7", "9", "1")
    assert (n["ult_a"], n["ult_r10"], n["ult_r100"]) == ("100,0 %", "90,0 %", "99,0 %")
    assert (n["trap"], n["trap_t41"], n["grid_default"]) == ("10,0 %", "90,0 %", "50,0 %")                       # G = 10, δ = 0,9, Regel A: T = 200 (gerade) 10 %, T = 41 (ungerade) 90 %


def test_h4_short_rounds(grid):
    n = numbers(grid)
    assert (n["day_first"], n["day_second"], n["hour_first"], n["minute_first"], n["limit_12"]) == ("91,00 %", "33,48 %", "68,05 %", "66,69 %", "66,67 %")
    assert (n["edge_1_01"], n["edge_1_0001"]) == ("0,0500", "0,000500")


def test_h5_costs(grid):
    n = numbers(grid)
    assert (n["costs_a"], n["costs_b"], n["costs_c"]) == ("69,0 %", "34,5 %", "95,5 %") and (n["costs_eq_even"], n["costs_eq_odd"]) == ("20,0 %", "100,0 %")
    # Bedingung „gleiche Kosten“: ungerade T gibt dem ersten Anbieter alles, gerade nur c – für diese Kosten über viele Horizonte
    for c in (0.05, 0.1, 0.2, 0.5):
        assert all(R.costs_backward(c, c, T)[0] == (pytest.approx(1.0) if T % 2 else pytest.approx(c)) for T in range(1, 80))


def test_h6_outside(grid):
    n = numbers(grid)
    assert (n["thr_a"], n["thr_c"]) == ("0,4737", "0,5263") and n["c_06"] == "54,0 %" and (n["scan_cells"], n["scan_diff"]) == ("2400", "0") and n["out_dev"] == "2,2·10⁻⁵"
    # Variante C: o = 0,6 gibt δ·o = 0,54 > 0,4737: der Halter bekommt 54,0 %; o = 0,5 gibt 0,45 < 0,4737: dasselbe wie ohne Option
    assert 1 - O.outside_backward(0.9, 0.9, 0.5, 1001, "C")[0] == pytest.approx(9 / 19, abs=1e-9)


def test_h7_scale(grid):
    n = numbers(grid)
    assert (n["ks_a05_d01"], n["egal_a05_d01"], n["egal_k10"], n["egal_k01"]) == ("59,2 %", "56,2 %", "96,2 %", "8,6 %")
    for a in C.E6_AS:
        rows = sorted((r for r in grid["scale"] if abs(r["a"] - a) < 1e-9), key=lambda r: r["k"])
        assert max(abs(r["nash"] - rows[0]["nash"]) for r in rows) < 1e-9 and max(abs(r["ks"] - rows[0]["ks"]) for r in rows) < 1e-9
        assert abs(rows[0]["egalitarian"] - rows[-1]["egalitarian"]) > 0.3


def test_h8_risk_and_short_rounds(grid):
    n = numbers(grid)
    assert (n["risk_nash"], n["lin_nash"], n["nash_a05"], n["curved_a05"]) == ("62,6 %", "45,0 %", "66,7 %", "70,11 %")
    gaps = {(r["a"], r["dt"]): r["gap"] for r in grid["nash_gap_short"]}
    assert all(0 < gaps[(a, 0.001)] < 5e-4 for a in (1.0, 0.5, 0.25))


def test_h9_learning_bands(grid):
    n = numbers(grid)
    assert n["learn_runs"] == "32" and n["learn_maxgap"] == "0,024"
    lo_hi = {(c["game"], c["rule"], c["eta"]): (min(c["shares"]), max(c["shares"])) for c in grid["learn"]}
    # nur Bandbreiten: Ultimatum Hedge liegt bei 70 bis 80 %, Regret Matching bei 70 bis 90 %, zwei Runden Hedge η₀ = 2 bei 18 %, Hedge η₀ = 8 und Regret Matching 20 bis 60 %
    assert lo_hi[("ultimatum", "hedge", 2.0)] == (pytest.approx(0.8), pytest.approx(0.8)) and lo_hi[("ultimatum", "rm", 0.0)][0] == pytest.approx(0.7) and lo_hi[("ultimatum", "rm", 0.0)][1] == pytest.approx(0.9)
    assert 0.18 <= lo_hi[("zwei_runden", "hedge", 2.0)][0] and lo_hi[("zwei_runden", "hedge", 2.0)][1] < 0.19
    assert lo_hi[("zwei_runden", "hedge", 8.0)] == (pytest.approx(0.2, abs=0.005), pytest.approx(0.6, abs=0.005)) and lo_hi[("zwei_runden", "rm", 0.0)] == (pytest.approx(0.2, abs=0.005), pytest.approx(0.6, abs=0.005))
    assert float(min(c["spe"] for c in grid["learn"] if c["game"] == "zwei_runden")) == pytest.approx(0.1) and n["learn_min"] == "18,2 %"


def test_the_preset_help_texts_carry_the_measured_numbers(grid):
    n = numbers(grid)
    help_ = C.PRESET_HELP
    assert "52,6 %" in help_["Gleiche Geduld (Voreinstellung)"] and n["x1_default"] == "52,6 %"
    assert "52,6 % auf 55,2 %" in help_["Geduld ist Macht"] and n["x1_hook"] == "55,2 %"
    assert "stetig bekommt er 100 %" in help_["Ultimatum (eine Runde)"] and "90 %" in help_["Ultimatum (eine Runde)"] and n["ult_r10"] == "90,0 %"
    assert "68,0 % statt 91,0 %" in help_["Kurze Runden"] and "66,7 %" in help_["Kurze Runden"]
    assert round(100 * ev.short_rounds_row(Settings(dt=1 / 24))["x_first"], 1) == 68.0 and round(100 * ev.short_rounds_row(Settings(dt=1.0))["x_first"], 1) == 91.0
    assert "100 %" in help_["Feste Kosten"] and "69,0 %" in help_["Feste Kosten"] and n["costs_a"] == "69,0 %"
    assert "0,474" in help_["Außenoption unter der Schwelle"] and "47,4 %" in help_["Außenoption unter der Schwelle"] and "60,0 %" in help_["Außenoption über der Schwelle"]
    assert "62,6 %" in help_["Risikoscheu"] and "45,0 %" in help_["Risikoscheu"] and n["risk_nash"] == "62,6 %" and n["lin_nash"] == "45,0 %"
    assert "10,0 %" in help_["Rasterfalle"] and "52,6 %" in help_["Rasterfalle"] and n["trap"] == "10,0 %"


def test_the_assumptions_text_is_the_same_in_app_and_readme():
    assert C.ASSUMPTIONS_MARKDOWN.strip() in README


def test_the_readme_names_the_hypotheses_with_yes_or_no():
    for h in range(1, 12):
        assert f"**H{h}**" in README, h
