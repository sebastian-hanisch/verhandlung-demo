"""Orakel-Tests Lösungskonzepte: scipy (minimize_scalar, brentq) und ein dichtes Gitter statt der Bisektion der Demo; Rubinstein bei gekrümmtem Nutzen gegen das Rückwärtsrechnen
direkt in Nutzen; Eigenschaftstests (Maßstab, Rolle, Grenzfall kurze Runden).

Referenzen: Nash und Kalai–Smorodinsky gegen scipy.optimize (bounded Brent / brentq auf unabhängig hergeleiteten Funktionen) UND ein Gitter mit 400 001 Punkten; Handwerte der Fixture
solutions_mini. Scipy gibt auf flachen Maxima nur ~1e-8 her (ORACLE-PLAYBOOK §2): die Toleranzen liegen bei 1e-6 (Gitter 1e-5)."""

import math

import numpy as np
import pytest

import vhd_solutions as S

scipy_opt = pytest.importorskip("scipy.optimize")

GRID = np.linspace(0.0, 1.0, 400_001)
CASES = [(a, k, t * k) for a in (0.25, 0.5, 1.0, 1.5, 2.0) for k in (0.1, 1.0, 10.0) for t in (0.0, 0.05, 0.1, 0.3)]


def grid_argmax(values, mask):
    v = np.where(mask, values, -np.inf)
    return GRID[int(np.argmax(v))]


def test_nash_equals_scipy_and_the_grid():
    for a, k, d2 in CASES:
        x_max = 1 - (d2 / k) ** (1 / a)
        inside = GRID <= x_max
        product = GRID * (k * (1 - GRID) ** a - d2)
        ref_grid = grid_argmax(product, inside)
        res = scipy_opt.minimize_scalar(lambda x: -(x * (k * (1 - x) ** a - d2)), bounds=(0, x_max), method="bounded", options={"xatol": 1e-12})
        mine = S.nash(a, k, d2)
        assert mine == pytest.approx(res.x, abs=1e-6), (a, k, d2)
        assert mine == pytest.approx(ref_grid, abs=1e-5), (a, k, d2)


def test_nash_closed_form_without_threat():
    for a in (0.25, 0.5, 1.0, 2.0):
        assert S.nash(a, 1.0, 0.0) == pytest.approx(1 / (1 + a), abs=1e-15)
        assert S.nash(a, 7.0, 0.0) == pytest.approx(1 / (1 + a), abs=1e-15)


def test_nash_is_the_root_of_the_independent_derivative():
    for a, k, d2 in CASES:

        def g(x):                                  # unabhängig von der Bisektion der Demo: brentq auf dem ganzen Zulässigkeitsintervall
            return k * (1 - x) ** (a - 1) * (1 - (1 + a) * x) - d2

        root = scipy_opt.brentq(g, 0.0, 1 / (1 + a), xtol=1e-14) if g(1 / (1 + a)) < 0 else 1 / (1 + a)
        assert S.nash(a, k, d2) == pytest.approx(root, abs=1e-9), (a, k, d2)


def test_kalai_smorodinsky_equals_brentq_and_the_grid():
    for a, k, d2 in CASES:
        x_max = 1 - (d2 / k) ** (1 / a)

        def h(x):
            return x / x_max - (k * (1 - x) ** a - d2) / (k - d2)

        ref = scipy_opt.brentq(h, 0.0, x_max, xtol=1e-14)
        inside = GRID <= x_max
        ref_grid = GRID[inside][int(np.argmin(np.abs(h(GRID[inside]))))]
        assert S.kalai_smorodinsky(a, k, d2) == pytest.approx(ref, abs=1e-9), (a, k, d2)
        assert S.kalai_smorodinsky(a, k, d2) == pytest.approx(ref_grid, abs=1e-5), (a, k, d2)


def test_egalitarian_equals_brentq_and_the_grid():
    for a, k, d2 in CASES:
        x_max = 1 - (d2 / k) ** (1 / a)

        def h(x):
            return x - (k * (1 - x) ** a - d2)

        ref = scipy_opt.brentq(h, 0.0, x_max, xtol=1e-14)
        inside = GRID <= x_max
        ref_grid = GRID[inside][int(np.argmin(np.abs(h(GRID[inside]))))]
        assert S.egalitarian(a, k, d2) == pytest.approx(ref, abs=1e-9), (a, k, d2)
        assert S.egalitarian(a, k, d2) == pytest.approx(ref_grid, abs=1e-5), (a, k, d2)


def test_utilitarian_equals_the_grid_maximum():
    for a, k, d2 in CASES:
        x_max = 1 - (d2 / k) ** (1 / a)
        inside = GRID <= x_max
        total = GRID + k * (1 - GRID) ** a
        ref_grid = grid_argmax(total, inside)                                        # erstes Maximum = kleinster Anteil bei Gleichstand
        assert S.utilitarian(a, k, d2) == pytest.approx(ref_grid, abs=1e-5), (a, k, d2)


# --- Eigenschaften --------------------------------------------------------------------------------------------------------------------------

def test_nash_and_ks_do_not_change_with_the_scale_but_egalitarian_and_utilitarian_do():
    for a in (0.5, 1.0, 1.5):
        base = S.all_solutions(a, 1.0, 0.1)
        for k in (0.1, 10.0):
            scaled = S.all_solutions(a, k, S.scaled_threat(0.1, k))
            assert scaled["nash"] == pytest.approx(base["nash"], abs=1e-9) and scaled["ks"] == pytest.approx(base["ks"], abs=1e-9), (a, k)
            assert abs(scaled["egalitarian"] - base["egalitarian"]) > 0.05, (a, k)
    for k in (0.1, 10.0):                                                              # utilitaristisch: bei a = 0,5 im Inneren bzw. an der Ecke
        assert S.all_solutions(0.5, k, S.scaled_threat(0.1, k))["utilitarian"] != pytest.approx(S.all_solutions(0.5, 1.0, 0.1)["utilitarian"], abs=0.05)


def test_ks_and_egalitarian_coincide_without_threat_and_split_with_it():
    for a in (0.25, 0.5, 2.0):
        s0 = S.all_solutions(a, 1.0, 0.0)
        assert s0["ks"] == pytest.approx(s0["egalitarian"], abs=1e-9)
        s1 = S.all_solutions(a, 1.0, 0.1)
        assert abs(s1["ks"] - s1["egalitarian"]) > 1e-3


def test_all_solutions_are_inside_the_agreement_set():
    for a, k, d2 in CASES:
        x_max = 1 - (d2 / k) ** (1 / a)
        for name, x in S.all_solutions(a, k, d2).items():
            assert 0.0 <= x <= x_max + 1e-12 or name == "half", (name, a, k, d2)


def test_risk_averse_second_player_shifts_nash_towards_the_first():
    assert S.nash(1.0, 1.0, 0.0) == pytest.approx(0.5) and S.nash(0.5, 1.0, 0.0) == pytest.approx(2 / 3) and S.nash(0.25, 1.0, 0.0) == pytest.approx(0.8)


def backward_in_utilities(d1, d2, a, T):
    """Rückwärtsrechnen im Spiel mit T Runden direkt in Nutzen: Spieler 2 hat den Nutzen y^a seines Anteils y, Spieler 1 den Anteil. Der Anbieter gibt dem Antwortenden genau
    dessen Fortsetzungsnutzen (gerechnet in Nutzen, dann zurück in den Anteil durch die Potenz 1/a)."""
    share_next = None                       # Anteil des Anbieters der Runde t + 1
    for t in range(T, 0, -1):
        if t == T:
            share_next = 1.0
            continue
        if t % 2 == 1:                      # Spieler 1 bietet, Spieler 2 antwortet: sein Fortsetzungsnutzen d₂ · (Anteil von Spieler 2 als Anbieter)^a
            cont_utility = d2 * share_next ** a
            share_next = 1 - cont_utility ** (1 / a)
        else:                               # Spieler 2 bietet, Spieler 1 (linear) antwortet
            share_next = 1 - d1 * share_next
    return share_next


def test_rubinstein_with_curved_utility_equals_the_backward_run_in_utilities():
    for a in (0.25, 0.5, 1.0, 2.0):
        for d1, d2 in ((0.9, 0.9), (0.8, 0.95), (0.95, 0.7)):
            f = S.rubinstein_utility(d1, d2, a)
            assert backward_in_utilities(d1, d2, a, 600) == pytest.approx(f, abs=1e-9), (a, d1, d2)
            assert backward_in_utilities(d1, d2, a, 601) == pytest.approx(f, abs=1e-9), (a, d1, d2)


def test_short_rounds_approach_the_nash_solution_linearly():
    for a in (0.25, 0.5, 1.0):
        diffs = []
        for dt in (1e-2, 1e-3, 1e-4):
            d = math.exp(-dt)
            diffs.append(S.rubinstein_utility(d, d, a) - S.nash(a, 1.0, 0.0))
        assert all(x > 0 for x in diffs) and diffs[0] / diffs[1] == pytest.approx(10, rel=0.05) and diffs[1] / diffs[2] == pytest.approx(10, rel=0.05)
        assert diffs[1] < 5e-4                                       # bei Rundenlänge 0,001 unter 5·10⁻⁴ (a = 1, 0,5, 0,25)


def test_a_riskier_second_player_gets_more():
    # a = 2 (Menge nicht konvex): das Nash-Produkt wird bei 1/3 maximal, Spieler 2 bekommt mehr als die Hälfte
    assert S.nash(2.0, 1.0, 0.0) == pytest.approx(1 / 3, abs=1e-12)
    assert S.rubinstein_utility(0.9, 0.9, 2.0) < S.rubinstein_utility(0.9, 0.9, 1.0) < S.rubinstein_utility(0.9, 0.9, 0.5)


def test_short_rounds_limit_is_the_weighted_nash_solution():
    """Der Grenzwert r₂/(r₁ + r₂) maximiert das gewichtete Nash-Produkt x^(1/r₁) · (1 − x)^(1/r₂) (Gitter mit 400 001 Punkten, unabhängig von der Formel)."""
    import vhd_rubinstein as R
    xs = GRID[1:-1]
    for r1, r2 in ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (1.0, 4.0), (0.5, 3.0), (5.0, 0.1)):
        log_product = np.log(xs) / r1 + np.log(1 - xs) / r2
        assert xs[int(np.argmax(log_product))] == pytest.approx(R.limit_share_short_rounds(r1, r2), abs=1e-5)
