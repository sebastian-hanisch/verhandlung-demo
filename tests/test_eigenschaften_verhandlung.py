"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, Spieltheorie und Mechanismen): Aussagen, die ohne das richtige Ergebnis gelten müssen.

Spieler umbenennen, positiv-affine Umrechnung der Nutzen (Nash und Kalai–Smorodinsky unverändert, egalitär nicht), Monotonie in der Geduld auch bei endlichem Horizont, gleicher Seed gleiche Ausgabe."""

import numpy as np
import pytest

import vhd_evaluation as ev
import vhd_learning as L
import vhd_rubinstein as R
from vhd_rng import SplitMix64
from vhd_scenario import Settings


def _pairs(n, seed):
    rng = SplitMix64(seed)
    return [(rng.below(99) / 100, rng.below(99) / 100) for _ in range(n)]


def test_renaming_the_players_mirrors_the_shares():
    for d1, d2 in _pairs(40, 1):
        for t in (1, 2, 5, 40):
            a = ev.analyse(Settings(d1=d1, d2=d2, first=1, t=t))
            b = ev.analyse(Settings(d1=d2, d2=d1, first=2, t=t))
            assert a.x1 == pytest.approx(b.x2, abs=1e-12) and a.x2 == pytest.approx(b.x1, abs=1e-12)


def test_more_patience_never_hurts_and_less_patience_of_the_other_never_hurts_at_every_horizon():
    for d1, d2 in _pairs(60, 2):
        for T in (1, 2, 3, 6, 7, 30, 31):
            base = R.backward_T(d1, d2, T)[0]
            assert R.backward_T(min(d1 + 0.01, 0.99), d2, T)[0] >= base - 1e-12
            assert R.backward_T(d1, min(d2 + 0.01, 0.99), T)[0] <= base + 1e-12


def test_shares_sum_to_one():
    for d1, d2 in _pairs(30, 3):
        a = ev.analyse(Settings(d1=d1, d2=d2, t=17))
        assert a.x1 + a.x2 == pytest.approx(1.0, abs=1e-15)


def _generic(u1, u2, d1, d2, n=20001):
    """Nash, Kalai–Smorodinsky und egalitär auf einer beliebigen Verhandlungsmenge {(u1(x), u2(x))} per Gitter über x (unabhängig von vhd_solutions)."""
    xs = np.linspace(0, 1, n)
    a, b = u1(xs), u2(xs)
    ok = (a >= d1 - 1e-12) & (b >= d2 - 1e-12)
    xs, a, b = xs[ok], a[ok], b[ok]
    nash = xs[int(np.argmax((a - d1) * (b - d2)))]
    ks = xs[int(np.argmin(np.abs((a - d1) / (a.max() - d1) - (b - d2) / (b.max() - d2))))]
    egal = xs[int(np.argmin(np.abs((a - d1) - (b - d2))))]
    return nash, ks, egal


def test_nash_and_ks_are_invariant_under_positive_affine_utility_changes_and_egalitarian_is_not():
    rng = SplitMix64(7)
    for _ in range(12):
        a = 0.3 + rng.below(150) / 100
        l1, b1, l2, b2 = 0.2 + rng.below(80) / 10, rng.below(100) / 10, 0.2 + rng.below(80) / 10, rng.below(100) / 10
        base = _generic(lambda x: x, lambda x: (1 - x) ** a, 0.0, 0.05)
        moved = _generic(lambda x: l1 * x + b1, lambda x: l2 * (1 - x) ** a + b2, b1, l2 * 0.05 + b2)
        assert moved[0] == pytest.approx(base[0], abs=1e-3) and moved[1] == pytest.approx(base[1], abs=1e-3)
        if abs(l1 - l2) > 1.5:
            assert abs(moved[2] - base[2]) > 0.01


def test_the_demos_solutions_agree_with_the_generic_grid_versions():
    import vhd_solutions as S
    for a in (0.5, 1.0, 1.5):
        nash, ks, egal = _generic(lambda x: x, lambda x: 2.0 * (1 - x) ** a, 0.0, 0.2)
        assert S.nash(a, 2.0, 0.2) == pytest.approx(nash, abs=1e-3) and S.kalai_smorodinsky(a, 2.0, 0.2) == pytest.approx(ks, abs=1e-3) and S.egalitarian(a, 2.0, 0.2) == pytest.approx(egal, abs=1e-3)


def test_same_seed_same_output_and_the_order_of_independent_runs_does_not_matter():
    u1, u2 = L.ultimatum_game(5)
    first = [L.run(u1, u2, "hedge", 2.0, 500, s) for s in (1, 2, 3)]
    again = [L.run(u1, u2, "hedge", 2.0, 500, s) for s in (3, 1, 2)]
    assert first[0] == again[1] and first[1] == again[2] and first[2] == again[0]


def test_learning_payoffs_are_bounded_and_strategies_are_distributions():
    u1, u2 = L.two_round_game(4, 0.9)
    r = L.run(u1, u2, "hedge", 8.0, 400, 5)
    assert min(r.p1) >= 0 and min(r.p2) >= 0 and sum(r.p1) == pytest.approx(1.0) and sum(r.p2) == pytest.approx(1.0)
    assert 0 <= r.share1 <= 1 and 0 <= r.share2 <= 1 and r.share1 + r.share2 <= 1 + 1e-12 and r.gap >= -1e-12
