"""Orakel-Tests Rubinstein (unabhängige Rechenwege, ORACLE-PLAYBOOK §2: Spieltheorie): Formel, Rückwärtsrechnen, feste Kosten.

Orakel: (1) das lineare Gleichungssystem A₁ = 1 − δ₂ A₂, A₂ = 1 − δ₁ A₁ mit numpy.linalg.solve, (2) die geschlossene Form der Rückwärtsfolge für endliches T (geometrische Summe),
(3) exaktes Rückwärtsrechnen in Brüchen für T = 500 und 501, (4) für feste Kosten eine Aufzählung ALLER Rasterangebote im Spiel mit T Runden (Gleichstandsregel P) und die
Fallunterscheidung von Hand. Das Orakel selbst ist an den Handwerten der Fixtures (tests/conftest.py) geprüft."""

from fractions import Fraction as F

import numpy as np
import pytest

import vhd_rubinstein as R
from vhd_rng import SplitMix64


def _pairs(n, seed=11):
    rng = SplitMix64(seed)
    return [(F(rng.below(99), 100), F(rng.below(99), 100)) for _ in range(n)]


def closed_form(d1, d2, T):
    """Geschlossene Form der Rückwärtsfolge: T = 2m: (1 − d₂)(1 − q^m)/(1 − q); T = 2m + 1: dasselbe plus q^m, mit q = d₁d₂. (Von Hand für T = 1, 2, 3 geprüft: 1, 1 − d₂, 1 − d₂ + d₁d₂.)"""
    q = d1 * d2
    m = T // 2
    base = (1 - d2) * (1 - q ** m) / (1 - q)
    return base + q ** m if T % 2 == 1 else base


def test_the_oracle_itself_on_the_hand_values(halves):
    for T, key in ((1, "T1"), (2, "T2"), (3, "T3")):
        assert closed_form(halves["d1"], halves["d2"], T) == halves[key]


def test_formula_equals_the_linear_system_for_200_random_pairs():
    for d1, d2 in _pairs(200):
        a = np.linalg.solve(np.array([[1.0, float(d2)], [float(d1), 1.0]]), np.ones(2))
        assert R.rubinstein_share(float(d1), float(d2)) == pytest.approx(a[0], abs=1e-12)
        assert R.second_bidder_share(float(d1), float(d2)) == pytest.approx(a[1], abs=1e-12)


@pytest.mark.parametrize("d1,d2", [(F(9, 10), F(9, 10)), (F(95, 100), F(8, 10)), (F(1, 2), F(9, 10)), (F(8, 10), F(95, 100))])
def test_backward_rounds_equal_the_closed_form_exactly_and_converge_to_the_formula(d1, d2):
    for T in range(1, 41):
        assert R.backward_T(d1, d2, T)[0] == closed_form(d1, d2, T)
    limit = R.rubinstein_share(d1, d2)
    for T in (500, 501):                                                       # exakt in Brüchen: Fehler ≈ (d₁d₂)^(T/2), hier unter 1e-20 (0,95·0,8 = 0,76: 1e-30)
        assert abs(R.backward_T(d1, d2, T)[0] - limit) < F(1, 10 ** 20)


def test_backward_error_falls_like_the_geometric_rate():
    d = F(9, 10)
    limit = R.rubinstein_share(d, d)
    for T in (10, 11, 40, 41):
        err = abs(R.backward_T(d, d, T)[0] - limit)
        assert err == abs(limit - closed_form(d, d, T)) and float(err) <= float(d * d) ** (T // 2)         # Fehler ≤ q^(⌊T/2⌋)


def test_odd_horizons_above_and_even_horizons_below_the_limit():
    for d1, d2 in _pairs(60, seed=5):
        limit = R.rubinstein_share(d1, d2)
        for T in range(1, 13):
            v = R.backward_T(d1, d2, T)[0]
            assert (v >= limit) if T % 2 == 1 else (v <= limit)


def test_share_at_horizons_equals_the_exact_runs_for_200_random_pairs():
    for d1, d2 in _pairs(200, seed=21):
        fast = R.share_at_horizons(float(d1), float(d2), 30)
        slow = [float(R.backward_T(d1, d2, T)[0]) for T in range(1, 31)]
        assert fast == pytest.approx(slow, abs=1e-12)


def test_the_shares_are_in_the_unit_interval_and_monotone_in_patience():
    for d1, d2 in _pairs(100, seed=3):
        for T in (1, 2, 7, 8, 100):
            assert 0 <= R.backward_T(d1, d2, T)[0] <= 1
        d1b = min(d1 + F(1, 100), F(99, 100))
        assert R.rubinstein_share(d1b, d2) >= R.rubinstein_share(d1, d2)             # geduldiger: mehr
        d2b = min(d2 + F(1, 100), F(99, 100))
        assert R.rubinstein_share(d1, d2b) <= R.rubinstein_share(d1, d2)             # der andere geduldiger: weniger


def test_roles_can_be_swapped():
    for d1, d2 in _pairs(50, seed=8):
        for T in (1, 2, 5, 6):
            assert R.share_of_player1(d1, d2, T, first=2) == 1 - R.backward_T(d2, d1, T)[0]
        assert R.rubinstein_share(d1, d2) == 1 - d2 * R.second_bidder_share(d1, d2)         # A₁ = 1 − d₂ A₂
        assert R.second_bidder_share(d1, d2) == 1 - d1 * R.rubinstein_share(d1, d2)         # A₂ = 1 − d₁ A₁


# --- feste Kosten: Aufzählung aller Rasterangebote -------------------------------------------------------------------------------------------

def costs_enumeration(c1, c2, T, G):
    """Aufzählung: der Anbieter wählt unter allen Angeboten k/G das beste; der Antwortende nimmt an, wenn sein Angebot mindestens seinen Fortsetzungswert nach Kosten erreicht
    (nächster Wert − Kosten); bei gleicher Auszahlung des Anbieters wird ein angenommenes Angebot vorgezogen, unter angenommenen das kleinste."""
    a_next = None
    b_next = None
    for t in range(T, 0, -1):
        c_prop, c_resp = (c1, c2) if t % 2 == 1 else (c2, c1)
        if t == T:
            cont_resp, reject_prop = F(0), F(0)
        else:
            cont_resp = a_next - c_resp                       # kann negativ sein: dann wird jedes Angebot ≥ 0 angenommen
            reject_prop = b_next - c_prop
        best = None
        for k in range(G + 1):
            y = F(k, G)
            if y >= cont_resp:
                option = (1 - y, y, True)
            else:
                option = (reject_prop, cont_resp, False)
            if best is None or option[0] > best[0] or (option[0] == best[0] and option[2] and not best[2]):
                best = option
        a_next, b_next = best[0], best[1]
    return a_next


@pytest.mark.parametrize("c1,c2", [(F(1, 20), F(1, 10)), (F(1, 10), F(1, 20)), (F(1, 5), F(1, 5)), (F(0), F(1, 10)), (F(3, 5), F(7, 10)), (F(6, 5), F(3, 10))])
def test_costs_backward_equals_the_enumeration_of_all_grid_offers(c1, c2):
    for T in range(1, 11):
        assert R.costs_backward(c1, c2, T)[0] == costs_enumeration(c1, c2, T, 20)           # alle Kosten sind Vielfache von 1/20: die stetige Lösung liegt auf dem Raster


def test_costs_prediction_equals_the_long_game_in_all_unequal_cases():
    for c1, c2 in ((0.05, 0.10), (0.10, 0.05), (0.02, 0.30), (0.30, 0.02), (0.0, 0.05), (0.05, 0.0), (0.6, 0.7), (1.2, 0.3)):
        a1, a2 = R.costs_prediction(c1, c2)
        for T in (1000, 1001):
            values = R.costs_backward(c1, c2, T)
            assert values[0] == pytest.approx(a1, abs=1e-9) and values[1] == pytest.approx(a2, abs=1e-9)


def test_equal_costs_depend_on_the_parity_of_the_horizon():
    assert R.costs_backward(0.1, 0.1, 1001)[0] == pytest.approx(1.0) and R.costs_backward(0.1, 0.1, 1000)[0] == pytest.approx(0.1)          # ungerade: 1, gerade: c


def test_costs_with_more_horizon_never_hurt_the_cheaper_player():
    # c₁ < c₂: der erste Anbieter bekommt in geraden Horizonten T = 2, 4, 6, … immer mehr, bis 1 (Muster der Messung, hier an einem Beispiel nachgezählt)
    values = [R.costs_backward(0.05, 0.10, T)[0] for T in range(2, 60, 2)]
    assert values == sorted(values) and values[0] == pytest.approx(0.10) and values[-1] == 1.0
