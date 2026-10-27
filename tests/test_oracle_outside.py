"""Orakel-Tests Außenoption: (1) Spielbaum-Lösung mit Aufzählung aller Rasterangebote UND aller Entscheidungen (annehmen, ablehnen und aussteigen, ablehnen und weitermachen) für kleine T,
(2) die Handformeln für lange Spiele, (3) Zweig-Tests: Variante A und C weichen ab, Variante B ist gleich A (der Grund, sie nicht anzubieten).

Das Orakel (1) rechnet in Brüchen mit δ = 1/2 und einem dyadischen Raster (G = 1 024): bei δ = 1/2, o ∈ {0, …, 8}/8 und T ≤ 5 haben alle Werte höchstens den Nenner 2⁸ = 256 und liegen auf dem Raster, also muss die stetige Lösung exakt herauskommen."""

from fractions import Fraction as F

import pytest

import vhd_outside as O
from vhd_rng import SplitMix64


def tree_solution(d_first, d_second, o, T, variant, G=1024):
    """Rückwärtsrechnen im Spielbaum: Wert (Anbieter, Antwortender) der Runde t. Der Halter (zweiter Anbieter) darf nach Ablehnung aussteigen (A, B: Erlös o sofort; C: d·o);
    in Variante B auch statt zu bieten. Annehmen bei Gleichstand; angenommenes Angebot bei gleicher Auszahlung bevorzugt."""
    values = [None] * (T + 2)
    ex = d_second * o if variant == "C" else o
    for t in range(T, 0, -1):
        holder_responds = t % 2 == 1
        d_prop, d_resp = (d_first, d_second) if holder_responds else (d_second, d_first)
        nxt = values[t + 1] if t < T else (F(0), F(0))
        cont_resp = d_resp * nxt[0]
        reject_prop = d_prop * nxt[1]
        if holder_responds:
            cont_resp = max(cont_resp, ex)                      # Ablehnen heißt: das Bessere aus Weitermachen und Aussteigen
        best = None
        for k in range(G + 1):
            y = F(k, G)
            option = (1 - y, y, True) if y >= cont_resp else (reject_prop, cont_resp, False)
            if best is None or option[0] > best[0] or (option[0] == best[0] and option[2] and not best[2]):
                best = option
        value = (best[0], best[1])
        if variant == "B" and not holder_responds and o > value[0]:          # der Halter steigt statt zu bieten aus
            value = (o, F(0))
        values[t] = value
    return values[1][0]


def test_the_tree_oracle_reproduces_the_hand_values(outside_mini):
    half = F(1, 2)
    assert tree_solution(half, half, outside_mini["o"], 1, "A") == 1 - outside_mini["o"]
    assert tree_solution(half, half, F(0), 3, "A") == F(3, 4)                         # ohne Option: das Rückwärtsrechnen der Fixture halves


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_outside_backward_equals_the_tree_oracle(variant):
    half = F(1, 2)
    for o in (F(0), F(1, 8), F(1, 4), F(3, 8), F(1, 2), F(3, 4), F(1)):
        for T in range(1, 6):
            assert O.outside_backward(half, half, o, T, variant)[0] == tree_solution(half, half, o, T, variant), (variant, o, T)


def test_outside_backward_equals_the_tree_oracle_for_random_dyadic_cases():
    rng = SplitMix64(9)
    for _ in range(40):
        d1, d2 = F(rng.below(5), 4), F(rng.below(5), 4)
        d1, d2 = min(d1, F(1, 2)) if d1 == 1 else d1, min(d2, F(1, 2)) if d2 == 1 else d2           # δ = 1 ausschließen
        o, T, variant = F(rng.below(9), 8), 1 + rng.below(5), "ABC"[rng.below(3)]
        assert O.outside_backward(d1, d2, o, T, variant)[0] == tree_solution(d1, d2, o, T, variant), (d1, d2, o, T, variant)


PAIRS = [(0.9, 0.9), (0.95, 0.8), (0.8, 0.95), (0.7, 0.9)]
OS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 0.95]


@pytest.mark.parametrize("variant", ["A", "B", "C"])
def test_long_games_equal_the_hand_formula(variant):
    for d1, d2 in PAIRS:
        for o in OS:
            for T in (1000, 1001):
                assert O.outside_backward(d1, d2, o, T, variant)[0] == pytest.approx(O.outside_hand(d1, d2, o, variant), abs=1e-9)
    # Beinahe nicht konvergiert (δ = 0,99): die Abweichung zur Formel ist die des Horizonts, kleiner als 1e-4
    assert O.outside_backward(0.99, 0.99, 0.6, 1000, variant)[0] == pytest.approx(O.outside_hand(0.99, 0.99, 0.6, variant), abs=1e-4)


def test_the_option_counts_exactly_above_the_threshold():
    for d1, d2 in PAIRS:
        for variant in ("A", "C"):
            x2 = O.holder_share_without_option(d1, d2)
            thr = O.outside_threshold(d1, d2, variant)

            def ex(o):
                return O.exit_payoff(o, d2, variant)

            below, above = thr - 0.01, thr + 0.01
            assert O.outside_hand(d1, d2, below, variant) == pytest.approx(1 - x2) and ex(below) < x2
            assert O.outside_hand(d1, d2, above, variant) == pytest.approx(1 - ex(above)) and ex(above) > x2


def test_variant_a_and_c_differ_where_the_option_is_between_the_two_thresholds():
    d = 0.9
    o = 0.5                                              # A-Schwelle 0,4737 < 0,5 < C-Schwelle 0,5263
    a = O.outside_backward(d, d, o, 1001, "A")[0]
    c = O.outside_backward(d, d, o, 1001, "C")[0]
    # A: o = 0,5 > 0,4737: der Halter bekommt o, also x₁ = 0,5. C: der Erlös ist nur 0,9 · 0,5 = 0,45 < 0,4737: die Option zählt nicht, x₁ = 1 − 0,4737 = 10/19
    assert a == pytest.approx(0.5, abs=1e-9) and c == pytest.approx(10 / 19, abs=1e-9) and abs(a - c) > 0.02


def test_variant_b_is_variant_a_in_every_cell_of_a_small_scan():
    n = diff = 0
    for d1, d2 in PAIRS:
        for o in OS:
            for T in range(1, 31):
                n += 1
                diff += abs(O.outside_backward(d1, d2, o, T, "A")[0] - O.outside_backward(d1, d2, o, T, "B")[0]) > 1e-12
    assert (n, diff) == (4 * 10 * 30, 0)


def test_outside_options_never_hurt_the_holder():
    for d1, d2 in PAIRS:
        for T in (5, 6, 50, 51):
            shares = [O.outside_backward(d1, d2, o, T, "A")[0] for o in OS]
            assert shares == sorted(shares, reverse=True)                                        # größeres o: weniger für den ersten Anbieter


def test_unknown_variant_is_refused():
    with pytest.raises(ValueError):
        O.outside_backward(0.9, 0.9, 0.5, 10, "X")


def test_zero_rounds_are_refused_in_every_variant_and_the_costs_function():
    import vhd_rubinstein as R
    for variant in ("A", "B", "C"):
        with pytest.raises(ValueError):
            O.outside_backward(0.9, 0.9, 0.5, 0, variant)
    with pytest.raises(ValueError):
        R.costs_backward(0.1, 0.2, 0)


def test_the_holder_never_prefers_to_exit_instead_of_bidding():
    """Warum Variante B = Variante A: in jeder Anbieterrunde des Halters ist sein Wert mindestens o (er bekommt 1 − d₁·(Wert des ersten Anbieters), und dessen Wert ist höchstens 1 − o).
    Der Zweig „aussteigen statt bieten“ greift also nie."""
    rng = SplitMix64(41)
    for _ in range(300):
        d1, d2, o = rng.below(100) / 100, rng.below(100) / 100, rng.below(100) / 100
        T = 2 + rng.below(40)
        values = O.outside_backward(d1, d2, o, T, "A")
        assert all(values[t - 1] >= o - 1e-12 for t in range(2, T + 1, 2)), (d1, d2, o, T)
