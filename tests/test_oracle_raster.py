"""Orakel-Tests Rasterspiel: Aufzählung aller Rasterangebote statt der Kurzformel ⌈G·Fortsetzungswert⌉, Aufzählung aller (A₁, A₂)-Paare für die Fixpunkte.

Das Orakel rechnet in Brüchen, probiert JEDES Angebot k/G aus und wendet die drei Gleichstandsregeln einzeln an (Antwortender: Regel A / R, Anbieter: angenommenes Angebot bevorzugt,
unter angenommenen Angeboten das kleinste = beste für den Anbieter). Handwerte: tests/conftest.py (grid_mini)."""

from fractions import Fraction as F

import pytest

import vhd_raster as Ra
import vhd_rubinstein as R
from vhd_rng import SplitMix64


def enumerate_game(d_first, d_second, T, G, rule_a):
    """Rückwärtsrechnen mit Aufzählung aller Angebote; Liste von (Wert des Anbieters, Wert des Antwortenden) je Runde."""
    out = [None] * T
    nxt = (F(0), F(0))
    for t in range(T, 0, -1):
        d_prop, d_resp = (d_first, d_second) if t % 2 == 1 else (d_second, d_first)
        cont = d_resp * nxt[0] if t < T else F(0)
        rej = d_prop * nxt[1] if t < T else F(0)
        best = None
        for k in range(G + 1):
            y = F(k, G)
            accepted = y >= cont if rule_a else y > cont
            option = (1 - y, y, True) if accepted else (rej, cont, False)
            if best is None or option[0] > best[0] or (option[0] == best[0] and option[2] and not best[2]):
                best = option
        nxt = (best[0], best[1])
        out[t - 1] = nxt
    return out


def stationary_by_enumeration(d_first, d_second, G, rule_a):
    """Alle Paare (A₁, A₂) auf dem Raster, die sich reproduzieren, durch Prüfen jedes Paares (G + 1)² Paare."""
    found = []
    for i in range(G + 1):
        for j in range(G + 1):
            a1, a2 = F(i, G), F(j, G)
            # der erste Anbieter bietet dem Zweiten das kleinste k/G mit k/G ≥ (bzw. >) d₂·A₂ …
            offers1 = [F(k, G) for k in range(G + 1) if (F(k, G) >= d_second * a2 if rule_a else F(k, G) > d_second * a2)]
            offers2 = [F(k, G) for k in range(G + 1) if (F(k, G) >= d_first * a1 if rule_a else F(k, G) > d_first * a1)]
            if offers1 and offers2 and 1 - min(offers1) == a1 and 1 - min(offers2) == a2:
                found.append((a1, a2))
    return found


PAIRS = [(F(9, 10), F(9, 10)), (F(95, 100), F(8, 10)), (F(8, 10), F(95, 100)), (F(1, 2), F(9, 10)), (F(3, 10), F(3, 10))]


@pytest.mark.parametrize("d1,d2", PAIRS)
@pytest.mark.parametrize("rule_a", [True, False])
def test_short_formula_equals_the_enumeration_of_all_offers(d1, d2, rule_a):
    for G in (10, 50):
        for T in list(range(1, 9)) + [40, 41]:
            assert Ra.grid_backward(d1, d2, T, G, rule_a) == enumerate_game(d1, d2, T, G, rule_a), (G, T)


def test_short_formula_equals_the_enumeration_for_random_pairs():
    rng = SplitMix64(5)
    for _ in range(60):
        d1, d2 = F(rng.below(100), 100), F(rng.below(100), 100)
        G, T, rule = (10, 20, 50)[rng.below(3)], 1 + rng.below(30), bool(rng.below(2))
        assert Ra.grid_backward(d1, d2, T, G, rule) == enumerate_game(d1, d2, T, G, rule)


def test_values_without_a_grid_move_are_still_exact_fractions():
    # Abgelehntes Angebot (Zweig d_Anbieter · Wert): kommt vor, wenn kein Angebot angenommen wird oder die Ablehnung besser ist; der Wert bleibt ein exakter Bruch
    res = Ra.grid_backward("99/100", "99/100", 40, 10, False)
    assert all(isinstance(a, F) and isinstance(b, F) for a, b in res)


@pytest.mark.parametrize("rule_a", [True, False])
@pytest.mark.parametrize("d1,d2", PAIRS[:3])
def test_fixed_points_equal_the_enumeration_of_all_pairs(d1, d2, rule_a):
    for G in (10, 20, 50):
        assert Ra.grid_fixed_points(d1, d2, G, rule_a) == stationary_by_enumeration(d1, d2, G, rule_a), G


def test_the_long_game_ends_in_a_fixed_point():
    for d1, d2 in PAIRS[:3]:
        for G in (10, 50, 100):
            for rule_a in (True, False):
                points = {a1 for a1, _ in Ra.grid_fixed_points(d1, d2, G, rule_a)}
                for T in (600, 601):
                    assert Ra.grid_share_first(d1, d2, T, G, rule_a) in points, (d1, d2, G, rule_a, T)


def test_the_grid_game_has_several_equilibria_where_the_continuous_one_is_unique():
    counts = {G: len(Ra.grid_fixed_points("9/10", "9/10", G, True)) for G in (10, 20, 50, 100)}
    assert counts[10] > 1 and counts[50] > 1                   # (Zählung: tests/test_claims.py)


def test_rule_a_and_rule_r_differ_on_the_ultimatum(grid_mini):
    d, G = grid_mini["d"], grid_mini["G"]
    a = Ra.grid_share_first(d, d, 1, G, True)
    r = Ra.grid_share_first(d, d, 1, G, False)
    assert (a, r) == (grid_mini["T1_A"], grid_mini["T1_R"]) and a - r == F(1, G)


def test_rule_a_and_r_differ_by_one_grid_step_at_most_on_long_games():
    for d1, d2 in PAIRS[:3]:
        for G in (10, 100):
            for T in (60, 61):
                assert abs(Ra.grid_share_first(d1, d2, T, G, True) - Ra.grid_share_first(d1, d2, T, G, False)) <= F(1, G)


def test_with_a_fine_grid_the_grid_game_is_close_to_the_formula():
    for d1, d2 in PAIRS[:3]:
        limit = R.rubinstein_share(d1, d2)
        for T in (600, 601):
            assert abs(Ra.grid_share_first(d1, d2, T, 10000, True) - limit) <= Ra.error_bound(d1, d2, 10000)


def test_on_a_dyadic_grid_the_grid_game_equals_the_continuous_one():
    # δ = 1/2: alle Werte der ersten fünf Runden sind Vielfache von 1/16, also trifft das Raster G = 16 die stetige Lösung genau
    half = F(1, 2)
    for T in range(1, 6):
        assert [a for a, _ in Ra.grid_backward(half, half, T, 16, True)] == R.backward_T(half, half, T)
