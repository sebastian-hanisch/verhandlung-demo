"""Orakel-Tests Lernende Verhandler: Schleifenfassung ohne numpy (math.exp, Listen) mit demselben Zufallsstrom gegen die vektorisierte Fassung; reine Nash-Gleichgewichte durch Prüfen
jedes Aktionspaars; Nash-Lücke durch Schleifen; teilspielperfektes Ergebnis durch Rückwärtsrechnen im Spielbaum (nicht aus der Normalform).

Lernergebnisse selbst (Anteile nach 20 000 Runden) sind plattform- und seedabhängig und stehen nur als Bandbreiten in tests/test_claims.py."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import vhd_learning as L
from vhd_rng import SplitMix64


def loop_run(u1, u2, rule, eta0, rounds, seed):
    """Schleifenfassung desselben Verfahrens: Listen statt Arrays, math.exp, kumulierte Summen von Hand."""
    rng = SplitMix64(seed)
    n1, n2 = len(u1), len(u1[0])
    c1, c2 = [0.0] * n1, [0.0] * n2
    p1, p2 = [1.0 / n1] * n1, [1.0 / n2] * n2

    def draw(p, u):
        acc = 0.0
        for i, w in enumerate(p):
            acc += w
            if u < acc:
                return i
        return len(p) - 1

    def weights(c, t):
        if rule == "hedge":
            eta = eta0 / math.sqrt(t)
            top = max(c)
            raw = [math.exp(eta * (x - top)) for x in c]
        else:
            raw = [max(x, 0.0) for x in c]
            if sum(raw) <= 0:
                return [1.0 / len(c)] * len(c)
        s = sum(raw)
        return [x / s for x in raw]

    for t in range(1, rounds + 1):
        a1 = draw(p1, (rng.next() >> 11) / float(1 << 53))
        a2 = draw(p2, (rng.next() >> 11) / float(1 << 53))
        v1 = [u1[i][a2] for i in range(n1)]
        v2 = [u2[a1][j] for j in range(n2)]
        if rule == "hedge":
            c1 = [c + v for c, v in zip(c1, v1)]
            c2 = [c + v for c, v in zip(c2, v2)]
        else:
            c1 = [c + v - v1[a1] for c, v in zip(c1, v1)]
            c2 = [c + v - v2[a2] for c, v in zip(c2, v2)]
        p1, p2 = weights(c1, t), weights(c2, t)
    return p1, p2


@pytest.mark.parametrize("rule,eta0", [("hedge", 0.5), ("hedge", 2.0), ("hedge", 8.0), ("rm", 0.0)])
@pytest.mark.parametrize("game,G", [("ultimatum", 4), ("zwei_runden", 2)])
def test_vectorised_run_equals_the_loop_version_with_the_same_random_stream(rule, eta0, game, G):
    u1, u2 = L.game_matrices(game, G, 0.9)
    for seed in (1, 2, 3):
        mine = L.run(u1, u2, rule, eta0, 400, seed)
        ref1, ref2 = loop_run(u1.tolist(), u2.tolist(), rule, eta0, 400, seed)
        assert list(mine.p1) == pytest.approx(ref1, abs=1e-9) and list(mine.p2) == pytest.approx(ref2, abs=1e-9)


def test_pure_nash_equals_the_check_of_every_action_pair():
    for game, G in (("ultimatum", 2), ("ultimatum", 5), ("ultimatum", 10), ("zwei_runden", 2), ("zwei_runden", 3)):
        u1, u2 = L.game_matrices(game, G, 0.9)
        brute = []
        for i in range(u1.shape[0]):
            for j in range(u1.shape[1]):
                if all(u1[i, j] >= u1[k, j] - 1e-12 for k in range(u1.shape[0])) and all(u2[i, j] >= u2[i, k] - 1e-12 for k in range(u1.shape[1])):
                    brute.append((i, j))
        assert L.pure_nash_equilibria(u1, u2) == brute, (game, G)


def test_the_ultimatum_has_G_plus_3_pure_equilibria_and_exactly_one_is_subgame_perfect():
    for G in (2, 4, 10):
        u1, u2 = L.ultimatum_game(G)
        eq = L.pure_nash_equilibria(u1, u2)
        assert len(eq) == G + 3                                    # (y, θ = y) für y = 0 … G, dazu (0, G), (0, G+1)
        assert (0, 0) in eq and sum(1 for i, j in eq if u1[i, j] == 1.0) == 1       # nur (0, 0) gibt dem Anbieter alles = Ergebnis des Rückwärtsrechnens (Annahme bei Gleichstand)


def test_the_subgame_perfect_outcome_by_backward_induction_in_the_tree():
    # Baum: Spieler 1 wählt y, Spieler 2 nimmt an (y/G, bei Gleichstand y = 0 annehmen) oder lehnt ab (0). Spieler 2 nimmt an, wenn y/G ≥ 0, also immer: Spieler 1 wählt y = 0 und behält alles.
    for G in (2, 10):
        best = max(F(G - y, G) for y in range(G + 1))
        assert L.spe_share("ultimatum", G) == best == 1
    # zwei Runden: Spieler 2 antwortet in Runde 1 mit 0,9 · (alles) als Alternative; Spieler 1 bietet ihm genau das (Raster: ⌈9⌉/10)
    assert L.spe_share("zwei_runden", 10, 0.9) == F(1, 10)


def test_nash_gap_equals_the_loop_version():
    rng = np.random.default_rng(0)
    for _ in range(50):
        u1, u2 = rng.random((4, 5)), rng.random((4, 5))
        p1, p2 = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(5))
        v1 = sum(p1[i] * p2[j] * u1[i, j] for i in range(4) for j in range(5))
        v2 = sum(p1[i] * p2[j] * u2[i, j] for i in range(4) for j in range(5))
        g1 = max(sum(p2[j] * u1[i, j] for j in range(5)) for i in range(4)) - v1
        g2 = max(sum(p1[i] * u2[i, j] for i in range(4)) for j in range(5)) - v2
        assert L.nash_gap(u1, u2, p1, p2) == pytest.approx(max(g1, g2), abs=1e-12)
        assert L.nash_gap(u1, u2, p1, p2) >= -1e-12


def test_expected_shares_equal_the_double_sum():
    u1, u2 = L.ultimatum_game(3)
    p1, p2 = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.1, 0.1, 0.1, 0.2])
    s1, s2 = L.expected_shares(u1, u2, p1, p2)
    assert s1 == pytest.approx(sum(p1[i] * p2[j] * u1[i, j] for i in range(4) for j in range(5)))
    assert s2 == pytest.approx(sum(p1[i] * p2[j] * u2[i, j] for i in range(4) for j in range(5)))


def test_the_two_learning_rules_differ_on_at_least_one_instance():
    u1, u2 = L.ultimatum_game(10)
    a = L.run(u1, u2, "hedge", 2.0, 2000, 1)
    b = L.run(u1, u2, "rm", 0.0, 2000, 1)
    c = L.run(u1, u2, "hedge", 8.0, 2000, 1)
    assert a != b and a != c and b != c


def test_a_nash_equilibrium_is_a_fixed_point_of_the_gap():
    u1, u2 = L.ultimatum_game(10)
    for i, j in L.pure_nash_equilibria(u1, u2):
        assert L.nash_gap(u1, u2, np.eye(11)[i], np.eye(12)[j]) == pytest.approx(0.0, abs=1e-12)
    assert L.nash_gap(u1, u2, np.eye(11)[3], np.eye(12)[7]) > 0.0               # (y = 0,3, θ = 0,7) ist keines
