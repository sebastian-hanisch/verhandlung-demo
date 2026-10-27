"""Einheiten an den von Hand gerechneten Mini-Instanzen (tests/conftest.py, DEMO-PLAYBOOK §1b): je eine Regel, ein Erwartungswert, der als Rechnung dasteht."""

import math
import time
from fractions import Fraction as F

import numpy as np
import pytest

import vhd_learning as L
import vhd_outside as O
import vhd_raster as Ra
import vhd_rubinstein as R
import vhd_solutions as S

# --- Rubinstein ------------------------------------------------------------------------------------------------------------------------------

def test_formula_on_the_halves(halves):
    assert R.rubinstein_share(halves["d1"], halves["d2"]) == F(1, 2) / F(3, 4) == halves["limit"]
    assert R.second_bidder_share(halves["d1"], halves["d2"]) == halves["second"]


def test_backward_rounds_on_the_halves(halves):
    d1, d2 = halves["d1"], halves["d2"]
    assert R.backward_T(d1, d2, 1) == [halves["T1"]]
    assert R.backward_T(d1, d2, 2) == [halves["T2"], F(1)]
    assert R.backward_T(d1, d2, 3) == [halves["T3"], F(1, 2), F(1)]


def test_backward_values_are_fractions_for_fractions_and_floats_for_floats(halves):
    assert isinstance(R.backward_T(halves["d1"], halves["d2"], 5)[0], F)
    assert isinstance(R.backward_T(0.5, 0.5, 5)[0], float)


def test_backward_needs_at_least_one_round():
    with pytest.raises(ValueError):
        R.backward_T(0.5, 0.5, 0)


def test_share_at_horizons_equals_the_single_runs(halves):
    got = R.share_at_horizons(0.5, 0.5, 6)
    assert got == pytest.approx([1.0, 0.5, 0.75, 0.625, 0.6875, 0.65625])      # 1; 1/2; 3/4; v₁(4) = 1 − 1/2 · v₂, v₂ = 3/4 ... von Hand: 5/8, 11/16, 21/32
    assert got == pytest.approx([float(R.backward_T(halves["d1"], halves["d2"], T)[0]) for T in range(1, 7)])


def test_share_of_player1_swaps_the_roles(halves):
    assert R.share_of_player1(F(1, 2), F(1, 4), 3, first=1) == 1 - F(1, 4) * (1 - F(1, 2))        # v₃ = 1, v₂ = 1 − δ₁ = 1/2, v₁ = 1 − δ₂ v₂ = 7/8
    assert R.share_of_player1(F(1, 2), F(1, 4), 3, first=2) == 1 - (1 - F(1, 2) * (1 - F(1, 4)))      # Spieler 2 zuerst: v₂' = 1 − 1/4·... von Hand: 1 − 5/8 = 3/8


def test_first_mover_edge_on_equal_patience():
    assert R.first_mover_edge(0.9, 0.9) == pytest.approx(0.1 / 1.9)                                   # (1 − d)/(1 + d)
    assert R.first_mover_edge(0.5, 0.5) == pytest.approx(1 / 3)


def test_settling_horizon_counts_from_the_last_miss():
    assert R.settling_horizon([5, 1, 1, 5, 1, 1], 1, 0.5) == 5                 # T = 4 ist wieder daneben: gezählt wird ab dem letzten Fehlschlag
    assert R.settling_horizon([5, 5, 5], 1, 0.5) is None
    assert R.settling_horizon([1, 1], 1, 0.5) == 1


def test_error_bound_rounds_is_the_geometric_prediction():
    assert R.error_bound_rounds(0.9, 0.9, 1e-6) == pytest.approx(2 * math.log(1e-6) / math.log(0.81))         # 131,1


def test_costs_on_the_mini_instance(costs_mini):
    assert R.costs_backward(costs_mini["c1"], costs_mini["c2"], 2) == [costs_mini["T2"], F(1)]
    assert R.costs_backward(costs_mini["c1"], costs_mini["c2"], 1) == [F(1)]


def test_costs_never_pay_less_than_nothing_to_the_responder():
    # c₂ = 2 übersteigt jeden Anteil: der Antwortende nimmt auch 0; v₁ = 1 − max(0, 1 − 2) = 1
    assert R.costs_backward(F(1, 10), F(2), 2) == [F(1), F(1)]


def test_costs_prediction_cases():
    assert R.costs_prediction(0.05, 0.10) == (1, 0.05)
    assert R.costs_prediction(0.10, 0.05) == (0.05, 1)
    assert R.costs_prediction(0.1, 0.1) is None


def test_rates_to_discount_and_short_round_limit():
    assert R.delta_from_rate(1.0, 0.0) == 1.0 and R.delta_from_rate(math.log(2), 1.0) == pytest.approx(0.5)
    assert R.limit_share_short_rounds(1.0, 2.0) == pytest.approx(2 / 3)
    assert R.short_round_rate(1.0, 1.0) == pytest.approx(0.25)
    assert R.share_first_continuous_time(math.log(2), math.log(2), 1.0) == pytest.approx(2 / 3)        # d₁ = d₂ = 1/2: 2/3


# --- Raster ----------------------------------------------------------------------------------------------------------------------------------

def test_smallest_acceptable_offer_by_rule():
    assert Ra.smallest_acceptable(F(9, 10), 10, True) == 9                  # ⌈9⌉
    assert Ra.smallest_acceptable(F(9, 10), 10, False) == 10                # ⌊9⌋ + 1
    assert Ra.smallest_acceptable(F(81, 100), 10, True) == 9                # ⌈8,1⌉
    assert Ra.smallest_acceptable(F(81, 100), 10, False) == 9               # ⌊8,1⌋ + 1
    assert Ra.smallest_acceptable(F(0), 10, True) == 0 and Ra.smallest_acceptable(F(0), 10, False) == 1


def test_to_fraction_has_no_binary_remainder():
    assert Ra.to_fraction(0.9) == F(9, 10) and Ra.to_fraction("0.05") == F(1, 20) and Ra.to_fraction(F(1, 3)) == F(1, 3)


def test_grid_backward_on_the_mini_instance(grid_mini):
    d, G = grid_mini["d"], grid_mini["G"]
    assert Ra.grid_backward(d, d, 1, G, True)[0][0] == grid_mini["T1_A"]
    assert Ra.grid_backward(d, d, 1, G, False)[0][0] == grid_mini["T1_R"]
    assert Ra.grid_backward(d, d, 2, G, True)[0] == (grid_mini["T2_A"], F(9, 10))
    assert Ra.grid_backward(d, d, 2, G, False)[0][0] == grid_mini["T2_R"]


def test_grid_values_stay_on_the_grid_when_an_offer_is_accepted():
    for a, b in Ra.grid_backward("9/10", "9/10", 7, 10, True):
        assert (a * 10).denominator == 1 and (b * 10).denominator == 1 and a + b == 1


def test_grid_fixed_points_contain_the_backward_result_of_long_games():
    points = Ra.grid_fixed_points("9/10", "9/10", 100, True)
    assert len(points) >= 1
    a_long = Ra.grid_share_first("9/10", "9/10", 600, 100, True)
    assert a_long in {a1 for a1, _ in points}


def test_error_bound_is_one_over_one_minus_product_times_g():
    assert Ra.error_bound(F(9, 10), F(9, 10), 100) == 1 / (F(19, 100) * 100)


# --- Außenoption -----------------------------------------------------------------------------------------------------------------------------

def test_outside_threshold_and_share_on_the_mini_instance(outside_mini):
    d1, d2, o = outside_mini["d1"], outside_mini["d2"], outside_mini["o"]
    assert O.holder_share_without_option(d1, d2) == outside_mini["threshold_A"]
    assert O.outside_threshold(d1, d2, "A") == outside_mini["threshold_A"] and O.outside_threshold(d1, d2, "C") == outside_mini["threshold_C"]
    assert O.outside_hand(d1, d2, o, "A") == outside_mini["x1_A"]
    assert O.outside_hand(d1, d2, o, "C") == outside_mini["x1_C"]
    assert O.outside_backward(d1, d2, o, 1, "A") == [1 - o]               # eine Runde: der Halter lehnt ab und steigt aus, also muss ihm der Anbieter o bieten
    assert O.outside_backward(d1, d2, o, 1, "C") == [1 - d2 * o]


def test_outside_exit_payoff_per_variant():
    assert O.exit_payoff(0.6, 0.9, "A") == 0.6 and O.exit_payoff(0.6, 0.9, "B") == 0.6 and O.exit_payoff(0.6, 0.9, "C") == pytest.approx(0.54)
    with pytest.raises(ValueError):
        O.exit_payoff(0.6, 0.9, "X")


def test_outside_below_threshold_changes_nothing_and_above_it_pays_exactly_o():
    base = R.backward_T(0.9, 0.9, 300)[0]
    assert O.outside_backward(0.9, 0.9, 0.4, 300, "A")[0] == pytest.approx(base, abs=1e-12)        # 0,4 < 0,4737
    assert O.outside_backward(0.9, 0.9, 0.6, 300, "A")[0] == pytest.approx(0.4, abs=1e-12)         # x₂ = o


def test_nash_with_threat_is_a_different_reading_of_outside_option():
    assert O.nash_with_threat_holder_share(0.2) == pytest.approx(0.6)
    assert O.nash_with_threat_holder_share(0.2) != pytest.approx(1 - O.outside_hand(0.9999, 0.9999, 0.2, "A"), abs=0.05)         # Ausstiegsprinzip: 0,5


# --- Lösungskonzepte -------------------------------------------------------------------------------------------------------------------------

def test_solutions_on_the_mini_instance(solutions_mini):
    m = solutions_mini
    got = S.all_solutions(m["a"], m["k"], m["d2"])
    assert got["nash"] == pytest.approx(m["nash"], abs=1e-12)
    assert got["utilitarian"] == pytest.approx(m["utilitarian"], abs=1e-12)
    assert got["egalitarian"] == pytest.approx(m["egalitarian"], abs=1e-12)
    assert got["ks"] == pytest.approx(m["ks"], abs=1e-12)
    assert got["half"] == 0.5


def test_max_share_is_the_ideal_point_on_the_agreements():
    assert S.max_share_player1(0.5, 1.0, 0.1) == pytest.approx(1 - 0.01)               # 1 − (0,1)^(1/0,5)
    assert S.max_share_player1(1.0, 1.0, 0.0) == 1.0


def test_linear_frontier_with_threat_gives_everybody_the_same_split():
    # u₁ = x, u₂ = 1 − x, Drohpunkt (0, 0,1): Überschuss 0,9, geteilt: x = 0,45 bei Nash, Kalai–Smorodinsky und egalitär
    got = S.all_solutions(1.0, 1.0, 0.1)
    assert got["nash"] == pytest.approx(0.45) and got["ks"] == pytest.approx(0.45) and got["egalitarian"] == pytest.approx(0.45)


def test_utilitarian_ties_go_to_the_smallest_share():
    assert S.utilitarian(1.0, 1.0, 0.0) == 0.0                 # u₁ + u₂ = 1 überall
    assert S.utilitarian(1.0, 2.0, 0.0) == 0.0                 # k > 1: alles an Spieler 2
    assert S.utilitarian(1.0, 0.5, 0.0) == 1.0                 # k < 1: alles an Spieler 1


def test_solution_inputs_are_checked():
    with pytest.raises(ValueError):
        S.nash(1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        S.nash(0.0, 1.0, 0.0)


def test_scaled_threat_and_rubinstein_utility():
    assert S.scaled_threat(0.1, 10.0) == pytest.approx(1.0)
    assert S.rubinstein_utility(0.9, 0.9, 1.0) == pytest.approx(10 / 19)
    # a = 1/2: Spielers 2 Diskont wirkt als 0,9² = 0,81: x₁ = (1 − 0,81)/(1 − 0,9·0,81) = 0,19/0,271
    assert S.rubinstein_utility(0.9, 0.9, 0.5) == pytest.approx(0.19 / 0.271)


# --- Lernen ----------------------------------------------------------------------------------------------------------------------------------

def test_ultimatum_matrices_on_the_mini_game(ultimatum2):
    u1, u2 = L.ultimatum_game(2)
    assert np.array_equal(u1, ultimatum2["u1"]) and np.array_equal(u2, ultimatum2["u2"])
    assert L.pure_nash_equilibria(u1, u2) == ultimatum2["nash"]


def test_two_round_game_entries_by_hand():
    u1, u2 = L.two_round_game(2, 0.9)
    assert u1.shape == u2.shape == (3 * 4, 4 * 3)

    def row(y, phi):
        return y * 4 + phi                    # (G + 2) Schwellen φ je Angebot y

    def col(theta, z):
        return theta * 3 + z                # (G + 1) Gegenangebote z je Schwelle θ

    assert (u1[row(1, 0), col(1, 0)], u2[row(1, 0), col(1, 0)]) == (0.5, 0.5)                   # Runde 1: y = 1 ≥ θ = 1 wird angenommen: (2 − 1)/2 und 1/2
    assert (u1[row(0, 1), col(1, 2)], u2[row(0, 1), col(1, 2)]) == (pytest.approx(0.9), 0.0)    # abgelehnt (0 < 1), z = 2 ≥ φ = 1: δ z/G = 0,9 und δ (G − z)/G = 0
    assert (u1[row(0, 2), col(2, 1)], u2[row(0, 2), col(2, 1)]) == (0.0, 0.0)                   # abgelehnt, z = 1 < φ = 2: beide 0


def test_game_matrices_dispatch():
    assert np.array_equal(L.game_matrices("ultimatum", 2)[0], L.ultimatum_game(2)[0])
    assert L.game_matrices("zwei_runden", 2, 0.9)[0].shape == (12, 12)
    with pytest.raises(ValueError):
        L.game_matrices("skat")


def test_hedge_and_regret_matching_weights_by_hand():
    assert L.hedge_weights(np.array([0.0, 0.0]), 3.0).tolist() == [0.5, 0.5]
    assert L.hedge_weights(np.array([1.0, 0.0]), math.log(3)) == pytest.approx([0.75, 0.25])                 # 1 : 1/3
    assert L.hedge_weights(np.array([1e6, 0.0]), 1.0) == pytest.approx([1.0, 0.0])                            # kein Überlauf
    assert L.regret_matching_weights(np.array([2.0, -1.0, 1.0])) == pytest.approx([2 / 3, 0.0, 1 / 3])
    assert L.regret_matching_weights(np.array([-1.0, -2.0])).tolist() == [0.5, 0.5]


def test_draw_action_follows_the_cumulative_distribution():
    p = np.array([0.25, 0.5, 0.25])
    assert [L.draw_action(p, u) for u in (0.0, 0.1, 0.25, 0.74, 0.75, 0.9999)] == [0, 0, 1, 1, 2, 2]


def test_uniform01_is_in_the_unit_interval_and_reproducible():
    from vhd_rng import SplitMix64
    a, b = SplitMix64(7), SplitMix64(7)
    draws = [L.uniform01(a) for _ in range(1000)]
    assert draws == [L.uniform01(b) for _ in range(1000)] and all(0.0 <= d < 1.0 for d in draws) and 0.4 < sum(draws) / 1000 < 0.6


def test_nash_gap_on_the_mini_game(ultimatum2):
    u1, u2 = ultimatum2["u1"], ultimatum2["u2"]

    def pure(n, i):
        return np.eye(n)[i]

    assert L.nash_gap(u1, u2, pure(3, 0), pure(4, 0)) == 0.0                              # (0, 0) ist ein Gleichgewicht
    assert L.nash_gap(u1, u2, pure(3, 0), pure(4, 1)) == pytest.approx(0.5)               # θ = 1: Spieler 1 verbessert sich mit y = 1 von 0 auf 1/2
    assert L.nash_gap(u1, u2, pure(3, 2), pure(4, 0)) == pytest.approx(1.0)               # y = 2 gegen θ = 0: Spieler 1 bekäme mit y = 0 1 statt 0
    half = np.array([0.5, 0.5, 0.0])
    assert L.nash_gap(u1, u2, half, pure(4, 0)) == pytest.approx(0.25)                     # Spieler 1 bekommt 1/2 · 1 + 1/2 · 1/2 = 3/4 statt best 1; Spieler 2 hat keine Verbesserung


def test_spe_share_by_hand():
    assert L.spe_share("ultimatum", 10) == 1
    assert L.spe_share("zwei_runden", 10, 0.9) == F(1, 10)                                  # 1 − ⌈10 · 0,9⌉/10


def test_a_learning_run_is_reproducible_and_seed_dependent():
    u1, u2 = L.two_round_game(10, 0.9)
    r = L.run(u1, u2, "rm", 0.0, 500, 1)
    assert r == L.run(u1, u2, "rm", 0.0, 500, 1) and r != L.run(u1, u2, "rm", 0.0, 500, 2)            # Regret Matching: der Zufalls-Seed entscheidet, wohin das Spiel läuft
    assert r.path_rounds[-1] == 500 and len(r.path_rounds) == len(r.path_share1) == len(r.path_gap)
    assert abs(sum(r.p1) - 1) < 1e-12 and abs(sum(r.p2) - 1) < 1e-12
    with pytest.raises(ValueError):
        L.run(u1, u2, "sgd", 1.0, 10, 1)


def test_learning_at_the_largest_slider_is_fast_enough():
    u1, u2 = L.two_round_game(10, 0.9)
    t0 = time.time()
    r = L.run(u1, u2, "hedge", 2.0, 50000, 1)
    assert time.time() - t0 < 120 and 0.0 <= r.share1 <= 1.0


def test_two_round_game_accepts_a_counter_offer_that_exactly_meets_the_threshold():
    """Gleichstand in Runde 2: z = φ wird angenommen (z ≥ φ). G = 2, δ = 0,9: Runde 1 abgelehnt (y = 0 < θ = 1), z = 1, φ = 1: Spieler 1 bekommt 0,9 · 1/2, Spieler 2 0,9 · (2 − 1)/2."""
    u1, u2 = L.two_round_game(2, 0.9)
    row, col = 0 * 4 + 1, 1 * 3 + 1                        # y = 0, φ = 1; θ = 1, z = 1
    assert (u1[row, col], u2[row, col]) == (pytest.approx(0.45), pytest.approx(0.45))


def test_draw_action_never_leaves_the_action_range_when_the_weights_fall_short_of_one():
    """Rundungsreste: summieren die Gewichte knapp unter 1 und ist u größer als ihre Summe, bleibt die Wahl bei der letzten Aktion (kein Index außerhalb)."""
    p = np.array([0.5, 0.4999999])
    assert L.draw_action(p, 0.99999995) == 1 and L.draw_action(p, 0.0) == 0


def test_learning_path_checkpoints_are_every_nth_round_and_the_last_one():
    u1, u2 = L.ultimatum_game(3)
    even = L.run(u1, u2, "hedge", 2.0, 500, 1)                              # 500 Runden, 50 Stützstellen: alle 10 Runden
    assert even.path_rounds == tuple(range(10, 501, 10))
    odd = L.run(u1, u2, "hedge", 2.0, 105, 1)                               # alle 2 Runden, dazu die letzte Runde 105
    assert odd.path_rounds == tuple(range(2, 105, 2)) + (105,)
    short = L.run(u1, u2, "rm", 0.0, 7, 1)                                  # weniger Runden als Stützstellen: jede Runde
    assert short.path_rounds == tuple(range(1, 8))
    assert even.path_share1[-1] == pytest.approx(even.share1) and even.path_gap[-1] == pytest.approx(even.gap)
