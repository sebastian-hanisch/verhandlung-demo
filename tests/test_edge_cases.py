"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): T = 1 und 2, δ = 0, genau gleiche Werte, inklusive Grenzen, Reglermaxima.

Die häufigsten Lücken, die der Mutationstest in anderen Demos fand: ein Zweig `n < 2` greift bei genau zwei Elementen, eine Grenze gilt nicht inklusive, ein Gleichstand wird anders aufgelöst als dokumentiert."""

from fractions import Fraction as F

import pytest

import vhd_constants as C
import vhd_evaluation as ev
import vhd_learning as L
import vhd_outside as O
import vhd_raster as Ra
import vhd_rubinstein as R
import vhd_solutions as S
from vhd_scenario import Settings


def test_exactly_two_rounds():
    """T = 2: der kleinste Fall, in dem es etwas zu vergleichen gibt. Der zweite Anbieter bekommt in Runde 2 alles, der erste muss ihm d₂ bieten."""
    assert R.backward_T(F(1, 2), F(1, 3), 2) == [F(2, 3), F(1)]
    assert R.share_at_horizons(0.5, 1 / 3, 2) == pytest.approx([1.0, 2 / 3])
    assert R.costs_backward(F(1, 10), F(1, 5), 2) == [F(1, 5), F(1)]            # 1 − max(0, 1 − 1/5) = 1/5


def test_one_round_is_the_ultimatum_for_every_model():
    assert R.backward_T(0.3, 0.8, 1) == [1]
    assert R.costs_backward(0.3, 0.8, 1) == [1]
    assert Ra.grid_share_first("3/10", "4/5", 1, 10, True) == 1 and Ra.grid_share_first("3/10", "4/5", 1, 10, False) == F(9, 10)
    assert O.outside_backward(0.9, 0.9, 0.0, 1, "A") == [1]                                 # Option 0: der Halter lehnt nie lieber ab
    assert O.outside_backward(0.9, 0.9, 0.7, 1, "A")[0] == pytest.approx(0.3)               # sonst muss der Anbieter ihm o bieten


def test_zero_patience_gives_everything_to_the_proposer():
    """δ = 0: nichts zählt nach der ersten Runde; der Anbieter bekommt alles. Der Zweite als Antwortender: x₂* = 0."""
    assert R.rubinstein_share(0.0, 0.0) == 1.0 and R.rubinstein_share(0.0, 0.9) == pytest.approx(0.1) and R.rubinstein_share(0.9, 0.0) == 1.0
    assert R.backward_T(0.0, 0.0, 7) == [1] * 7
    assert O.holder_share_without_option(0.9, 0.0) == 0.0 and O.outside_threshold(0.9, 0.0, "C") == float("inf")


def test_equal_values_on_the_grid_follow_the_documented_rules():
    """Gleichstand: das Angebot 9/10 ist genau der Fortsetzungswert 9/10: A nimmt an, R lehnt ab und der Anbieter muss 10/10 bieten."""
    assert Ra.smallest_acceptable(F(9, 10), 10, True) == 9 and Ra.smallest_acceptable(F(9, 10), 10, False) == 10
    assert Ra.smallest_acceptable(F(1), 10, True) == 10 and Ra.smallest_acceptable(F(1), 10, False) == 11           # 11 > G: kein Angebot wird angenommen


def test_no_acceptable_offer_falls_back_to_the_rejected_branch():
    """δ₁ = 1/2, δ₂ = 1, Regel R, T = 2, G = 10. Runde 2: der Antwortende (Spieler 1) lehnt bei Gleichstand ab, also bietet Spieler 2 ihm 1/10 und behält 9/10.
    Runde 1: Spieler 2 hat den Fortsetzungswert 1 · 9/10; Regel R verlangt k = ⌊9⌋ + 1 = 10, also das ganze Angebot: Spieler 1 bekäme 1 − 1 = 0. Abgelehnt wird er als
    Antwortender der Runde 2 mit 1/2 · 1/10 = 1/20 belohnt, das ist mehr: der Anbieter stellt ein Angebot, das abgelehnt wird."""
    values = Ra.grid_backward("1/2", "1", 2, 10, False)
    assert values[0] == (F(1, 20), F(9, 10))
    assert values[1] == (F(9, 10), F(1, 10))


def test_grid_with_a_single_step():
    """G = 1: nur die Angebote 0 und 1."""
    assert Ra.grid_share_first("1/2", "1/2", 3, 1, True) in (F(0), F(1))
    assert Ra.grid_fixed_points("1/2", "1/2", 1, True)
    with pytest.raises(ValueError):
        Ra.grid_backward("1/2", "1/2", 0, 10)
    with pytest.raises(ValueError):
        Ra.grid_backward("1/2", "1/2", 3, 0)


def test_grid_bounds_are_inclusive_at_the_slider_maximum():
    s = Settings(t=C.T_MAX, g=C.G_OPTIONS[-1])
    state = ev.grid_state(s)
    assert 0 <= state["share_first"] <= 1 and state["ratio"] <= 1 and state["points"] >= 1


def test_outside_option_at_the_inclusive_limits():
    zero = O.outside_backward(0.9, 0.9, 0.0, 50, "A")[0]
    base = R.backward_T(0.9, 0.9, 50)[0]
    assert zero == pytest.approx(base)                                                      # o = 0: wie ohne Option
    top = O.outside_backward(0.9, 0.9, C.O_MAX, 50, "A")[0]
    assert top == pytest.approx(1 - C.O_MAX, abs=1e-12)                                     # o = 0,99: der Halter bekommt o
    assert O.outside_threshold(0.9, 0.9, "A") == pytest.approx(0.9 * 0.1 / 0.19)


def test_threshold_is_not_inclusive_equal_to_threshold_counts_as_no_effect():
    thr = O.holder_share_without_option(0.5, 0.5)                                          # 1/3
    assert O.outside_hand(0.5, 0.5, thr, "A") == 1 - thr                                    # o = Schwelle: dasselbe wie ohne Option
    assert O.outside_hand(0.5, 0.5, thr + 0.01, "A") < 1 - thr


def test_solutions_at_the_extremes():
    for a in (C.CURV_MIN, C.CURV_MAX):
        for k in C.SCALE_OPTIONS:
            for threat in (C.THREAT_MIN, C.THREAT_MAX):
                sol = S.all_solutions(a, k, S.scaled_threat(threat, k))
                assert all(0.0 <= x <= 1.0 for x in sol.values())
    with pytest.raises(ValueError):
        S.all_solutions(1.0, 1.0, 1.0)                                                      # Drohpunkt = Idealgewinn: keine Verhandlungsmenge


def test_learning_with_one_round_and_with_one_action():
    u1, u2 = L.ultimatum_game(1)
    r = L.run(u1, u2, "hedge", 1.0, 1, 0)
    assert r.path_rounds == (1,) and abs(sum(r.p1) - 1) < 1e-12
    r2 = L.run(u1[:1], u2[:1], "rm", 0.0, 10, 0)                                           # eine einzige Aktion für Spieler 1: bleibt bei 1
    assert r2.p1 == (1.0,)


def test_the_largest_sliders_run_through_the_analysis():
    s = Settings(d1=C.D_MAX, d2=C.D_MAX, t=C.T_MAX, g=10000)
    a = ev.analyse(s)
    assert 0 <= a.x1 <= 1 and len(a.values) == ev.ROUNDS_SHOWN
    assert ev.grid_curve(s)["grid"] and ev.costs_curve(Settings(c1=C.COST_MAX, c2=C.COST_MAX, t=C.T_MAX))["costs"]


def test_costs_above_one_have_no_discount_comparison():
    curve = ev.costs_curve(Settings(c1=1.2, c2=0.3))
    assert curve["discount"] is None and curve["costs"]


def test_swapping_the_first_bidder_swaps_the_roles_in_the_analysis():
    a = ev.analyse(Settings(d1=0.8, d2=0.95, first=1))
    b = ev.analyse(Settings(d1=0.8, d2=0.95, first=2))
    assert a.x1 == pytest.approx(R.backward_T(0.8, 0.95, 200)[0]) and b.x1 == pytest.approx(1 - R.backward_T(0.95, 0.8, 200)[0])
    assert a.other_order_x1 == pytest.approx(b.x1) and b.other_order_x1 == pytest.approx(a.x1)


def test_share_at_horizons_with_exactly_one_round_and_the_refusal_of_zero():
    """t_max = 1 ist erlaubt (inklusive Grenze): das Ultimatum, der erste Anbieter bekommt alles. t_max = 0 gibt es nicht."""
    assert R.share_at_horizons(0.7, 0.4, 1) == [1.0]
    assert R.share_at_horizons(0.7, 0.4, 2) == pytest.approx([1.0, 0.6])
    with pytest.raises(ValueError):
        R.share_at_horizons(0.7, 0.4, 0)


def test_a_negative_threat_point_is_refused_and_zero_is_allowed():
    """Der Drohpunkt liegt im Intervall [0, k): negativ ist ungültig, 0 gültig, k ungültig."""
    for solution in (S.nash, S.kalai_smorodinsky, S.egalitarian, S.utilitarian, S.max_share_player1):
        with pytest.raises(ValueError):
            solution(1.0, 1.0, -0.1)
        with pytest.raises(ValueError):
            solution(1.0, 1.0, 1.0)
        assert 0.0 <= solution(1.0, 1.0, 0.0) <= 1.0
    with pytest.raises(ValueError):
        S.nash(1.0, 0.0, 0.0)
