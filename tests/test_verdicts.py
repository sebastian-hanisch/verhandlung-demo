"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jeder Zweig der Art steht hier. Ein Verdikt ist eine reine Funktion (Art, Text).

Preset-Texte der App (§11a Punkt 7): je Preset die erwartete Verdikt-Zeile; ein Preset ohne Effekt meldet keinen Effekt."""

import pytest

import vhd_constants as C
import vhd_evaluation as ev
from vhd_scenario import Settings


def _s(**kw):
    return Settings(**{**C.PRESETS["Gleiche Geduld (Voreinstellung)"], **kw})


def test_main_verdict_names_both_orders_and_the_formula():
    kind, text = ev.verdict_main(ev.analyse(_s()))
    assert kind == "info"
    assert "Der Verlader bekommt 52,6 %" in text and "der Spediteur 47,4 %" in text and "T = 200 Runden" in text and "Halbe-halbe wäre 50,0 %" in text
    assert "gibt dem Verlader 52,6 %" in text and "Bietet der Verlader zuerst, bekommt er 52,6 %" in text and "bietet der Spediteur zuerst, 47,4 %" in text
    assert "+5,3 Prozentpunkte" in text                                                    # 52,63 − 47,37


def test_main_verdict_when_the_spediteur_bids_first():
    text = ev.verdict_main(ev.analyse(_s(first=2)))[1]
    assert "Spediteur bietet zuerst, T = 200" in text and "Der Verlader bekommt 47,4 %" in text and "Bietet der Verlader zuerst, bekommt er 52,6 %" in text


def test_main_verdict_for_the_ultimatum_ignores_patience():
    text = ev.verdict_main(ev.analyse(_s(t=1, d1=0.3, d2=0.99)))[1]
    assert text.startswith("Eine Runde (Ultimatum)") and "bekommt Verlader 100,0 %" in text and "Die Geduld spielt keine Rolle" in text
    text2 = ev.verdict_main(ev.analyse(_s(t=1, first=2)))[1]
    assert "bekommt Spediteur 100,0 %" in text2 and "der Verlader also 0,0 %" in text2


def test_patience_verdict_numbers_and_the_edge_of_the_slider():
    text = ev.verdict_patience(_s())[1]
    assert "δ₂ = 0,90" in text and "δ₁ = 0,90 gibt dem Verlader 52,6 %" in text and "eine Stufe mehr (δ₁ = 0,91) 55,2 %" in text and "+2,62 Prozentpunkte" in text and "bei 2,49" in text
    top = ev.verdict_patience(_s(d1=C.D_MAX))[1]
    assert "eine Stufe weniger (δ₁ = 0,98)" in top and "-" in top.split("also")[1][:6]


def test_horizon_verdict_parity_and_ultimatum():
    even = ev.verdict_horizon(_s(t=10))[1]
    assert "T = 10 (gerade)" in even and "34,28 %" in even and "unter dem Grenzwert 52,63 %" in even and "δ₁δ₂ = 0,810" in even
    odd = ev.verdict_horizon(_s(t=11))[1]
    assert "T = 11 (ungerade)" in odd and "69,15 %" in odd and "über dem Grenzwert" in odd
    assert ev.verdict_horizon(_s(t=1))[1].startswith("T = 1 ist das Ultimatum: Verlader bekommt 100,0 %")
    exact = ev.verdict_horizon(_s(t=2, d1=0.0, d2=0.0))[1]
    assert "auf dem Grenzwert" in exact and "Abstand 0 Prozentpunkte" in exact


def test_short_rounds_verdict_numbers():
    text = ev.verdict_short_rounds(_s(dt=1 / 24))[1]
    assert "Runden von 1 Stunde" in text and "r₁ = 1,0, r₂ = 2,0" in text and "bekommt er 68,05 %" in text and "r₂/(r₁ + r₂) = 66,67 %" in text and "50,0 %" in text
    day = ev.verdict_short_rounds(_s(dt=1.0))[1]
    assert "Runden von 1 Tag" in day and "91,00 %" in day and "33,48 %" in day and "+57,52 Prozentpunkte" in day


def test_costs_verdict_three_cases_and_the_comparison():
    smaller = ev.verdict_costs(_s(c1=0.05, c2=0.10))[1]
    assert "der erste Anbieter bekommt 100,0 %" in smaller and "Verlader hat die kleineren Kosten (0,05 < 0,10)" in smaller and "als zweiter Anbieter nur 5,0 %" in smaller and "bekäme der erste Anbieter 69,0 %" in smaller
    bigger = ev.verdict_costs(_s(c1=0.10, c2=0.05))[1]
    assert "Verlader hat die größeren Kosten (0,10 > 0,05)" in bigger and "bekommt er nur 5,0 %" in bigger and "bekäme der erste Anbieter 34,5 %" in bigger
    even = ev.verdict_costs(_s(c1=0.10, c2=0.10, t=200))[1]
    assert "Gleiche Kosten" in even and "ein gerades T gibt dem ersten Anbieter nur seine Kosten c" in even
    assert "nur seine Kosten c" in even and "bekommt 10,0 %" in even
    odd = ev.verdict_costs(_s(c1=0.10, c2=0.10, t=201))[1]
    assert "ein ungerades T gibt dem ersten Anbieter den ganzen" in odd and "den ganzen Überschuss" in odd and "bekommt 100,0 %" in odd
    high = ev.verdict_costs(_s(c1=1.2, c2=0.3))[1]
    assert "Diskont" not in high


def test_costs_verdict_when_the_spediteur_bids_first():
    text = ev.verdict_costs(_s(c1=0.05, c2=0.10, first=2))[1]
    assert "Spediteur bietet zuerst" in text and "Spediteur hat die größeren Kosten (0,10 > 0,05)" in text and "der Verlader 95,0 %" in text


def test_outside_verdict_with_and_without_effect():
    no = ev.verdict_outside(_s(o=0.4))[1]
    assert no.startswith("Die Option ändert nichts") and "o = 0,400 liegt nicht über dem Anteil 47,4 %" in no and "Schwelle für o: 0,474" in no and "Nash-Lösung mit Drohpunkt o gäbe dem Spediteur dagegen 70,0 %" in no
    yes = ev.verdict_outside(_s(o=0.6))[1]
    assert yes.startswith("Die Option zählt: der Spediteur bekommt 60,0 % statt 47,4 % ohne Option") and "Variante A: Erlös o = 0,600" in yes and "gäbe dem Spediteur 80,0 %" in yes
    c = ev.verdict_outside(_s(o=0.6, variant="C"))[1]
    assert "Variante C: Erlös δ·o = 0,540" in c and "54,0 %" in c
    swapped = ev.verdict_outside(_s(o=0.6, first=2))[1]
    assert "der Verlader bekommt 60,0 %" in swapped


def test_solutions_verdict_names_all_five_and_compares_with_the_other_scale():
    text = ev.verdict_solutions(_s(curv=0.5, scale=1.0, threat=0.1))[1]
    assert "a = 0,50" in text and "k = 1,0" in text and "Drohpunkt 0,100" in text and "Nash 62,6 %" in text and "Kalai–Smorodinsky 59,2 %" in text and "egalitär 56,2 %" in text and "utilitaristisch 75,0 %" in text
    assert "Beim Maßstab k = 10,0 statt 1,0: Nash 62,6 %" in text and "egalitär 96,2 %" in text
    other = ev.verdict_solutions(_s(curv=0.5, scale=10.0, threat=0.1))[1]
    assert "Beim Maßstab k = 1,0 statt 10,0" in other and "Drohpunkt 1,000" in other


def test_grid_verdict_and_the_bound():
    text = ev.verdict_grid(_s(g=10))[1]
    assert "Raster G = 10, Regel A, T = 200" in text and "bekommt 10,00 %" in text and "stetigen Spiel 52,63 %" in text and "9 (Anteil des ersten Anbieters von 10,0 % bis 90,0 %)" in text and "wird eingehalten" in text
    assert "Mit der anderen Gleichstandsregel: 10,00 %" in text
    ultimatum = ev.verdict_grid(_s(g=10, t=1, rule="R"))[1]
    assert "bekommt 90,00 %" in ultimatum and "Mit der anderen Gleichstandsregel: 100,00 %" in ultimatum
    violated = ev.verdict_grid(_s(g=10000, d1=0.99, d2=0.99, t=400))[1]
    assert "nicht eingehalten" in violated                                                     # T noch nicht konvergiert (Ausnahmen der Messreihe)


def test_learning_verdict_reports_gap_and_distance_to_the_subgame_perfect_outcome():
    run = ev.learn_once("ultimatum", "hedge", 2.0, 3000, 1)
    text = ev.verdict_learning(run)[1]
    assert "Hedge (Entropie-Gewichte), Ultimatumspiel (eine Runde), 3000 Runden, Zufalls-Seed 1" in text and "80,0 %" in text and "20,0 %" in text and "höchstens 0,05" in text and "100,0 %" in text and "-20,0 Prozentpunkte" in text
    from types import SimpleNamespace
    far = {**run, "result": SimpleNamespace(share1=0.5, share2=0.5, gap=0.2)}
    assert "über 0,05" in ev.verdict_learning(far)[1] and "Nash-Lücke 0,200" in ev.verdict_learning(far)[1]


GRID = ev.load_grid()


@pytest.mark.skipif(GRID is None, reason="data/negotiation_grid.json fehlt")
def test_experiment_verdicts_quote_the_measured_rows():
    e1 = ev.verdict_e1(GRID["settle"])[1]
    assert "(0,90; 0,90): 126 gemessen, 131 vorhergesagt" in e1 and "(0,99; 0,99): 1307 gemessen, 1375 vorhergesagt" in e1 and "höchstens um 5,2 %" in e1 and "höchstens T = 2000" in e1
    e2 = ev.verdict_e2(GRID["bound3"], GRID["bound_more_a"], GRID["bound_more_b"], GRID["rule_gap"])[1]
    assert "60 von 60 Zellen" in e2 and "196 von 200" in e2 and "die 4 Ausnahmen" in e2 and "größtes Verhältnis 0,82" in e2 and "allen 12 Zellen (4 Paare, G = 10, 100, 1000) genau 1/G" in e2
    e3 = ev.verdict_e3(GRID["rounds"])[1]
    assert "(1,0; 2,0): 0,333328 gegen 0,333333" in e3 and "(0,5; 3,0): 0,214277 gegen 0,214286" in e3 and "von 0,0500 bei Δ = 0,1 auf 0,000500 bei Δ = 0,001" in e3
    e4 = ev.verdict_e4(GRID["costs"])[1]
    assert "In 8 von 8 Fällen" in e4 and "T = 1000 dem ersten Anbieter 20,0 % und T = 1001 100,0 %" in e4 and "(0,05; 0,10) 69,0 % statt 100 %" in e4
    e5 = ev.verdict_e5(GRID["outside_scan"], GRID["outside_check"], GRID["nash_vs_exit"])[1]
    assert "in 0 von 2400 Zellen" in e5 and "größte Abweichung 2,2·10⁻⁵" in e5 and "60,0 % gegen 50,0 %" in e5
    e6 = ev.verdict_e6(GRID["scale"])[1]
    assert "Nash (45,0 %) und Kalai–Smorodinsky (45,0 %)" in e6 and "von 8,2 % (k = 0,1) über 45,0 % (k = 1) auf 81,8 % (k = 10)" in e6
    e7 = ev.verdict_e7(GRID["learn"])[1]
    assert "32 Läufe (8 Zellen mal 4 Zufalls-Seeds" in e7 and "0 enden im teilspielperfekten Anteil" in e7 and "32 haben eine Nash-Lücke von höchstens 0,05" in e7


def test_e4_counts_agreements_not_assumes_them():
    rows = [{"c_first": 0.05, "c_second": 0.10, "pred_first": 1, "pred_second": 0.05, "t1000_first": 1, "t1001_first": 1, "t1000_second": 0.05, "t1001_second": 0.05, "discount": 0.5},
            {"c_first": 0.10, "c_second": 0.05, "pred_first": 0.05, "pred_second": 1, "t1000_first": 0.5, "t1001_first": 0.05, "t1000_second": 1, "t1001_second": 1, "discount": 0.4},
            {"c_first": 0.2, "c_second": 0.2, "pred_first": None, "pred_second": None, "t1000_first": 0.2, "t1001_first": 1, "t1000_second": 1, "t1001_second": 0.2, "discount": 0.5}]
    assert "In 1 von 2 Fällen" in ev.verdict_e4(rows)[1]


def test_e7_counts_hits_and_far_gaps():
    cells = [{"shares": [0.1, 0.5], "gaps": [0.0, 0.2], "spe": 0.1, "seeds": [1, 2], "rounds": 100}]
    text = ev.verdict_e7(cells)[1]
    assert "2 Läufe (1 Zellen mal 2 Zufalls-Seeds, 100 Runden" in text and "1 enden im teilspielperfekten Anteil" in text and "1 haben eine Nash-Lücke von höchstens 0,05" in text


def test_e1_prediction_deviation_is_taken_from_the_rows():
    rows = [{"d1": 0.9, "d2": 0.9, "settled": 100, "predicted": 110.0}, {"d1": 0.5, "d2": 0.5, "settled": 10, "predicted": 9.0}]
    text = ev.verdict_e1(rows)[1]
    assert "100 gemessen, 110 vorhergesagt" in text and "höchstens um 10,0 %" in text
