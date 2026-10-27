"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Werte sind von Hand gerechnet, die Rechnung steht daneben."""

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def halves():
    """δ₁ = δ₂ = 1/2. Unendlicher Horizont: x₁ = (1 − 1/2) / (1 − 1/4) = (1/2) / (3/4) = 2/3.
    Rückwärtsrechnen: T = 1: der erste Anbieter bekommt 1. T = 2: Spieler 2 bietet in Runde 2 und bekommt alles, Spieler 1 verlangt in Runde 1 also 1/2 · 1 für Spieler 2: x₁ = 1 − 1/2 = 1/2.
    T = 3: v₃ = 1; v₂ = 1 − δ₁ v₃ = 1/2; v₁ = 1 − δ₂ v₂ = 1 − 1/4 = 3/4.
    Wert des zweiten Anbieters, wenn er Runde 2 eröffnete: (1 − δ₁)/(1 − δ₁δ₂) = (1/2)/(3/4) = 2/3."""
    return {"d1": F(1, 2), "d2": F(1, 2), "limit": F(2, 3), "T1": F(1), "T2": F(1, 2), "T3": F(3, 4), "second": F(2, 3)}


@pytest.fixture
def grid_mini():
    """Raster G = 10, δ = 9/10. T = 1 (Ultimatum): Regel A: der Antwortende nimmt jedes Angebot ≥ 0 an, der Anbieter bietet 0 und behält 1. Regel R: er lehnt bei Gleichstand ab, also braucht
    er mindestens 1/10: k = ⌊10 · 0⌋ + 1 = 1, der Anbieter behält 9/10. T = 2, Regel A: Spieler 2 bietet in Runde 2 und behält 1; Spieler 1 muss ihm in Runde 1 mindestens 9/10 · 1 bieten:
    k = ⌈10 · 9/10⌉ = 9, er behält 1/10. T = 2, Regel R: in Runde 2 muss Spieler 2 dem Spieler 1 1/10 lassen und behält 9/10; Fortsetzungswert in Runde 1: 9/10 · 9/10 = 81/100,
    k = ⌊8,1⌋ + 1 = 9, Spieler 1 behält wieder 1/10 (die Regeln unterscheiden sich hier nicht)."""
    return {"d": "9/10", "G": 10, "T1_A": F(1), "T1_R": F(9, 10), "T2_A": F(1, 10), "T2_R": F(1, 10)}


@pytest.fixture
def costs_mini():
    """Feste Kosten c₁ = 0,05, c₂ = 0,10, T = 2: A₂ = 1 (letzte Runde). A₁ = 1 − max(0, A₂ − c₂) = 1 − 0,9 = 0,10."""
    return {"c1": F(5, 100), "c2": F(10, 100), "T2": F(1, 10)}


@pytest.fixture
def outside_mini():
    """δ₁ = δ₂ = 1/2: x₂* = (1/2)(1 − 1/2) / (1 − 1/4) = (1/4) / (3/4) = 1/3 (Schwelle der Variante A). o = 1/2 liegt darüber: x₂ = 1/2, also x₁ = 1/2.
    Variante C: Schwelle x₂*/δ₂ = 2/3; o = 1/2 ergibt δ₂ o = 1/4 < 1/3: die Option zählt nicht, x₁ = 1 − 1/3 = 2/3."""
    return {"d1": F(1, 2), "d2": F(1, 2), "threshold_A": F(1, 3), "threshold_C": F(2, 3), "o": F(1, 2), "x1_A": F(1, 2), "x1_C": F(2, 3)}


@pytest.fixture
def solutions_mini():
    """u₂ = (1 − x)^(1/2) (a = 1/2, k = 1, Drohpunkt 0). Nash: x · (1 − x)^(1/2) wird maximal, wo die Ableitung (1 − x)^(1/2) − x / (2 (1 − x)^(1/2)) null ist: 2 (1 − x) = x, x = 2/3.
    Utilitaristisch: x + (1 − x)^(1/2): Ableitung 1 − 1/(2 (1 − x)^(1/2)) = 0 bei (1 − x)^(1/2) = 1/2, x = 3/4. Egalitär und Kalai–Smorodinsky (Idealpunkt (1, 1)): x = (1 − x)^(1/2),
    also x² = 1 − x, x = (√5 − 1)/2 = 0,6180339887…. Halbe-halbe: 1/2."""
    return {"a": 0.5, "k": 1.0, "d2": 0.0, "nash": 2 / 3, "utilitarian": 3 / 4, "egalitarian": (5 ** 0.5 - 1) / 2, "ks": (5 ** 0.5 - 1) / 2, "half": 0.5}


@pytest.fixture
def ultimatum2():
    """Ultimatumspiel auf dem Raster G = 2: Angebote y ∈ {0, 1, 2}/2 (Zeilen), Schwellen θ ∈ {0, 1, 2, 3}/2 (Spalten, 3 = alles ablehnen). Angenommen, wenn y ≥ θ.
    Spieler 1 bekommt (2 − y)/2, Spieler 2 y/2, sonst beide 0. Reine Nash-Gleichgewichte (Beste-Antwort-Mengen von Hand): (0, 0), (1, 1), (0, 2), (2, 2), (0, 3): fünf."""
    u1 = np.array([[1.0, 0, 0, 0], [0.5, 0.5, 0, 0], [0.0, 0.0, 0.0, 0.0]])
    u2 = np.array([[0.0, 0, 0, 0], [0.5, 0.5, 0, 0], [1.0, 1.0, 1.0, 0.0]])
    return {"u1": u1, "u2": u2, "nash": [(0, 0), (0, 2), (0, 3), (1, 1), (2, 2)]}
