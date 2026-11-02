"""Lernende Verhandler: zwei No-Regret-Lernregeln (Hedge, Regret Matching) spielen wiederholt dasselbe Verhandlungsspiel gegeneinander.

Spiele (Normalform, Angebote auf dem Raster G; Anteile am Überschuss = 1):
  ultimatum    Spieler 1 bietet y ∈ {0, …, G}/G an Spieler 2; Spieler 2 wählt eine Schwelle θ ∈ {0, …, G+1}/G (G+1 = alles ablehnen) und nimmt an, wenn y ≥ θ.
  zwei_runden  Spieler 1 wählt (y, φ) = Angebot in Runde 1 und Schwelle für Runde 2; Spieler 2 wählt (θ, z) = Schwelle für Runde 1 und Gegenangebot z (Anteil für Spieler 1)
               in Runde 2. Wird in Runde 1 abgelehnt und z ≥ φ, zählt die Einigung mit dem Diskont; sonst bekommt keiner etwas.
Pro Runde ziehen beide Spieler eine Aktion aus ihrer gemischten Strategie (SplitMix64, fester Zufalls-Seed); jeder aktualisiert mit den Auszahlungen ALLER eigenen Aktionen
gegen die gezogene Aktion des Gegners. Ausgewiesen wird der LETZTE Iterierte (kein Zeitmittel). Nash-Lücke = größte Auszahlungsverbesserung durch eine reine Abweichung."""

from dataclasses import dataclass

import numpy as np

from vhd_raster import grid_share_first
from vhd_rng import SplitMix64


def ultimatum_game(G: int):
    """Auszahlungsmatrizen (Spieler 1, Spieler 2) des Ultimatumspiels; Zeilen y = 0 … G, Spalten θ = 0 … G+1."""
    y = np.arange(G + 1)[:, None]
    theta = np.arange(G + 2)[None, :]
    accepted = y >= theta
    return np.where(accepted, (G - y) / G, 0.0), np.where(accepted, y / G, 0.0)


def two_round_game(G: int, delta: float):
    """Auszahlungsmatrizen des Zwei-Runden-Spiels. Zeilen von Spieler 1: (y, φ) mit y = 0 … G, φ = 0 … G+1 (Index y·(G+2) + φ); Spalten von Spieler 2: (θ, z) mit θ = 0 … G+1, z = 0 … G (Index θ·(G+1) + z)."""
    ys, phis = np.meshgrid(np.arange(G + 1), np.arange(G + 2), indexing="ij")
    ys, phis = ys.ravel()[:, None], phis.ravel()[:, None]
    thetas, zs = np.meshgrid(np.arange(G + 2), np.arange(G + 1), indexing="ij")
    thetas, zs = thetas.ravel()[None, :], zs.ravel()[None, :]
    first_round = ys >= thetas
    second_round = zs >= phis
    u1 = np.where(first_round, (G - ys) / G, np.where(second_round, delta * zs / G, 0.0))
    u2 = np.where(first_round, ys / G, np.where(second_round, delta * (G - zs) / G, 0.0))
    return u1, u2


def game_matrices(game: str, G: int = 10, delta: float = 0.9):
    if game == "ultimatum":
        return ultimatum_game(G)
    if game == "zwei_runden":
        return two_round_game(G, delta)
    raise ValueError(f"unbekanntes Spiel {game!r}")


def pure_nash_equilibria(u1: np.ndarray, u2: np.ndarray, tol: float = 1e-12) -> list:
    """Alle reinen Nash-Gleichgewichte (i, j): keiner kann sich durch eine andere reine Aktion verbessern."""
    best1 = u1.max(axis=0)
    best2 = u2.max(axis=1)
    rows, cols = np.nonzero((u1 >= best1[None, :] - tol) & (u2 >= best2[:, None] - tol))
    return list(zip(rows.tolist(), cols.tolist()))


def spe_share(game: str, G: int = 10, delta: float = 0.9):
    """Anteil von Spieler 1 im teilspielperfekten Ergebnis der Regel A (Zustimmung bei Gleichstand), aus dem Rückwärtsrechnen des Rasterspiels: Ultimatum T = 1, zwei Runden T = 2. Bei Regel R (Ablehnung bei Gleichstand) ist im Ultimatum auch 90 % teilspielperfekt."""
    T = 1 if game == "ultimatum" else 2
    return grid_share_first(str(delta), str(delta), T, G, True)


def hedge_weights(cumulative: np.ndarray, eta: float) -> np.ndarray:
    """Hedge: Gewichte proportional zu exp(η · kumulierte Auszahlung); um das Maximum verschoben (kein Überlauf)."""
    z = np.exp(eta * (cumulative - cumulative.max()))
    return z / z.sum()


def regret_matching_weights(regrets: np.ndarray) -> np.ndarray:
    """Regret Matching: Gewichte proportional zu den positiven Regrets; ohne positiven Regret gleichverteilt."""
    positive = np.maximum(regrets, 0.0)
    total = positive.sum()
    if total <= 0:
        return np.full(len(regrets), 1.0 / len(regrets))
    return positive / total


def draw_action(p: np.ndarray, u: float) -> int:
    """Aktion aus der gemischten Strategie p mit der Zufallszahl u ∈ [0, 1): erste Aktion, deren kumulierte Wahrscheinlichkeit über u liegt."""
    return int(min(np.searchsorted(np.cumsum(p), u, side="right"), len(p) - 1))


def uniform01(rng: SplitMix64) -> float:
    """Gleichverteilte Zahl in [0, 1) aus den oberen 53 Bit des SplitMix64-Stroms."""
    return (rng.next() >> 11) / float(1 << 53)


def nash_gap(u1: np.ndarray, u2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Größte Verbesserung, die eine reine Abweichung eines Spielers gegen die Strategie des anderen bringt (0 = Nash-Gleichgewicht)."""
    v1 = p1 @ u1 @ p2
    v2 = p1 @ u2 @ p2
    return float(max((u1 @ p2).max() - v1, (p1 @ u2).max() - v2))


def expected_shares(u1: np.ndarray, u2: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple:
    """Erwartete Auszahlungen (Spieler 1, Spieler 2) unter dem Strategieprofil."""
    return float(p1 @ u1 @ p2), float(p1 @ u2 @ p2)


@dataclass(frozen=True)
class LearnResult:
    p1: tuple
    p2: tuple
    share1: float            # erwarteter Anteil von Spieler 1 unter dem Endprofil
    share2: float
    gap: float               # Nash-Lücke des Endprofils
    path_rounds: tuple       # Stützstellen (Rundenzahl)
    path_share1: tuple       # erwarteter Anteil von Spieler 1 an den Stützstellen
    path_gap: tuple          # Nash-Lücke an den Stützstellen


def run(u1: np.ndarray, u2: np.ndarray, rule: str, eta0: float, rounds: int, seed: int, checkpoints: int = 50) -> LearnResult:
    """Ein Lernlauf. rule: "hedge" (Lernrate η₀/√t) oder "rm" (Regret Matching, η₀ wird nicht benutzt)."""
    if rule not in ("hedge", "rm"):
        raise ValueError(f"unbekannte Lernregel {rule!r}")
    rng = SplitMix64(seed)
    n1, n2 = u1.shape
    c1, c2 = np.zeros(n1), np.zeros(n2)          # kumulierte Auszahlungen (Hedge) bzw. Regrets (Regret Matching)
    p1, p2 = np.full(n1, 1.0 / n1), np.full(n2, 1.0 / n2)
    every = max(1, rounds // checkpoints)
    path_rounds, path_share1, path_gap = [], [], []
    for t in range(1, rounds + 1):
        a1 = draw_action(p1, uniform01(rng))
        a2 = draw_action(p2, uniform01(rng))
        v1 = u1[:, a2]                           # Auszahlungen aller eigenen Aktionen gegen die gezogene Aktion des Gegners
        v2 = u2[a1, :]
        if rule == "hedge":
            c1 += v1
            c2 += v2
            eta = eta0 / np.sqrt(t)
            p1, p2 = hedge_weights(c1, eta), hedge_weights(c2, eta)
        else:
            c1 += v1 - v1[a1]
            c2 += v2 - v2[a2]
            p1, p2 = regret_matching_weights(c1), regret_matching_weights(c2)
        if t % every == 0 or t == rounds:
            path_rounds.append(t)
            path_share1.append(float(p1 @ u1 @ p2))
            path_gap.append(nash_gap(u1, u2, p1, p2))
    s1, s2 = expected_shares(u1, u2, p1, p2)
    return LearnResult(tuple(p1.tolist()), tuple(p2.tolist()), s1, s2, nash_gap(u1, u2, p1, p2), tuple(path_rounds), tuple(path_share1), tuple(path_gap))
