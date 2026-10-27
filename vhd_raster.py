"""Rubinstein-Verhandlung mit Angeboten auf einem Raster {0, 1/G, …, 1}: Rückwärtsrechnen in Brüchen und stationäre Gleichgewichte.

Alles ganzzahlig bzw. in fractions.Fraction, nie in Gleitkomma: ob ein Angebot genau dem Fortsetzungswert des Antwortenden entspricht (Gleichstand), muss exakt
entschieden werden. Es gibt drei Ebenen des Gleichstands: (1) der Antwortende bei gleicher Auszahlung (Regel A: annehmen, Regel R: ablehnen), (2) der Anbieter, wenn
Annehmen und Abgelehnt-werden dieselbe Auszahlung ergäben (feste Regel P: er zieht das angenommene Angebot vor), (3) die Wahl unter gleich guten angenommenen Angeboten
(es gibt nur eines: das kleinste annehmbare, denn jedes größere Angebot kostet den Anbieter etwas)."""

import math
from fractions import Fraction


def to_fraction(x) -> Fraction:
    """Dezimalzahl der Regler als Bruch (ohne Binärrest): 0,9 wird 9/10."""
    return Fraction(str(x)) if not isinstance(x, Fraction) else x


def smallest_acceptable(cont: Fraction, G: int, rule_a: bool) -> int:
    """Kleinstes ganzzahliges k, für das das Angebot k/G angenommen wird, wenn der Antwortende abgelehnt den Wert `cont` bekäme.
    Regel A (Annahme bei Gleichstand): k/G ≥ cont, also k = ⌈G·cont⌉. Regel R (Ablehnung bei Gleichstand): k/G > cont, also k = ⌊G·cont⌋ + 1."""
    scaled = cont * G
    return math.ceil(scaled) if rule_a else math.floor(scaled) + 1


def grid_backward(d_first, d_second, T: int, G: int, rule_a: bool = True) -> list:
    """Rückwärtsrechnen im Spiel mit T Runden und Angeboten k/G. Liste von (a_t, b_t) für t = 1 … T: Wert des Anbieters und des Antwortenden der Runde t, in Termen der Runde t.

    Die Diskontfaktoren sind Brüche. Der Anbieter wählt das beste Rasterangebot: das kleinste annehmbare (Auszahlung 1 − k/G) oder, wenn das schlechter ist als die Ablehnung
    (d_Anbieter · Wert als Antwortender der nächsten Runde), ein abgelehntes Angebot. Bei gleicher Auszahlung zieht er das angenommene Angebot vor (Regel P)."""
    if T < 1 or G < 1:
        raise ValueError("T und G müssen mindestens 1 sein")
    d_first, d_second = to_fraction(d_first), to_fraction(d_second)
    zero = Fraction(0)
    a_next = b_next = zero
    values = [None] * T
    for t in range(T, 0, -1):
        proposer_is_first = t % 2 == 1
        d_prop, d_resp = (d_first, d_second) if proposer_is_first else (d_second, d_first)
        if t == T:
            cont_resp, reject_prop = zero, zero
        else:
            cont_resp = d_resp * a_next                  # der Antwortende wird in der nächsten Runde Anbieter
            reject_prop = d_prop * b_next                # der Anbieter wäre dann Antwortender
        k = smallest_acceptable(cont_resp, G, rule_a)
        if k <= G and 1 - Fraction(k, G) >= reject_prop:
            a_t, b_t = 1 - Fraction(k, G), Fraction(k, G)
        else:
            a_t, b_t = reject_prop, cont_resp
        values[t - 1] = (a_t, b_t)
        a_next, b_next = a_t, b_t
    return values


def grid_share_first(d_first, d_second, T: int, G: int, rule_a: bool = True) -> Fraction:
    """Anteil des ersten Anbieters im Rasterspiel mit T Runden."""
    return grid_backward(d_first, d_second, T, G, rule_a)[0][0]


def grid_fixed_points(d_first, d_second, G: int, rule_a: bool = True) -> list:
    """Alle stationären Gleichgewichte des Rasterspiels mit sofortiger Einigung: Paare (A₁, A₂) mit A₁ = 1 − k₁/G und A₂ = 1 − k₂/G, wobei k₁ das kleinste annehmbare Angebot
    bei dem Fortsetzungswert d₂·A₂ ist und k₂ bei d₁·A₁. Für jedes A₁ ist A₂ durch die zweite Gleichung festgelegt, deshalb O(G)."""
    d_first, d_second = to_fraction(d_first), to_fraction(d_second)
    found = []
    for i in range(G + 1):
        a1 = Fraction(i, G)
        k2 = smallest_acceptable(d_first * a1, G, rule_a)
        if k2 > G:
            continue
        a2 = 1 - Fraction(k2, G)
        k1 = smallest_acceptable(d_second * a2, G, rule_a)
        if k1 <= G and 1 - Fraction(k1, G) == a1:
            found.append((a1, a2))
    return found


def error_bound(d_first, d_second, G: int) -> Fraction:
    """Schranke aus der Messung (kein Satz): |Anteil im Rasterspiel − Formel| ≤ 1 / ((1 − d₁ d₂) · G)."""
    d_first, d_second = to_fraction(d_first), to_fraction(d_second)
    return 1 / ((1 - d_first * d_second) * G)
