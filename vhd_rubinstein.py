"""Rubinstein-Verhandlung mit abwechselnden Angeboten: Formel, Rückwärtsrechnen im Spiel mit T Runden, feste Kosten, kurze Runden.

Alle Anteile sind Anteile am Überschuss der Tour (Überschuss = 1). Runde 1 bietet der erste Anbieter, Runde 2 der zweite usw.; wer antwortet, nimmt an oder lehnt ab.
`d_first`, `d_second` sind die Diskontfaktoren (Geduld) des ersten und des zweiten Anbieters: eine Einigung in Runde t zählt mit d^(t−1). Die Funktionen rechnen mit
Gleitkomma- oder mit Bruchzahlen (fractions.Fraction), je nach den übergebenen Zahlen; die Tests rechnen exakt in Brüchen.
"""

import math


def rubinstein_share(d_first, d_second):
    """Anteil des ersten Anbieters im unendlichen Spiel (teilspielperfekt, stationär): (1 − d₂) / (1 − d₁ d₂)."""
    return (1 - d_second) / (1 - d_first * d_second)


def second_bidder_share(d_first, d_second):
    """Anteil des zweiten Anbieters, wenn er in Runde 2 anböte (der Wert des Anbieters, wenn die Verhandlung eine Runde später begänne): (1 − d₁) / (1 − d₁ d₂)."""
    return (1 - d_first) / (1 - d_first * d_second)


def first_mover_edge(d1, d2):
    """Vorteil des Zuerstbietens für Spieler 1: sein Anteil, wenn er zuerst bietet, minus sein Anteil, wenn Spieler 2 zuerst bietet (unendlicher Horizont).
    Bei gleicher Geduld d ist der Anteil des ersten Anbieters 1/(1 + d), der Vorteil also (1 − d)/(1 + d)."""
    return rubinstein_share(d1, d2) - (1 - rubinstein_share(d2, d1))


def backward_T(d_first, d_second, T: int) -> list:
    """Rückwärtsrechnen im Spiel mit T Runden (stetige Angebote, Annahme bei Gleichstand): Anteil des Anbieters der Runde t für t = 1 … T (Liste, Index 0 = Runde 1).

    v_T = 1 (in der letzten Runde nimmt der Antwortende alles an); v_t = 1 − d_Antwortende · v_(t+1). In ungeraden Runden antwortet der zweite Anbieter."""
    if T < 1:
        raise ValueError("T muss mindestens 1 sein")
    values = [None] * T
    values[T - 1] = d_first * 0 + 1                       # 1 im Zahlentyp der Eingabe (ein Bruch bleibt ein Bruch)
    for t in range(T - 1, 0, -1):                         # t = Runde (1-basiert)
        responder = d_second if t % 2 == 1 else d_first
        values[t - 1] = 1 - responder * values[t]
    return values


def share_at_horizons(d_first: float, d_second: float, t_max: int) -> list:
    """Anteil des ersten Anbieters für jeden Horizont T = 1 … t_max in einem Durchgang O(t_max).

    Mit r Restrunden bietet in der ersten dieser Runden der erste Anbieter genau dann, wenn r und T dieselbe Parität haben. Der Wert hängt also nur an der Parität von T:
    zwei verschränkte Folgen w(1) = 1, w(r) = 1 − d_Antwortende · w(r − 1), die Folge des Horizonts T liefert bei r = T den gesuchten Wert."""
    if t_max < 1:
        raise ValueError("t_max muss mindestens 1 sein")
    w_odd = 1.0                  # Horizont ungerade: bei r Restrunden bietet der erste Anbieter, wenn r ungerade ist
    w_even = 1.0                 # Horizont gerade: bei r Restrunden bietet der erste Anbieter, wenn r gerade ist
    out = []
    for r in range(1, t_max + 1):
        if r > 1:
            w_odd = 1 - (d_second if r % 2 == 1 else d_first) * w_odd
            w_even = 1 - (d_second if r % 2 == 0 else d_first) * w_even
        out.append(w_odd if r % 2 == 1 else w_even)
    return out


def share_of_player1(d1, d2, T: int, first: int = 1):
    """Anteil des Verladers (Spieler 1) bei Horizont T, wenn Spieler `first` zuerst bietet."""
    if first == 1:
        return backward_T(d1, d2, T)[0]
    return 1 - backward_T(d2, d1, T)[0]


def error_bound_rounds(d_first: float, d_second: float, tol: float) -> float:
    """Vorhergesagte Rundenzahl, ab der der Fehler gegen die Formel unter tol liegt: (d₁ d₂)^(T/2) < tol, also T > 2 ln(tol) / ln(d₁ d₂). Muster aus der Messung, kein Beweis."""
    return 2 * math.log(tol) / math.log(d_first * d_second)


def settling_horizon(shares: list, limit: float, tol: float):
    """Kleinstes T, ab dem der Fehler |Anteil(T') − Formel| für ALLE T' ≥ T (bis zum Ende der Liste) unter tol bleibt; None, wenn selbst das letzte T noch zu weit weg ist."""
    settled = None
    for T, value in enumerate(shares, start=1):
        if abs(value - limit) < tol:
            if settled is None:
                settled = T
        else:
            settled = None
    return settled


def costs_backward(c_first, c_second, T: int) -> list:
    """Feste Kosten je Runde: Auszahlung x − c·(Runde − 1). Anteil des Anbieters der Runde t (Liste, Index 0 = Runde 1).

    v_T = 1; v_t = 1 − max(0, v_(t+1) − c_Antwortende): der Antwortende verlangt, was er als Anbieter der nächsten Runde nach Abzug seiner Wartekosten bekäme, mindestens 0."""
    if T < 1:
        raise ValueError("T muss mindestens 1 sein")
    values = [None] * T
    values[T - 1] = c_first * 0 + 1
    for t in range(T - 1, 0, -1):
        responder = c_second if t % 2 == 1 else c_first
        values[t - 1] = 1 - max(0, values[t] - responder)
    return values


def costs_prediction(c_first, c_second):
    """Handformel für den unendlichen Horizont mit festen Kosten (Anteil des ersten, Anteil des zweiten Anbieters als Anbieter): c₁ < c₂: (1, c₁); c₁ > c₂: (c₂, 1); gleich: None (nicht eindeutig)."""
    if c_first < c_second:
        return 1, c_first
    if c_first > c_second:
        return c_second, 1
    return None


def discount_from_costs(c):
    """Diskontfaktor mit derselben Verlustrate in Runde 1 wie die festen Kosten c: d = 1 − c (nur für den Vergleich)."""
    return 1 - c


def delta_from_rate(rate: float, dt: float) -> float:
    """Diskontfaktor einer Runde der Länge dt bei der Verlustrate rate je Zeiteinheit: exp(−rate · dt)."""
    return math.exp(-rate * dt)


def share_first_continuous_time(r_first: float, r_second: float, dt: float) -> float:
    """Anteil des ersten Anbieters im unendlichen Spiel bei Runden der Länge dt: (1 − e^(−r₂ dt)) / (1 − e^(−(r₁+r₂) dt)); numerisch stabil über expm1."""
    return math.expm1(-r_second * dt) / math.expm1(-(r_first + r_second) * dt)


def limit_share_short_rounds(r_first: float, r_second: float) -> float:
    """Grenzwert für Rundenlänge → 0: r₂ / (r₁ + r₂); das ist die Nash-Lösung mit den Gewichten 1/r₁ und 1/r₂."""
    return r_second / (r_first + r_second)


def short_round_rate(r_first: float, r_second: float) -> float:
    """Steigung des Abstands zum Grenzwert in der Rundenlänge (Handentwicklung): x₁ − Grenzwert ≈ dt · r₁ r₂ / (2 (r₁ + r₂))."""
    return r_first * r_second / (2 * (r_first + r_second))
