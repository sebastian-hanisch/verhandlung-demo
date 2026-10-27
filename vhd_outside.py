"""Außenoption: der Antwortende der ersten Runde (der zweite Anbieter, „Halter“) kann aussteigen und bekommt o.

Varianten (immer nennen):
  A  nach einer Ablehnung steigt der Halter aus statt ein Gegenangebot zu machen; Erlös o ohne weitere Verzögerung.
  C  wie A, aber der Erlös kommt eine Runde später (zählt d_Halter · o).
  B  wie A, und der Halter kann zusätzlich in seiner eigenen Anbieterrunde aussteigen statt zu bieten (Erlös o). Wird in der App NICHT angeboten:
     gemessen ist B = A in 2 400 von 2 400 Zellen (tests/test_claims.py, tools/make_grid.py), der Zweig ändert nichts.
Anteile sind Anteile des ersten Anbieters am Überschuss (Überschuss = 1); der Halter bekommt 1 minus diesen Anteil."""

OFFERED_VARIANTS = ("A", "C")
ALL_VARIANTS = ("A", "B", "C")


def exit_payoff(o, d_holder, variant: str):
    """Wert des Ausstiegs für den Halter, in Termen der Runde, in der er abgelehnt hat: A und B o, C d·o."""
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unbekannte Variante {variant!r}")
    return d_holder * o if variant == "C" else o


def outside_backward(d_first, d_second, o, T: int, variant: str = "A") -> list:
    """Rückwärtsrechnen im Spiel mit T Runden (stetige Angebote, Annahme bei Gleichstand): Anteil des Anbieters der Runde t für t = 1 … T (Index 0 = Runde 1).

    Ungerade Runden: der erste Anbieter bietet, der Halter antwortet und lehnt nur ab, wenn max(Ausstieg, d₂ · Wert als Anbieter der nächsten Runde) mehr ist als das Angebot.
    Gerade Runden: der Halter bietet, der erste Anbieter antwortet (Variante B: der Halter kann stattdessen aussteigen und o nehmen)."""
    if T < 1:
        raise ValueError("T muss mindestens 1 sein")
    ex = exit_payoff(o, d_second, variant)
    values = [None] * T
    nxt = 0
    for t in range(T, 0, -1):
        if t % 2 == 1:
            cont_holder = max(ex, d_second * nxt)
            values[t - 1] = 1 - cont_holder
        else:
            value = 1 - d_first * nxt
            if variant == "B":
                value = max(value, o)
            values[t - 1] = value
        nxt = values[t - 1]
    return values


def outside_share_first(d_first, d_second, o, T: int, variant: str = "A"):
    """Anteil des ersten Anbieters im Spiel mit T Runden und Außenoption des Halters."""
    return outside_backward(d_first, d_second, o, T, variant)[0]


def holder_share_without_option(d_first, d_second):
    """Rubinstein-Anteil des Halters (zweiter Anbieter) ohne Option: d₂ (1 − d₁) / (1 − d₁ d₂)."""
    return d_second * (1 - d_first) / (1 - d_first * d_second)


def outside_threshold(d_first, d_second, variant: str = "A"):
    """Schwelle für den Wert o, ab dem die Option zählt: A und B x₂* = d₂ (1 − d₁)/(1 − d₁ d₂); C x₂*/d₂ = (1 − d₁)/(1 − d₁ d₂) (bei d₂ = 0 gibt es keine)."""
    x2 = holder_share_without_option(d_first, d_second)
    if variant == "C":
        return x2 / d_second if d_second else float("inf")
    return x2


def outside_hand(d_first, d_second, o, variant: str = "A"):
    """Handformel (unendlicher Horizont) für den Anteil des ersten Anbieters: 1 − max(Ausstieg, x₂*), mit Ausstieg o (A, B) bzw. d₂·o (C)."""
    return 1 - max(exit_payoff(o, d_second, variant), holder_share_without_option(d_first, d_second))


def nash_with_threat_holder_share(o):
    """Nash-Lösung mit dem Drohpunkt (0, o), Nutzen linear: der Halter bekommt o + (1 − o)/2 = (1 + o)/2. Eine ANDERE Lesart von „Außenoption“ als das Ausstiegsprinzip (A)."""
    return (1 + o) / 2
