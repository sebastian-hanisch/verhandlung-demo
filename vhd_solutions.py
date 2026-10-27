"""Lösungskonzepte für die Aufteilung: Nash, Kalai–Smorodinsky, egalitär, utilitaristisch, Halbe-halbe; dazu Rubinstein bei gekrümmtem Nutzen.

Verhandlungsmenge: Spieler 1 bekommt den Anteil x des Überschusses und hat den Nutzen u₁ = x; Spieler 2 hat u₂ = k·(1 − x)^a mit dem Maßstab k > 0 und der Krümmung a
(a < 1: risikoscheu, die Menge ist konvex; a = 1: linear; a > 1: die Menge ist nicht konvex). Drohpunkt (0, d₂) mit 0 ≤ d₂ < k. Alle Lösungen sind Anteile x von Spieler 1.
Kein scipy zur Laufzeit: die Nullstellen werden durch Bisektion gefunden (die Funktionen sind auf dem Suchintervall monoton, siehe die Docstrings)."""

BISECTION_STEPS = 200


def u2(x: float, a: float, k: float) -> float:
    """Nutzen von Spieler 2 beim Anteil x von Spieler 1: k (1 − x)^a."""
    return k * (1 - x) ** a


def max_share_player1(a: float, k: float, d2: float) -> float:
    """Größter Anteil von Spieler 1, bei dem Spieler 2 noch seinen Drohpunkt erreicht (u₂ ≥ d₂): 1 − (d₂/k)^(1/a). Das ist der Idealpunkt von Spieler 1 auf der Menge der Einigungen."""
    _check(a, k, d2)
    return 1 - (d2 / k) ** (1 / a)


def _check(a: float, k: float, d2: float) -> None:
    if a <= 0 or not 0 <= d2 < k:                          # 0 ≤ d₂ < k verlangt schon k > 0
        raise ValueError("verlangt: a > 0, k > 0 und 0 ≤ d₂ < k")


def _bisect(f, lo: float, hi: float) -> float:
    """Nullstelle von f auf [lo, hi] bei einem Vorzeichenwechsel f(lo) ≤ 0 ≤ f(hi) (wachsendes f) bzw. f(lo) ≥ 0 ≥ f(hi) (fallendes f)."""
    rising = f(hi) >= f(lo)
    for _ in range(BISECTION_STEPS):
        mid = (lo + hi) / 2
        if (f(mid) < 0) == rising:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def nash(a: float, k: float, d2: float) -> float:
    """Nash-Lösung: maximiert x · (u₂ − d₂). Die Ableitung g(x) = k (1 − x)^(a−1) (1 − (1 + a) x) − d₂ fällt auf [0, 1/(1 + a)] von k − d₂ > 0 auf −d₂ ≤ 0;
    ihre Nullstelle ist das Maximum (bei d₂ = 0 ist es 1/(1 + a), die Bisektion trifft es auf 15 Stellen); rechts davon ist g negativ."""
    _check(a, k, d2)

    def derivative(x: float) -> float:
        return k * (1 - x) ** (a - 1) * (1 - (1 + a) * x) - d2

    return _bisect(derivative, 0.0, 1 / (1 + a))


def kalai_smorodinsky(a: float, k: float, d2: float) -> float:
    """Kalai–Smorodinsky: der Punkt auf dem Rand der Menge, an dem die Gewinne im Verhältnis der Idealgewinne stehen: x / x_max = (u₂ − d₂) / (k − d₂).
    Idealpunkt: (größter Nutzen von Spieler 1 auf der Menge der Einigungen, k). Die Differenz h(x) = x/x_max − (u₂ − d₂)/(k − d₂) wächst von −1 auf +1."""
    _check(a, k, d2)
    x_max = max_share_player1(a, k, d2)

    def gap(x: float) -> float:
        return x / x_max - (u2(x, a, k) - d2) / (k - d2)

    return _bisect(gap, 0.0, x_max)


def egalitarian(a: float, k: float, d2: float) -> float:
    """Egalitär (Kalai 1977): gleiche Gewinne, x = u₂ − d₂. Die Differenz x − (u₂ − d₂) wächst von −(k − d₂) < 0 auf x_max > 0."""
    _check(a, k, d2)

    def gap(x: float) -> float:
        return x - (u2(x, a, k) - d2)

    return _bisect(gap, 0.0, max_share_player1(a, k, d2))


def utilitarian(a: float, k: float, d2: float) -> float:
    """Utilitaristisch: maximiert die Summe u₁ + u₂ auf [0, x_max]. Bei a < 1 ist die Summe streng konkav (innerer Punkt 1 − (a k)^(1/(1 − a)), auf das Intervall begrenzt);
    bei a ≥ 1 ist sie konvex bzw. linear, das Maximum liegt an einem Rand. Bei Gleichstand gilt der kleinste Anteil von Spieler 1 (z. B. a = 1 und k = 1: jede Aufteilung gleich gut)."""
    _check(a, k, d2)
    x_max = max_share_player1(a, k, d2)
    if a < 1:
        inner = 1 - (a * k) ** (1 / (1 - a))
        return min(max(inner, 0.0), x_max)

    def total(x: float) -> float:
        return x + u2(x, a, k)

    return 0.0 if total(0.0) >= total(x_max) else x_max


def half_half() -> float:
    """Halbe-halbe; bei zwei Spielern auch der Shapley-Wert des Überschusses. Kennt weder Geduld noch Nutzenform."""
    return 0.5


def all_solutions(a: float, k: float, d2: float) -> dict:
    """Alle Lösungskonzepte für dieselbe Verhandlungsmenge: Name -> Anteil von Spieler 1."""
    return {
        "nash": nash(a, k, d2),
        "ks": kalai_smorodinsky(a, k, d2),
        "egalitarian": egalitarian(a, k, d2),
        "utilitarian": utilitarian(a, k, d2),
        "half": half_half(),
    }


def scaled_threat(threat: float, k: float) -> float:
    """Drohpunkt, der mit dem Maßstab mitskaliert: d₂ = threat · k (damit die Verhandlungsmenge dieselbe Form behält, nur Spieler 2 misst in anderen Einheiten)."""
    return threat * k


def rubinstein_utility(d1: float, d2: float, a: float) -> float:
    """Rubinstein-Anteil des ersten Anbieters, wenn Spieler 2 den Nutzen y^a seines Anteils y hat (Spieler 1 linear) und beide mit d diskontieren, unendlicher Horizont:
    Spieler 2 ist bei einem Angebot y genau dann indifferent, wenn y^a = d₂ · (Anteil als nächster Anbieter)^a, also wirkt sein Diskont als d₂^(1/a)."""
    d2_eff = d2 ** (1 / a)
    return (1 - d2_eff) / (1 - d1 * d2_eff)
