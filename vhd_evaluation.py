"""Auswertung: Kennzahlen je Reiter, vorgerechnete Messreihen (data/negotiation_grid.json), Verdikte als reine Funktionen.

Anteile sind Anteile am Überschuss der Tour (Überschuss = 1); „Verlader“ ist Spieler 1, „Spediteur“ Spieler 2. Alle Verdikte sind `verdict_*(...) -> (Art, Text)`; die App ruft
`getattr(st, Art)(Text)`. Jede Zahl im Text wird aus der Rechnung gebildet oder steht in tests/test_claims.py (DEMO-PLAYBOOK §1d)."""

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import vhd_constants as C
import vhd_learning as L
import vhd_outside as O
import vhd_raster as Ra
import vhd_rubinstein as R
import vhd_solutions as S
from vhd_scenario import (  # noqa: F401  (Settings wird hier wieder ausgegeben)
    Settings,
    holder,
    ordered_costs,
    ordered_patience,
    other,
    to_player1,
)

GRID_PATH = Path(__file__).resolve().parent / "data" / "negotiation_grid.json"
ROUNDS_SHOWN = 12            # so viele Runden zeigt die Rückwärtstabelle der Hauptansicht
HORIZON_PLOT = 60            # Horizonte T = 1 … 60 in den Verlaufsbildern
NASH_GAP_LIMIT = 0.05        # ein Endprofil mit Nash-Lücke bis zu dieser Größe gilt in den Texten als „nahe an einem Nash-Gleichgewicht“ (Auszahlungen liegen in [0, 1])


# --- Formatierung (deutsches Dezimalkomma) ---------------------------------------------------------------------------------------------------------

def de(value: float, digits: int = 3) -> str:
    return f"{round(value, digits) + 0.0:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    return f"{round(100 * value, digits) + 0.0:.{digits}f}".replace(".", ",") + " %"


def pp(value: float, digits: int = 1) -> str:
    """Prozentpunkte mit Vorzeichen."""
    return f"{round(100 * value, digits) + 0.0:+.{digits}f}".replace(".", ",") + " Prozentpunkte"


def name_of(player: int) -> str:
    return C.NAMES[player]


# --- Hauptansicht: der Ablauf mit den Einstellungen der Seitenleiste -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Analysis:
    settings: Settings
    values: tuple            # Wert des Anbieters der Runde t (t = 1 … min(T, ROUNDS_SHOWN)), in Termen der Runde t
    share_first: float       # Anteil des ersten Anbieters bei Horizont T
    limit_first: float       # Rubinstein-Formel (unendlicher Horizont)
    x1: float                # Anteil des Verladers bei Horizont T
    x2: float
    limit_x1: float          # Formel, Sicht des Verladers
    other_order_x1: float    # Anteil des Verladers, wenn der andere zuerst bietet (Horizont T)
    other_order_limit_x1: float
    error: float             # |Anteil bei Horizont T − Formel| (erster Anbieter)


def analyse(s: Settings) -> Analysis:
    d_first, d_second = ordered_patience(s)
    values = R.backward_T(d_first, d_second, s.t)
    share_first = values[0]
    limit = R.rubinstein_share(d_first, d_second)
    other_first = 3 - s.first
    other_share = R.backward_T(d_second, d_first, s.t)[0]
    other_limit = R.rubinstein_share(d_second, d_first)
    x1 = to_player1(s.first, share_first)
    return Analysis(
        settings=s, values=tuple(values[:ROUNDS_SHOWN]), share_first=share_first, limit_first=limit, x1=x1, x2=1 - x1, limit_x1=to_player1(s.first, limit),
        other_order_x1=to_player1(other_first, other_share), other_order_limit_x1=to_player1(other_first, other_limit), error=abs(share_first - limit),
    )


# --- Reiter 1: Geduld ist Macht ---------------------------------------------------------------------------------------------------------------------

def patience_curve(d2: float, steps: int = 100) -> dict:
    """Anteil des Verladers über seiner Geduld δ₁ = 0 … 0,99, wenn er zuerst bietet bzw. wenn der Spediteur zuerst bietet (unendlicher Horizont)."""
    d1s = [i / steps for i in range(steps)]
    return {
        "d1": d1s,
        "verlader_first": [R.rubinstein_share(d1, d2) for d1 in d1s],
        "spediteur_first": [1 - R.rubinstein_share(d2, d1) for d1 in d1s],
        "half": [0.5] * steps,
    }


def patience_slope(d1: float, d2: float) -> float:
    """Ableitung des Anteils des ersten Anbieters nach seiner Geduld: (1 − d₂) d₂ / (1 − d₁ d₂)²; sie wächst, je geduldiger er schon ist."""
    return (1 - d2) * d2 / (1 - d1 * d2) ** 2


def patience_rows(d2: float, d1s=(0.0, 0.5, 0.8, 0.9, 0.99)) -> list:
    """Anteil des Verladers bei verschiedener Geduld δ₁ (Verlader bietet zuerst bzw. Spediteur bietet zuerst) und die Steigung nach δ₁; Spediteur mit Geduld d2, unendlicher Horizont."""
    return [{"d1": d1, "first": R.rubinstein_share(d1, d2), "second": 1 - R.rubinstein_share(d2, d1), "slope": patience_slope(d1, d2)} for d1 in d1s]


def patience_step(s: Settings, step: float = 0.01) -> tuple:
    """(d1_alt, x_alt, d1_neu, x_neu): der Verlader bei seiner Geduld und eine Stufe mehr (bei δ₁ am Rand eine Stufe weniger, dann ist `neu` der kleinere Wert); Verlader bietet zuerst, unendlicher Horizont."""
    up = s.d1 + step <= C.D_MAX + 1e-12
    other_d1 = round(s.d1 + step, 6) if up else round(s.d1 - step, 6)
    return s.d1, R.rubinstein_share(s.d1, s.d2), other_d1, R.rubinstein_share(other_d1, s.d2)


# --- Reiter 2: Zeit und Horizont ---------------------------------------------------------------------------------------------------------------------

def horizon_curve(s: Settings, n: int = HORIZON_PLOT) -> dict:
    """Anteil des Verladers für T = 1 … n bei der gewählten Reihenfolge, dazu der Grenzwert."""
    d_first, d_second = ordered_patience(s)
    first_share = R.share_at_horizons(d_first, d_second, n)
    return {"T": list(range(1, n + 1)), "x1": [to_player1(s.first, v) for v in first_share], "limit": to_player1(s.first, R.rubinstein_share(d_first, d_second))}


def error_curves(pairs=C.E1_PAIRS, t_max: int = 400) -> dict:
    """Fehler |Anteil bei Horizont T − Formel| über T = 1 … t_max je Paar (δ₁; δ₂); Werte unter 1e-12 werden für die logarithmische Achse auf 1e-12 gehoben."""
    series = {}
    for d1, d2 in pairs:
        limit = R.rubinstein_share(d1, d2)
        series[f"({de(d1, 2)}; {de(d2, 2)})"] = [max(abs(v - limit), 1e-12) for v in R.share_at_horizons(d1, d2, t_max)]
    return {"T": list(range(1, t_max + 1)), "series": series}


def settle_row(d1: float, d2: float, tol: float = C.E1_TOL, t_max: int = C.E1_TMAX) -> dict:
    """Konvergenz der Rückwärtsinduktion gegen die Formel: kleinstes T, ab dem der Fehler für alle größeren T bis t_max unter tol bleibt; die Vorhersage aus (d₁d₂)^(T/2) < tol;
    dazu Anteile und Fehler bei T = 10, 11, 200 (Gleitkomma) und 500 (exakt in Brüchen)."""
    shares = R.share_at_horizons(d1, d2, t_max)
    limit = R.rubinstein_share(d1, d2)
    f1, f2 = Fraction(str(d1)), Fraction(str(d2))
    exact = {T: R.backward_T(f1, f2, T)[0] for T in (500, 501)}
    return {
        "d1": d1, "d2": d2, "limit": limit, "settled": R.settling_horizon(shares, limit, tol), "predicted": R.error_bound_rounds(d1, d2, tol),
        "x10": shares[9], "x11": shares[10], "x200": shares[199], "err200": abs(shares[199] - limit),
        "err500": float(abs(exact[500] - R.rubinstein_share(f1, f2))), "err501": float(abs(exact[501] - R.rubinstein_share(f1, f2))),
    }


# --- Reiter 3: Kurze Runden -----------------------------------------------------------------------------------------------------------------------

def short_rounds_rows(r1: float, r2: float, dts=C.E3_DTS) -> list:
    """Anteil des Verladers (Rate r₁) für Rundenlängen dt, wenn er zuerst bietet bzw. der Spediteur (Rate r₂) zuerst bietet; Grenzwert r₂/(r₁ + r₂); Vorteil des Zuerstbietens;
    Abstand zum Grenzwert geteilt durch dt gegen die Taylor-Entwicklung r₁ r₂/(2 (r₁ + r₂))."""
    limit = R.limit_share_short_rounds(r1, r2)
    rows = []
    for dt in dts:
        v_first = R.share_first_continuous_time(r1, r2, dt)
        v_second = 1 - R.share_first_continuous_time(r2, r1, dt)
        rows.append({"dt": dt, "x_first": v_first, "x_second": v_second, "limit": limit, "edge": v_first - v_second, "rate": (v_first - limit) / dt, "rate_hand": R.short_round_rate(r1, r2)})
    return rows


def short_rounds_row(s: Settings) -> dict:
    return short_rounds_rows(s.r1, s.r2, (s.dt,))[0]


# --- Reiter 4: Feste Kosten --------------------------------------------------------------------------------------------------------------------------

def costs_curve(s: Settings, n: int = 40) -> dict:
    """Anteil des Verladers für T = 1 … n bei festen Kosten und, wenn beide Kosten unter 1 liegen, bei Diskont d = 1 − c (gleiche Diskontrate in Runde 1)."""
    c_first, c_second = ordered_costs(s)
    cost_values = [to_player1(s.first, R.costs_backward(c_first, c_second, T)[0]) for T in range(1, n + 1)]
    comparable = c_first < 1 and c_second < 1
    discount = None
    if comparable:
        d_first, d_second = R.discount_from_costs(c_first), R.discount_from_costs(c_second)
        discount = [to_player1(s.first, v) for v in R.share_at_horizons(d_first, d_second, n)]
    return {"T": list(range(1, n + 1)), "costs": cost_values, "discount": discount}


def costs_case_row(c_first: float, c_second: float) -> dict:
    """Ein Fall der Messreihe: Handformel gegen die Rückwärtsinduktion bei T = 1000 und 1001, Diskont mit d = 1 − c daneben (nur für Kosten unter 1)."""
    pred = R.costs_prediction(c_first, c_second)
    r0 = R.costs_backward(c_first, c_second, 1000)
    r1 = R.costs_backward(c_first, c_second, 1001)
    row = {"c_first": c_first, "c_second": c_second, "pred_first": None if pred is None else pred[0], "pred_second": None if pred is None else pred[1],
           "t1000_first": r0[0], "t1000_second": r0[1], "t1001_first": r1[0], "t1001_second": r1[1], "discount": None}
    if c_first < 1 and c_second < 1:
        row["discount"] = R.rubinstein_share(R.discount_from_costs(c_first), R.discount_from_costs(c_second))
    return row


# --- Reiter 5: Außenoption --------------------------------------------------------------------------------------------------------------------------

def outside_curve(s: Settings, steps: int = 100) -> dict:
    """Anteil des Halters (Antwortender der ersten Runde) über der Option o = 0 … 0,99, unendlicher Horizont: Variante A, Variante C, Nash-Lösung mit Drohpunkt o."""
    d_first, d_second = ordered_patience(s)
    os_ = [i / steps for i in range(steps)]
    return {
        "o": os_,
        "A": [1 - O.outside_hand(d_first, d_second, o, "A") for o in os_],
        "C": [1 - O.outside_hand(d_first, d_second, o, "C") for o in os_],
        "nash": [O.nash_with_threat_holder_share(o) for o in os_],
        "without": 1 - R.rubinstein_share(d_first, d_second),
        "threshold_A": O.outside_threshold(d_first, d_second, "A"),
        "threshold_C": O.outside_threshold(d_first, d_second, "C"),
    }


def outside_state(s: Settings) -> dict:
    """Der Halter bei Horizont T mit und ohne Option; „counts“ heißt: der Anteil des Halters ändert sich um mehr als 1e-6."""
    d_first, d_second = ordered_patience(s)
    with_option = 1 - O.outside_backward(d_first, d_second, s.o, s.t, s.variant)[0]
    without = 1 - R.backward_T(d_first, d_second, s.t)[0]
    return {
        "holder": holder(s.first), "with": with_option, "without": without, "counts": abs(with_option - without) > 1e-6,
        "threshold": O.outside_threshold(d_first, d_second, s.variant), "payoff": O.exit_payoff(s.o, d_second, s.variant),
        "nash_threat": O.nash_with_threat_holder_share(s.o),
    }


def outside_scan(pairs=C.E5_PAIRS, os_=C.E5_OS, ts=C.E5_TS) -> dict:
    """Zählung für Variante B gegen A: in wie vielen Zellen (Paar, o, T) weicht B von A ab?"""
    cells = diff = 0
    for d1, d2 in pairs:
        for o in os_:
            for T in ts:
                cells += 1
                diff += abs(O.outside_backward(d1, d2, o, T, "A")[0] - O.outside_backward(d1, d2, o, T, "B")[0]) > 1e-12
    return {"cells": cells, "diff": diff}


def outside_check(pairs=C.E5_PAIRS, os_=C.E5_OS) -> dict:
    """Größte Abweichung der Rückwärtsinduktion (T = 1000 und 1001) von der Handformel je Variante."""
    out = {}
    for variant in ("A", "B", "C"):
        worst = 0.0
        for d1, d2 in pairs:
            for o in os_:
                for T in (1000, 1001):
                    worst = max(worst, abs(O.outside_backward(d1, d2, o, T, variant)[0] - O.outside_hand(d1, d2, o, variant)))
        out[variant] = worst
    return out


def nash_vs_exit_rows(d: float = 0.9999, os_=(0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95)) -> list:
    """Anteil des Halters nach Nash mit Drohpunkt o gegen das Outside-Option-Prinzip (Variante A) bei sehr großer Geduld beider Seiten."""
    return [{"o": o, "nash": O.nash_with_threat_holder_share(o), "exit": 1 - O.outside_hand(d, d, o, "A")} for o in os_]


# --- Reiter 6: Lösungskonzepte -----------------------------------------------------------------------------------------------------------------------

def solutions_for(s: Settings) -> dict:
    return S.all_solutions(s.curv, s.scale, S.scaled_threat(s.threat, s.scale))


def scale_rows(a: float, threat: float, ks=C.E6_KS) -> list:
    """Alle Lösungskonzepte für dieselbe Verhandlungsmenge in verschiedenen Maßstäben k von Spieler 2 (der Drohpunkt skaliert mit)."""
    return [{"a": a, "k": k, "threat": threat, **S.all_solutions(a, k, S.scaled_threat(threat, k))} for k in ks]


def curved_rubinstein_row(d1: float, d2: float, a: float) -> dict:
    """Rubinstein-Anteil des Verladers bei gekrümmtem Nutzen des Spediteurs (Diskont wirkt als d₂^(1/a)) und die Nash-Lösung ohne Drohpunkt."""
    return {"a": a, "rubinstein": S.rubinstein_utility(d1, d2, a), "nash": S.nash(a, 1.0, 0.0), "effective_d2": d2 ** (1 / a)}


def short_round_nash_gap(a: float, dt: float) -> float:
    """Rubinstein bei gleicher Rate und Rundenlänge dt minus Nash-Lösung (ohne Drohpunkt)."""
    d = math.exp(-dt)
    return S.rubinstein_utility(d, d, a) - S.nash(a, 1.0, 0.0)


# --- Reiter 7: Raster und Gleichstand ------------------------------------------------------------------------------------------------------------

def grid_curve(s: Settings, n: int = HORIZON_PLOT) -> dict:
    """Anteil des Verladers für T = 1 … n im Rasterspiel (G, Regel) und im stetigen Spiel."""
    d_first, d_second = ordered_patience(s)
    rule_a = s.rule == "A"
    grid = [to_player1(s.first, float(Ra.grid_share_first(d_first, d_second, T, s.g, rule_a))) for T in range(1, n + 1)]
    cont = R.share_at_horizons(d_first, d_second, n)
    return {"T": list(range(1, n + 1)), "grid": grid, "continuous": [to_player1(s.first, v) for v in cont],
            "limit": to_player1(s.first, R.rubinstein_share(d_first, d_second))}


def grid_state(s: Settings) -> dict:
    """Das Rasterspiel bei den gewählten Einstellungen: Anteil des ersten Anbieters, Fehler gegen die Formel mal G, Schranke, stationäre Gleichgewichte, Regel A gegen R."""
    d_first, d_second = ordered_patience(s)
    f1, f2 = Ra.to_fraction(d_first), Ra.to_fraction(d_second)
    rule_a = s.rule == "A"
    share = Ra.grid_share_first(f1, f2, s.t, s.g, rule_a)
    other_share = Ra.grid_share_first(f1, f2, s.t, s.g, not rule_a)
    limit = R.rubinstein_share(d_first, d_second)
    cont = R.backward_T(d_first, d_second, s.t)[0]
    points = Ra.grid_fixed_points(f1, f2, s.g, rule_a)
    bound = float(Ra.error_bound(f1, f2, s.g))
    shares = [float(a1) for a1, _ in points]
    return {
        "share_first": float(share), "other_rule_share": float(other_share), "continuous": cont, "limit": limit, "error_times_g": (float(share) - limit) * s.g,
        "bound": bound, "ratio": abs(float(share) - limit) / bound, "points": len(points), "points_lo": min(shares) if shares else None, "points_hi": max(shares) if shares else None,
    }


def bound_cells(pairs, gs, ts) -> dict:
    """Zählung zur Schranke |Anteil im Raster − Formel| ≤ 1/((1 − d₁d₂) G): Zellen, eingehaltene, größtes Verhältnis, Ausnahmen (beide Regeln)."""
    cells = ok = 0
    worst = 0.0
    exceptions = []
    for d1, d2 in pairs:
        f1, f2 = Ra.to_fraction(d1), Ra.to_fraction(d2)
        limit = R.rubinstein_share(f1, f2)
        for G in gs:
            bound = Ra.error_bound(f1, f2, G)
            for rule_a in (True, False):
                for T in ts:
                    ratio = float(abs(Ra.grid_share_first(f1, f2, T, G, rule_a) - limit) / bound)
                    cells += 1
                    ok += ratio <= 1.0
                    worst = max(worst, ratio)
                    if ratio > 1.0:
                        exceptions.append([d1, d2, G, "A" if rule_a else "R", T, ratio])
    return {"cells": cells, "ok": ok, "worst": worst, "exceptions": exceptions}


def fixed_row(d1: float, d2: float, g: int, rule_a: bool) -> dict:
    f1, f2 = Ra.to_fraction(d1), Ra.to_fraction(d2)
    pts = [float(a1) for a1, _ in Ra.grid_fixed_points(f1, f2, g, rule_a)]
    return {"d1": d1, "d2": d2, "g": g, "rule": "A" if rule_a else "R", "count": len(pts), "lo": min(pts) if pts else None, "hi": max(pts) if pts else None, "limit": R.rubinstein_share(d1, d2)}


def rule_gap(d1: float, d2: float, g: int, ts=range(1, 61)) -> float:
    """Größter Unterschied zwischen Regel A und R über T = 1 … 60 (erster Anbieter)."""
    f1, f2 = Ra.to_fraction(d1), Ra.to_fraction(d2)
    return float(max(abs(Ra.grid_share_first(f1, f2, T, g, True) - Ra.grid_share_first(f1, f2, T, g, False)) for T in ts))


# --- Reiter 8: Lernende Verhandler -----------------------------------------------------------------------------------------------------------------

def learn_once(game: str, rule: str, eta: float, rounds: int, seed: int) -> dict:
    u1, u2 = L.game_matrices(game, C.LEARN_GRID, C.LEARN_DELTA)
    res = L.run(u1, u2, rule, eta, rounds, seed)
    return {"game": game, "rule": rule, "eta": eta, "rounds": rounds, "seed": seed, "result": res, "spe": float(L.spe_share(game, C.LEARN_GRID, C.LEARN_DELTA)),
            "pure_nash": len(L.pure_nash_equilibria(u1, u2))}


def learn_cell(game: str, rule: str, eta: float, seeds=C.GRID_LEARN_SEEDS, rounds: int = C.GRID_LEARN_ROUNDS) -> dict:
    """Vier Zufalls-Seeds je Zelle: Anteil des Anbieters (Spieler 1) und Nash-Lücke des letzten Iterierten."""
    shares, gaps = [], []
    for seed in seeds:
        res = learn_once(game, rule, eta, rounds, seed)["result"]
        shares.append(res.share1)
        gaps.append(res.gap)
    return {"game": game, "rule": rule, "eta": eta, "rounds": rounds, "seeds": list(seeds), "shares": shares, "gaps": gaps,
            "spe": float(L.spe_share(game, C.LEARN_GRID, C.LEARN_DELTA))}


def learn_cells() -> list:
    return [(game, rule, eta) for game in C.GAMES for rule, eta in (("hedge", 0.5), ("hedge", 2.0), ("hedge", 8.0), ("rm", 0.0))]


# --- Vorgerechnete Messreihen ---------------------------------------------------------------------------------------------------------------------

def load_grid():
    """Die Messreihen aus data/negotiation_grid.json (tools/make_grid.py); None, wenn die Datei fehlt."""
    try:
        return json.loads(GRID_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find(cells: list, **kw):
    """Erste Zelle, deren Felder alle den Werten in kw entsprechen (Zahlen mit Toleranz 1e-9)."""
    for cell in cells:
        if all((abs(cell[k] - v) < 1e-9 if isinstance(v, float) else cell[k] == v) for k, v in kw.items()):
            return cell
    return None


def grid_probe() -> dict:
    """Kleine feste Probe, die tests/test_grid.py gegen die gespeicherte Datei prüft: ändert sich Rechnung oder Konstante, passt die Datei nicht mehr (dann tools/make_grid.py)."""
    return {
        "settle": settle_row(0.9, 0.9, t_max=300),
        "bound": bound_cells([(0.9, 0.9)], (10, 100), (600, 601)),
        "fixed": fixed_row(0.9, 0.9, 10, True),
        "rounds": short_rounds_rows(1.0, 2.0, (1.0, 0.1)),
        "costs": costs_case_row(0.05, 0.10),
        "outside": outside_scan([(0.9, 0.9)], (0.2, 0.6), range(1, 11)),
        "scale": scale_rows(0.5, 0.1),
        "learn": learn_cell("ultimatum", "hedge", 2.0, seeds=(1,), rounds=2000),
    }


# --- Verdikte (reine Funktionen: (Art, Text)); Zahlen kommen aus der Rechnung ------------------------------------------------------------------

def _dt_label(dt: float) -> str:
    return C.DT_LABELS.get(dt, f"{de(dt, 4)} Tage")


def _side(player: int) -> str:
    return C.NAMES[player]


def _the(player: int) -> str:
    return "der " + C.NAMES[player]


def _dat(player: int) -> str:
    return "dem " + C.NAMES[player]


def verdict_main(a: Analysis) -> tuple:
    s = a.settings
    first, second = _side(s.first), _side(other(s.first))
    if s.t == 1:
        return "info", (f"Eine Runde (Ultimatum): {first} bietet, {second} nimmt an oder geht leer aus. Bei stetigen Angeboten und Annahme bei Gleichstand bekommt {first} {pct(a.share_first)} des "
                        f"Überschusses, der Verlader also {pct(a.x1)}. Die Geduld spielt keine Rolle, nur wer bietet.")
    v_first = a.x1 if s.first == 1 else a.other_order_x1
    v_second = a.other_order_x1 if s.first == 1 else a.x1
    return "info", (f"Der Verlader bekommt {pct(a.x1)} des Überschusses, der Spediteur {pct(a.x2)} ({first} bietet zuerst, T = {s.t} Runden, stetige Angebote, Annahme bei Gleichstand). "
                    f"Halbe-halbe wäre 50,0 %; die Rubinstein-Formel (unendlicher Horizont) gibt dem Verlader {pct(a.limit_x1)}. Bietet der Verlader zuerst, bekommt er {pct(v_first)}, bietet der "
                    f"Spediteur zuerst, {pct(v_second)} (Vorteil des Zuerstbietens für den Verlader: {pp(v_first - v_second)}).")


def verdict_patience(s: Settings) -> tuple:
    d_old, x_old, d_new, x_new = patience_step(s)
    word = "eine Stufe mehr" if d_new > d_old else "eine Stufe weniger"
    return "info", (f"Bei Geduld des Spediteurs δ₂ = {de(s.d2, 2)} (Verlader bietet zuerst, unendlicher Horizont): δ₁ = {de(d_old, 2)} gibt dem Verlader {pct(x_old)}, {word} (δ₁ = {de(d_new, 2)}) "
                    f"{pct(x_new)}, also {pp(x_new - x_old, 2)}. Halbe-halbe (Shapley-Wert) bleibt bei 50,0 %. Die Steigung dx₁/dδ₁ liegt bei δ₁ = {de(d_old, 2)} bei {de(patience_slope(d_old, s.d2), 2)}.")


def verdict_horizon(s: Settings) -> tuple:
    a = analyse(s)
    d_first, d_second = ordered_patience(s)
    first = _side(s.first)
    if s.t == 1:
        return "info", f"T = 1 ist das Ultimatum: {first} bekommt {pct(a.share_first)}, der andere nimmt an, weil Ablehnen nichts bringt. Der Grenzwert der Formel wäre {pct(a.limit_first)}."
    side = "über" if a.share_first > a.limit_first else "unter" if a.share_first < a.limit_first else "auf"
    parity = "ungerade" if s.t % 2 == 1 else "gerade"
    gap = "0" if a.error == 0 else sci(100 * a.error)
    return "info", (f"Bei T = {s.t} ({parity}) bekommt der erste Anbieter ({first}) {pct(a.share_first, 2)}, {side} dem Grenzwert {pct(a.limit_first, 2)}; Abstand {gap} Prozentpunkte. "
                    f"Ungerade Horizonte liegen nicht unter, gerade nicht über dem Grenzwert, und der Abstand schrumpft je zwei Runden um den Faktor δ₁δ₂ = {de(d_first * d_second, 3)}.")


def verdict_short_rounds(s: Settings) -> tuple:
    row = short_rounds_row(s)
    return "info", (f"Runden von {_dt_label(s.dt)} (Diskontraten r₁ = {de(s.r1, 1)}, r₂ = {de(s.r2, 1)} je Tag): bietet der Verlader zuerst, bekommt er {pct(row['x_first'], 2)}, bietet der Spediteur zuerst, "
                    f"{pct(row['x_second'], 2)}; Vorteil des Zuerstbietens {pp(row['edge'], 2)}. Grenzwert für Runden gegen null: r₂/(r₁ + r₂) = {pct(row['limit'], 2)} (Nash-Lösung mit den Gewichten 1/r₁ "
                    f"und 1/r₂), Halbe-halbe 50,0 %.")


def verdict_costs(s: Settings) -> tuple:
    c_first, c_second = ordered_costs(s)
    first = _side(s.first)
    first_share = R.costs_backward(c_first, c_second, s.t)[0]
    x1 = to_player1(s.first, first_share)
    discount_text = ""
    if c_first < 1 and c_second < 1:
        d_first, d_second = R.discount_from_costs(c_first), R.discount_from_costs(c_second)
        discount_text = f" Mit Diskont d = 1 − c (gleiche Diskontrate in Runde 1) bekäme der erste Anbieter {pct(R.backward_T(d_first, d_second, s.t)[0])}."
    pred = R.costs_prediction(c_first, c_second)
    head = f"Feste Kosten je Runde (Verlader {de(s.c1, 2)}, Spediteur {de(s.c2, 2)}), {first} bietet zuerst, T = {s.t}: der erste Anbieter bekommt {pct(first_share)}, der Verlader {pct(x1)}."
    if pred is None:
        parity = "ungerade" if s.t % 2 == 1 else "gerade"
        return "info", (f"{head} Gleiche Kosten: im unendlichen Spiel gibt es kein eindeutiges Ergebnis; ein {parity}s T gibt dem ersten Anbieter "
                        f"{'den ganzen Überschuss' if s.t % 2 == 1 else 'nur seine Kosten c'}.{discount_text}")
    if c_first < c_second:
        return "info", (f"{head} {first} hat die kleineren Kosten ({de(c_first, 2)} < {de(c_second, 2)}): im langen Spiel bekommt er genau 100 %, als zweiter Anbieter nur {pct(c_first)}.{discount_text}")
    return "info", (f"{head} {first} hat die größeren Kosten ({de(c_first, 2)} > {de(c_second, 2)}): im langen Spiel bekommt er nur {pct(c_second)} (die Kosten des anderen), der andere als erster "
                    f"Anbieter 100 %.{discount_text}")


def verdict_outside(s: Settings) -> tuple:
    st = outside_state(s)
    who = _the(st["holder"])
    dat = _dat(st["holder"])
    unit = "o" if s.variant == "A" else "δ·o"
    if st["counts"]:
        return "info", (f"Die Option zählt: {who} bekommt {pct(st['with'])} statt {pct(st['without'])} ohne Option (Variante {s.variant}: Erlös {unit} = {de(st['payoff'], 3)}, Schwelle für o: "
                        f"{de(st['threshold'], 3)}, o = {de(s.o, 2)}, T = {s.t}). Die Nash-Lösung mit Drohpunkt o gäbe {dat} {pct(st['nash_threat'])}: eine andere Lesart von „Außenoption“.")
    return "info", (f"Die Option ändert nichts: der Erlös {unit} = {de(st['payoff'], 3)} liegt nicht über dem Anteil {pct(st['without'])}, den {who} ohne Option bekommt (Schwelle für o: "
                    f"{de(st['threshold'], 3)}, o = {de(s.o, 2)}, Variante {s.variant}, T = {s.t}). Die Nash-Lösung mit Drohpunkt o gäbe {dat} dagegen {pct(st['nash_threat'])}.")


def verdict_solutions(s: Settings) -> tuple:
    sol = solutions_for(s)
    k_other = 1.0 if s.scale != 1.0 else 10.0
    other_sol = S.all_solutions(s.curv, k_other, S.scaled_threat(s.threat, k_other))
    d2 = S.scaled_threat(s.threat, s.scale)
    return "info", (f"Anteil des Verladers bei Krümmung a = {de(s.curv, 2)}, Maßstab k = {de(s.scale, 1)}, Drohpunkt {de(d2, 3)} (= {de(s.threat, 2)}·k): Nash {pct(sol['nash'])}, Kalai–Smorodinsky "
                    f"{pct(sol['ks'])}, egalitär {pct(sol['egalitarian'])}, utilitaristisch {pct(sol['utilitarian'])}, Halbe-halbe 50,0 %. Beim Maßstab k = {de(k_other, 1)} statt {de(s.scale, 1)}: Nash "
                    f"{pct(other_sol['nash'])}, Kalai–Smorodinsky {pct(other_sol['ks'])}, egalitär {pct(other_sol['egalitarian'])}, utilitaristisch {pct(other_sol['utilitarian'])}.")


def verdict_grid(s: Settings) -> tuple:
    g = grid_state(s)
    first = _side(s.first)
    holds = "eingehalten" if g["ratio"] <= 1 else "nicht eingehalten"
    points = "keines" if g["points"] == 0 else f"{g['points']} (Anteil des ersten Anbieters von {pct(g['points_lo'], 1)} bis {pct(g['points_hi'], 1)})"
    return "info", (f"Raster G = {s.g}, Regel {s.rule}, T = {s.t}: der erste Anbieter ({first}) bekommt {pct(g['share_first'], 2)}, im stetigen Spiel {pct(g['continuous'], 2)}, nach der Formel {pct(g['limit'], 2)}. "
                    f"Fehler gegen die Formel mal G: {de(g['error_times_g'], 2)}; die Schranke 1/((1 − δ₁δ₂)·G) aus der Messung (kein Satz) liegt bei {de(g['bound'] * s.g, 2)} mal 1/G und wird {holds}. "
                    f"Mit der anderen Gleichstandsregel: {pct(g['other_rule_share'], 2)}. Stationäre Gleichgewichte mit sofortiger Einigung: {points}.")


def verdict_learning(run: dict) -> tuple:
    res = run["result"]
    reach = "höchstens" if res.gap <= NASH_GAP_LIMIT else "über"
    return "info", (f"{C.LEARN_RULES[run['rule']]}, {C.GAMES[run['game']]}, {run['rounds']} Runden, Zufalls-Seed {run['seed']}: Spieler 1 (Anbieter) bekommt unter dem Endprofil {pct(res.share1)}, "
                    f"Spieler 2 {pct(res.share2)}; Nash-Lücke {de(res.gap, 3)} ({reach} {de(NASH_GAP_LIMIT, 2)}). Das teilspielperfekte Ergebnis wäre {pct(run['spe'])} für Spieler 1: "
                    f"Abstand {pp(res.share1 - run['spe'])}. Ein einzelner Lauf sagt nichts darüber, wie echte Verhandler handeln.")


# --- Verdikte der Experimente (aus den vorgerechneten Messreihen) --------------------------------------------------------------------------------

_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in der Schreibweise 2,2·10⁻⁵ (deutsches Komma)."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10 ** e, digits)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{digits}f}".replace(".", ",") + "·10" + str(e).translate(_SUPER)


def _pair(d1: float, d2: float) -> str:
    return f"({de(d1, 2)}; {de(d2, 2)})"


def verdict_e1(rows: list) -> tuple:
    parts = [f"{_pair(r['d1'], r['d2'])}: {r['settled']} gemessen, {round(r['predicted'])} vorhergesagt" for r in rows]
    worst = max(abs(r["predicted"] / r["settled"] - 1) for r in rows)
    return "info", (f"Kleinstes T, ab dem der Fehler gegen die Formel für alle größeren T bis {C.E1_TMAX} unter 10⁻⁶ bleibt, je Paar (δ₁; δ₂): " + "; ".join(parts) + f". Die Vorhersage aus (δ₁δ₂)^(T/2) < 10⁻⁶ "
                    f"weicht in diesen {len(rows)} Paaren höchstens um {de(100 * worst, 1)} % ab. Bei δ₁ = δ₂ = 0,99 sind es {rows[1]['settled']} Runden; der Regler erlaubt höchstens T = {C.T_MAX}.")


def verdict_e2(bound3: dict, more_a: dict, more_b: dict, gap_rows: list) -> tuple:
    cells = bound3["cells"] + more_a["cells"] + more_b["cells"]
    ok = bound3["ok"] + more_a["ok"] + more_b["ok"]
    gaps = sorted({round(r["gap"] * r["g"], 9) for r in gap_rows})
    gap_text = "genau 1/G" if gaps == [1.0] else "zwischen " + de(gaps[0], 3) + " und " + de(gaps[-1], 3) + " mal 1/G"
    return "info", (f"Die Schranke |Anteil im Raster − Formel| ≤ 1/((1 − δ₁δ₂)·G) gilt in {bound3['ok']} von {bound3['cells']} Zellen der drei Paare (G = 10 bis 10 000, beide Regeln, T = 600 und 601; größtes "
                    f"Verhältnis {de(bound3['worst'], 2)}) und in {ok} von {cells} Zellen insgesamt; die {cells - ok} Ausnahmen sind δ₁ = δ₂ = 0,99 mit G = 10 000 bei T = 400 und 401, wo T selbst noch nicht "
                    f"konvergiert ist. Die Schranke ist eine Messhypothese, kein Satz. Regel A gegen R: der größte Unterschied über T = 1 bis 60 ist in allen {len(gap_rows)} Zellen "
                    f"(4 Paare, G = 10, 100, 1000) {gap_text}.")


def verdict_e3(rows_by_rates: dict) -> tuple:
    parts = []
    for key, rows in rows_by_rates.items():
        r1, r2 = (float(x) for x in key.split("|"))
        last = rows[-1]
        parts.append(f"({de(r1, 1)}; {de(r2, 1)}): {de(last['rate'], 6)} gegen {de(last['rate_hand'], 6)}")
    first = next(iter(rows_by_rates.values()))
    at01, at0001 = find(first, dt=0.1), find(first, dt=0.001)
    return "info", (f"Abstand zum Grenzwert geteilt durch die Rundenlänge bei Δ = {de(first[-1]['dt'], 4)}, je Raten (r₁; r₂), gemessen gegen die Taylor-Entwicklung r₁r₂/(2(r₁ + r₂)): " + "; ".join(parts) +
                    f". Bei gleichen Raten (1; 1) sinkt der Vorteil des Zuerstbietens von {de(at01['edge'], 4)} bei Δ = {de(at01['dt'], 1)} auf {de(at0001['edge'], 6)} bei Δ = {de(at0001['dt'], 3)}.")


def verdict_e4(rows: list) -> tuple:
    unequal = [r for r in rows if r["pred_first"] is not None]
    agree = sum(1 for r in unequal if all(abs(r[k] - v) < 1e-9 for k, v in (("t1000_first", r["pred_first"]), ("t1001_first", r["pred_first"]), ("t1000_second", r["pred_second"]), ("t1001_second", r["pred_second"]))))
    equal = next((r for r in rows if r["pred_first"] is None), None)
    tail = ""
    if equal is not None:
        tail = (f" Bei gleichen Kosten ({de(equal['c_first'], 2)}) gibt T = 1000 dem ersten Anbieter {pct(equal['t1000_first'])} und T = 1001 {pct(equal['t1001_first'])}: kein eindeutiges Ergebnis.")
    return "info", (f"In {agree} von {len(unequal)} Fällen mit ungleichen Kosten stimmt die Rückwärtsinduktion bei T = 1000 und 1001 mit der Handformel überein (wer weniger Kosten hat und zuerst bietet, bekommt "
                    f"genau 100 %; hat der erste die größeren, bekommt er die Kosten des anderen).{tail} Mit Diskont d = 1 − c stünde im Fall (0,05; 0,10) {pct(find(rows, c_first=0.05, c_second=0.1)['discount'])} statt 100 %.")


def verdict_e5(scan: dict, check: dict, rows: list) -> tuple:
    r02 = next(r for r in rows if abs(r["o"] - 0.2) < 1e-9)
    return "info", (f"Variante B (der Halter kann auch in seiner eigenen Anbieterrunde aussteigen) weicht in {scan['diff']} von {scan['cells']} Zellen (4 Paare, 10 Werte o, T = 1 bis 60) von Variante A ab: "
                    f"deshalb wird sie nicht als Regler angeboten. Rückwärtsinduktion gegen Handformel bei T = 1000 und 1001: größte Abweichung {sci(max(check.values()))}. Nash-Lösung mit Drohpunkt o = 0,2 gegen "
                    f"Outside-Option-Prinzip bei δ = 0,9999: {pct(r02['nash'], 1)} gegen {pct(r02['exit'], 1)} für den Halter.")


def verdict_e6(rows: list) -> tuple:
    by_a = {}
    for r in rows:
        by_a.setdefault(r["a"], []).append(r)
    inv = max(max(abs(r["nash"] - rs[0]["nash"]), abs(r["ks"] - rs[0]["ks"])) for rs in by_a.values() for r in rs)
    half = by_a[1.0]
    same = "auf 9 Stellen" if inv < 1e-9 else f"Abweichung bis {sci(inv)}"
    return "info", (f"Bei Krümmung a = 1 und Drohpunkt 0,1·k bleiben Nash ({pct(half[0]['nash'])}) und Kalai–Smorodinsky ({pct(half[0]['ks'])}) für k = 0,1, 1 und 10 gleich ({same} bei allen vier "
                    f"Krümmungen); der egalitäre Anteil des Verladers springt von {pct(half[2]['egalitarian'])} (k = 0,1) über {pct(half[0]['egalitarian'])} (k = 1) auf {pct(half[1]['egalitarian'])} "
                    f"(k = 10).")


def verdict_e7(cells: list) -> tuple:
    runs = sum(len(c["shares"]) for c in cells)
    hit = sum(1 for c in cells for x in c["shares"] if abs(x - c["spe"]) < 1e-9)
    near = sum(1 for c in cells for g in c["gaps"] if g <= NASH_GAP_LIMIT)
    return "info", (f"{runs} Läufe ({len(cells)} Zellen mal {len(cells[0]['seeds'])} Zufalls-Seeds, {cells[0]['rounds']} Runden, letzter Iterierter): {hit} enden im teilspielperfekten Anteil der Regel A, {near} haben eine Nash-Lücke "
                    f"von höchstens {de(NASH_GAP_LIMIT, 2)}. Der Anteil hängt an der Lernrate und, bei Regret Matching, am Zufalls-Seed (Bänder in der Tabelle). Das ist ein Befund über diese beiden Lernregeln "
                    f"in diesen zwei Spielen, keine Aussage über echte Verhandler.")
