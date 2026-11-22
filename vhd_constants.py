"""Konstanten: Regler-Grenzen, Wahlmöglichkeiten, Raster der Messreihen, Farben und Presets.

Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): Rückwärtsinduktion und Kosten sind O(T), also bei T = 2 000 Millisekunden; das Rasterspiel rechnet in
Brüchen (fractions) ebenfalls O(T), bei T = 2 000 und G = 10 000 gemessen unter einer halben Sekunde; die stationären Fixpunkte des Rasters kosten O(G). Die
Lernläufe (vhd_learning) sind die einzige Rechnung mit Wartezeit: O(Runden · Aktionen) je Lauf, deshalb hinter einem Knopf mit Spinner (Messung im
Abschnitt „Lernende Verhandler“ der App und in der README). Die Messreihen der Experimente stehen vorgerechnet in data/negotiation_grid.json."""

from vhd_rng import (
    MASK,  # noqa: F401  (Wiederverwendung der Kopie; vhd_rng.py bleibt wörtlich)
)

# --- Geduld, Horizont, wer zuerst -------------------------------------------------------------------------------------------------------------
D_MIN, D_MAX = 0.0, 0.99            # δ = 0,999 würde in 2 000 Runden nicht konvergieren (Fehler < 10⁻⁶ erst nach ≈ 13 000 Runden); 0,99 braucht 1 307
DEFAULT_D1, DEFAULT_D2 = 0.9, 0.9
T_MIN, T_MAX, DEFAULT_T = 1, 2000, 200
FIRSTS = {1: "Verlader (Spieler 1)", 2: "Spediteur (Spieler 2)"}
DEFAULT_FIRST = 1
NAMES = {1: "Verlader", 2: "Spediteur"}

# --- Raster ------------------------------------------------------------------------------------------------------------------------------------
G_OPTIONS = (10, 20, 50, 100, 1000, 10000)
DEFAULT_G = 100
RULES = {"A": "A: Annahme bei Gleichstand", "R": "R: Ablehnung bei Gleichstand"}
DEFAULT_RULE = "A"
FIXPOINT_G_MAX = 10000              # O(G) je Aufruf (grid_fixed_points)

# --- feste Kosten -----------------------------------------------------------------------------------------------------------------------------
COST_MIN, COST_MAX = 0.0, 1.2
DEFAULT_C1, DEFAULT_C2 = 0.10, 0.10    # gleiche Kosten: kein eindeutiges Ergebnis (Reiter „Feste Kosten“)

# --- kurze Runden ------------------------------------------------------------------------------------------------------------------------------
RATE_MIN, RATE_MAX = 0.1, 5.0       # Diskontrate je Tag: δ = exp(−r·Δ)
DEFAULT_R1, DEFAULT_R2 = 1.0, 2.0
DT_OPTIONS = (1.0, 0.5, 0.25, 0.1, 1 / 24, 0.01, 1 / 1440, 0.0001)   # Rundenlänge in Tagen: 1 Tag … 1 Stunde … 1 Minute … 8,6 s
DT_LABELS = {1.0: "1 Tag", 0.5: "12 Stunden", 0.25: "6 Stunden", 0.1: "2,4 Stunden", 1 / 24: "1 Stunde", 0.01: "14,4 Minuten", 1 / 1440: "1 Minute", 0.0001: "8,64 Sekunden"}
DEFAULT_DT = 1.0

# --- Außenoption ------------------------------------------------------------------------------------------------------------------------------
O_MIN, O_MAX, DEFAULT_O = 0.0, 0.99, 0.3
VARIANTS = {"A": "A: Erlös sofort nach der Ablehnung", "C": "C: Erlös eine Runde später"}
DEFAULT_VARIANT = "A"

# --- Lösungskonzepte -------------------------------------------------------------------------------------------------------------------------------
CURV_MIN, CURV_MAX, DEFAULT_CURV = 0.25, 2.0, 1.0      # Nutzenkrümmung a: u₂ = k·(1 − x)^a (a < 1: risikoscheu, a > 1: Menge nicht konvex)
SCALE_OPTIONS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)   # Maßstab k des Nutzens von Spieler 2
DEFAULT_SCALE = 1.0
THREAT_MIN, THREAT_MAX, DEFAULT_THREAT = 0.0, 0.5, 0.1  # Drohpunkt von Spieler 2 als Anteil von k (wird mit k mitskaliert)

# --- Lernende Verhandler -----------------------------------------------------------------------------------------------------------------------
GAMES = {"ultimatum": "Ultimatumspiel (eine Runde)", "zwei_runden": "Zwei Runden (Diskont 0,9)"}
DEFAULT_GAME = "ultimatum"
LEARN_RULES = {"hedge": "Hedge (Entropie-Gewichte)", "rm": "Regret Matching"}
DEFAULT_LEARN_RULE = "hedge"
ETA_OPTIONS = (0.5, 2.0, 8.0)                           # Lernrate η₀ (Hedge: η₀/√t)
DEFAULT_ETA = 2.0
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 1000, 50000, 20000   # O(Runden · Aktionen) je Lauf
LEARN_GRID = 10                                          # Rasterfeinheit G der Lernspiele
LEARN_DELTA = 0.9                                        # Diskont im Zwei-Runden-Spiel
SEED_MAX, DEFAULT_SEED = 999999, 35
GRID_LEARN_SEEDS = (1, 2, 3, 4)                          # vier Zufalls-Seeds je Zelle der vorgerechneten Lernläufe
GRID_LEARN_ROUNDS = 20000

# --- Messreihen (tools/make_grid.py) -----------------------------------------------------------------------------------------------------
E1_PAIRS = ((0.9, 0.9), (0.99, 0.99), (0.95, 0.8), (0.5, 0.9))   # Paare (δ₁, δ₂) der Konvergenzmessung
E1_TOL = 1e-6
E1_TMAX = 2000
E2_PAIRS3 = ((0.9, 0.9), (0.95, 0.8), (0.8, 0.95))               # die drei Paare der Schranken-Zählung
E2_PAIRS_MORE = ((0.99, 0.99), (0.99, 0.9), (0.5, 0.5), (0.7, 0.7), (0.97, 0.97))
E2_GS = (10, 50, 100, 1000, 10000)
E2_TS = (600, 601)
E3_RATES = ((1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (1.0, 4.0), (0.5, 3.0))
E3_DTS = (1.0, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001)
E4_CASES = ((0.05, 0.10), (0.10, 0.05), (0.02, 0.30), (0.30, 0.02), (0.20, 0.20), (0.0, 0.05), (0.05, 0.0), (0.6, 0.7), (1.2, 0.3))
E5_PAIRS = ((0.9, 0.9), (0.95, 0.8), (0.8, 0.95), (0.99, 0.99))
E5_OS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6, 0.8, 0.95)
E5_TS = tuple(range(1, 61))
E6_AS = (1.0, 0.5, 0.25, 2.0)
E6_KS = (1.0, 10.0, 0.1)

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------------
COLOR_P1 = "#4c78a8"        # Verlader (Spieler 1)
COLOR_P2 = "#e45756"        # Spediteur (Spieler 2)
COLOR_FORMULA = "#54a24b"   # Formel / Grenzwert
COLOR_HALF = "#9d9d9d"      # Halbe-halbe (Shapley)
COLOR_ALT = "#f58518"       # zweite Kurve (Variante, Kosten, Regel R)
COLOR_DARK = "#222222"

_BASE = {
    "d1": DEFAULT_D1, "d2": DEFAULT_D2, "t": DEFAULT_T, "first": DEFAULT_FIRST,
    "g": DEFAULT_G, "rule": DEFAULT_RULE,
    "c1": DEFAULT_C1, "c2": DEFAULT_C2,
    "r1": DEFAULT_R1, "r2": DEFAULT_R2, "dt": DEFAULT_DT,
    "o": DEFAULT_O, "variant": DEFAULT_VARIANT,
    "curv": DEFAULT_CURV, "scale": DEFAULT_SCALE, "threat": DEFAULT_THREAT,
    "game": DEFAULT_GAME, "learn_rule": DEFAULT_LEARN_RULE, "eta": DEFAULT_ETA, "rounds": DEFAULT_ROUNDS, "seed": DEFAULT_SEED,
}
PRESETS = {
    "Gleiche Geduld (Voreinstellung)": dict(_BASE),
    "Geduld ist Macht": {**_BASE, "d1": 0.91},
    "Ultimatum (eine Runde)": {**_BASE, "t": 1, "g": 10, "rule": "R"},
    "Kurze Runden": {**_BASE, "dt": 1 / 24},
    "Feste Kosten": {**_BASE, "c1": 0.05, "c2": 0.10},
    "Außenoption unter der Schwelle": {**_BASE, "o": 0.4},
    "Außenoption über der Schwelle": {**_BASE, "o": 0.6},
    "Risikoscheu": {**_BASE, "curv": 0.5},
    "Rasterfalle": {**_BASE, "g": 10},
}
# Die Hilfetexte nennen Zahlen der Auswertung; tests/test_presets.py und tests/test_claims.py prüfen jede davon (Bedingung im Text).
PRESET_HELP = {
    "Gleiche Geduld (Voreinstellung)": "Beide Seiten mit Geduld 0,9, 200 Runden, stetige Angebote, der Verlader bietet zuerst: er bekommt 52,6 % des Überschusses (Formel 10/19), nicht die Hälfte.",
    "Geduld ist Macht": "Der Verlader wird von 0,90 auf 0,91 geduldiger: sein Anteil steigt von 52,6 % auf 55,2 % (Spediteur 0,90, 200 Runden, stetige Angebote, Verlader bietet zuerst).",
    "Ultimatum (eine Runde)": "Nur eine Runde, Verlader bietet: stetig bekommt er 100 %; auf dem Raster mit Zehnteln und Ablehnung bei Gleichstand (Regel R) 90 %. Reiter „Raster und Gleichstand“.",
    "Kurze Runden": "Diskontraten 1 (Verlader) und 2 (Spediteur) je Tag, Runden von einer Stunde statt einem Tag: der Verlader bekommt als erster Anbieter 68,0 % statt 91,0 %; Grenzwert für Runden gegen null 66,7 %. Reiter „Kurze Runden“.",
    "Feste Kosten": "Feste Kosten je Runde (Verlader 0,05, Spediteur 0,10, Verlader bietet zuerst, 200 Runden): der Verlader bekommt 100 %; bei Diskont mit d = 1 − c wären es 69,0 %. Reiter „Feste Kosten“.",
    "Außenoption unter der Schwelle": "Außenoption 0,4 des Spediteurs liegt unter der Schwelle 0,474 (Geduld 0,9, Variante A, 200 Runden): sie ändert nichts, der Spediteur bekommt 47,4 %. Reiter „Außenoption“.",
    "Außenoption über der Schwelle": "Außenoption 0,6 über der Schwelle 0,474 (Geduld 0,9, Variante A, 200 Runden): der Spediteur bekommt genau 60,0 %. Reiter „Außenoption“.",
    "Risikoscheu": "Der Nutzen des Spediteurs ist gekrümmt (Krümmung 0,5, Maßstab 1, Drohpunkt 0,1; in der Nash-Lösung risikoscheu, im Rubinstein-Spiel wie größere Ungeduld): die Nash-Lösung gibt dem Verlader 62,6 % statt 45,0 % bei linearem Nutzen. Reiter „Lösungskonzepte“.",
    "Rasterfalle": "Angebote nur in Zehnteln, Geduld 0,9, 200 Runden (gerade), Annahme bei Gleichstand: der Verlader bekommt 10,0 % statt 52,6 % im stetigen Spiel. Reiter „Raster und Gleichstand“.",
}

# Dieser Text steht wortgleich in der README (tests/test_texts.py vergleicht beide Fassungen).
ASSUMPTIONS_MARKDOWN = """**Annahmen**

- **Vollständige Information:** beide Seiten kennen Wert, Kosten und Geduld der anderen; private Information ist nicht Teil dieser Demo (die Myerson–Satterthwaite-Demo rechnet private Werte, ohne Ablauf in der Zeit).
- **Zwei Parteien, abwechselnde Angebote:** der Überschuss S = v − c der Tour ist auf 1 normiert, x ist der Anteil des Verladers, die Rate ist c + (1 − x)·S. Wer antwortet, nimmt an oder lehnt ab; Wartekosten sind ein Diskontfaktor je Runde oder feste Kosten je Runde.
- **Eigennutz, Geduld und Nutzenfunktionen sind Modellannahmen**, keine Aussagen über echte Verhandler. Der Nutzen ist linear im Anteil, nur im Reiter „Lösungskonzepte“ hat der Spediteur die Krümmung a.
- **Teilspielperfektes Gleichgewicht** mit Annahme bei Gleichstand; im Rasterspiel wird der Gleichstand in drei Ebenen ausgewiesen (Antwortender: Regel A oder R, Anbieter: angenommenes Angebot bevorzugt, Wahl unter gleichen Angeboten: das kleinste).
- **Außenoption** hat nur der Antwortende der ersten Runde; angeboten sind die Varianten A (Erlös sofort nach der Ablehnung) und C (Erlös eine Runde später).

**Grenzen**

- **Keine Daten echter Verhandler:** die Experimente zeigen das Verhalten des Modells, nicht das von Menschen oder Betrieben.
- **Die Schranke 1/((1 − δ₁δ₂)·G)** für den Rasterfehler ist eine Hypothese aus 200 gemessenen Zellen, kein Satz; das Rasterspiel hat in den gemessenen Zellen mehrere stationäre Gleichgewichte, die Eindeutigkeit gehört zum stetigen Spiel.
- **Lernende Verhandler:** nur Hedge und Regret Matching in zwei kleinen Spielen (Ultimatum und zwei Runden auf dem Zehntelraster); andere Lernregeln, andere Spiele und Q-Learning sind nicht gemessen.
- **Maßstabsprobe:** nur für die Menge u₁ = x, u₂ = k·(1 − x)^a mit Drohpunkt (0, d₂) und d₂ = Anteil · k; keine allgemeinen Verhandlungsmengen.
- **Nicht gebaut:** Verhandlung mit privater Information und Verzögerung, Mehrparteien-Verhandlung, Hin- und Rückfahrt mit Flussungleichgewicht, Daten echter Verhandler.

**Bekannte Probleme und Unsicherheiten**

- **Quellen:** Nash 1950, Rubinstein 1982, Binmore, Rubinstein und Wolinsky 1986, Kalai und Smorodinsky 1975, Kalai 1977 und die übrige Literatur wurden für diese Demo nicht im Volltext gelesen; die Formeln stützen sich auf die Messungen und die unabhängigen Rechenwege der Tests.
- **Lernergebnisse** hängen an der Lernrate und, bei Regret Matching, am Zufalls-Seed; in den Tabellen stehen sie als Bänder über vier Zufalls-Seeds, der Lernlauf der App zeigt einen einzelnen Lauf mit seinem Zufalls-Seed.
- **Kalai–Smorodinsky und utilitaristisch** rechnen auf der Menge der Einigungen, in denen der Spediteur seinen Drohpunkt erreicht; bei Gleichstand des utilitaristischen Maßes gilt der kleinste Anteil des Verladers.
- **Nur mit Standard-Schnittstellen geprüft:** AppTest und Rechenwege; ein Durchlauf im Browser mit echten Mausklicks ist nicht Teil der Tests.
"""
