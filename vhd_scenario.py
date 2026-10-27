"""Szenario: die Einstellungen der Verhandlung (ein unveränderlicher, hashbarer Satz Werte, damit st.cache_data ihn als Schlüssel nehmen kann) und die Umrechnung in Rollen.

Spieler 1 ist der Verlader, Spieler 2 der Spediteur. `first` sagt, wer in Runde 1 bietet. Alle Rechnungen der Module arbeiten mit „erster Anbieter“ und „zweiter Anbieter“;
`ordered_patience` und `to_player1` übersetzen zwischen beiden Sichten."""

from dataclasses import dataclass

import vhd_constants as C


@dataclass(frozen=True)
class Settings:
    d1: float = C.DEFAULT_D1
    d2: float = C.DEFAULT_D2
    t: int = C.DEFAULT_T
    first: int = C.DEFAULT_FIRST
    g: int = C.DEFAULT_G
    rule: str = C.DEFAULT_RULE
    c1: float = C.DEFAULT_C1
    c2: float = C.DEFAULT_C2
    r1: float = C.DEFAULT_R1
    r2: float = C.DEFAULT_R2
    dt: float = C.DEFAULT_DT
    o: float = C.DEFAULT_O
    variant: str = C.DEFAULT_VARIANT
    curv: float = C.DEFAULT_CURV
    scale: float = C.DEFAULT_SCALE
    threat: float = C.DEFAULT_THREAT
    game: str = C.DEFAULT_GAME
    learn_rule: str = C.DEFAULT_LEARN_RULE
    eta: float = C.DEFAULT_ETA
    rounds: int = C.DEFAULT_ROUNDS
    seed: int = C.DEFAULT_SEED


def ordered_patience(s: Settings) -> tuple:
    """(Geduld des ersten Anbieters, Geduld des zweiten Anbieters)."""
    return (s.d1, s.d2) if s.first == 1 else (s.d2, s.d1)


def ordered_costs(s: Settings) -> tuple:
    """(Kosten je Runde des ersten Anbieters, des zweiten Anbieters)."""
    return (s.c1, s.c2) if s.first == 1 else (s.c2, s.c1)


def to_player1(first: int, share_of_first_bidder: float) -> float:
    """Anteil des Verladers (Spieler 1), wenn der erste Anbieter `share_of_first_bidder` bekommt und Spieler `first` zuerst bietet."""
    return share_of_first_bidder if first == 1 else 1 - share_of_first_bidder


def holder(first: int) -> int:
    """Wer die Außenoption hält: der Antwortende der ersten Runde, also der andere als `first`."""
    return 2 if first == 1 else 1


def other(player: int) -> int:
    return 2 if player == 1 else 1
