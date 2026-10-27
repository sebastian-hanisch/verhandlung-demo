"""Permalink (Begrenzen, Einrasten), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in der Seitenleiste und stehen in SETTING_SPECS: ihre Änderung hebt die Markierung des aktiven Presets auf (tests/test_layout.py prüft das für jeden Regler)."""

import random
from dataclasses import dataclass

import streamlit as st

import vhd_constants as C
from vhd_scenario import Settings


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "d1_slider": SettingSpec("d1", float, C.DEFAULT_D1, C.D_MIN, C.D_MAX, 0.01),
    "d2_slider": SettingSpec("d2", float, C.DEFAULT_D2, C.D_MIN, C.D_MAX, 0.01),
    "t_slider": SettingSpec("t", int, C.DEFAULT_T, C.T_MIN, C.T_MAX, 1),
    "first_radio": SettingSpec("first", int, C.DEFAULT_FIRST, choices=tuple(C.FIRSTS)),
    "g_select": SettingSpec("g", int, C.DEFAULT_G, choices=C.G_OPTIONS),
    "rule_radio": SettingSpec("rule", str, C.DEFAULT_RULE, choices=tuple(C.RULES)),
    "c1_slider": SettingSpec("c1", float, C.DEFAULT_C1, C.COST_MIN, C.COST_MAX, 0.01),
    "c2_slider": SettingSpec("c2", float, C.DEFAULT_C2, C.COST_MIN, C.COST_MAX, 0.01),
    "r1_slider": SettingSpec("r1", float, C.DEFAULT_R1, C.RATE_MIN, C.RATE_MAX, 0.1),
    "r2_slider": SettingSpec("r2", float, C.DEFAULT_R2, C.RATE_MIN, C.RATE_MAX, 0.1),
    "dt_select": SettingSpec("dt", float, C.DEFAULT_DT, choices=C.DT_OPTIONS),
    "o_slider": SettingSpec("o", float, C.DEFAULT_O, C.O_MIN, C.O_MAX, 0.01),
    "variant_radio": SettingSpec("variant", str, C.DEFAULT_VARIANT, choices=tuple(C.VARIANTS)),
    "curv_slider": SettingSpec("curv", float, C.DEFAULT_CURV, C.CURV_MIN, C.CURV_MAX, 0.05),
    "scale_select": SettingSpec("scale", float, C.DEFAULT_SCALE, choices=C.SCALE_OPTIONS),
    "threat_slider": SettingSpec("threat", float, C.DEFAULT_THREAT, C.THREAT_MIN, C.THREAT_MAX, 0.05),
    "game_select": SettingSpec("game", str, C.DEFAULT_GAME, choices=tuple(C.GAMES)),
    "learn_rule_select": SettingSpec("lrule", str, C.DEFAULT_LEARN_RULE, choices=tuple(C.LEARN_RULES)),
    "eta_select": SettingSpec("eta", float, C.DEFAULT_ETA, choices=C.ETA_OPTIONS),
    "rounds_slider": SettingSpec("rounds", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX, 1000),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX, 1),
}
# Schlüssel der Presets (Feldnamen von Settings) -> Schlüssel im Session State
PRESET_KEYS = {
    "d1": "d1_slider", "d2": "d2_slider", "t": "t_slider", "first": "first_radio", "g": "g_select", "rule": "rule_radio", "c1": "c1_slider", "c2": "c2_slider",
    "r1": "r1_slider", "r2": "r2_slider", "dt": "dt_select", "o": "o_slider", "variant": "variant_radio", "curv": "curv_slider", "scale": "scale_select", "threat": "threat_slider",
    "game": "game_select", "learn_rule": "learn_rule_select", "eta": "eta_select", "rounds": "rounds_slider", "seed": "seed_input",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clean_value(spec: SettingSpec, raw):
    """Wert aus der Adresszeile: umwandeln, auf Reglergrenzen begrenzen, auf die Stufen des Reglers einrasten; unbekannte Auswahl oder Unsinn gibt None."""
    try:
        value = spec.caster(raw)
    except (ValueError, TypeError):
        return None
    if spec.choices is not None:
        return value if value in spec.choices else None
    if isinstance(value, float) and value != value:                         # nan
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step and isinstance(value, float):
        value = round(round((value - spec.lo) / spec.step) * spec.step + spec.lo, 6)
        value = min(spec.hi, max(spec.lo, value))
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = clean_value(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    """Setzt ALLE Regler: die des Presets auf seine Werte, die übrigen auf ihren Standard (so bleibt die Hervorhebung des Presets nach dem Anklicken gültig)."""
    preset = C.PRESETS[name]
    key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = preset.get(key_of.get(state_key), spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


def settings_from_state() -> Settings:
    """Die Einstellungen aus dem Session State (nach den Reglern der Seitenleiste)."""
    values = {field: st.session_state.get(state_key, SETTING_SPECS[state_key].default) for field, state_key in PRESET_KEYS.items()}
    return Settings(**values)
