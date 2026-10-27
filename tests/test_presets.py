"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, jedes Preset ist unterscheidbar; Permalink-Bereinigung."""

import dataclasses

import pytest

import vhd_constants as C
import vhd_presets as P
from vhd_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_keys_are_exactly_the_fields_of_settings():
    assert set(P.PRESET_KEYS) == {f.name for f in dataclasses.fields(Settings)}
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_preset_values_are_inside_the_regler_bounds_and_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec, value = P.SETTING_SPECS[state_key], preset[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key, value)
            else:
                assert spec.lo <= value <= spec.hi, (name, key, value)


def test_default_preset_equals_the_default_settings():
    assert Settings(**C.PRESETS["Gleiche Geduld (Voreinstellung)"]) == Settings()
    assert next(iter(C.PRESETS)) == "Gleiche Geduld (Voreinstellung)"


def test_every_other_preset_differs_from_the_default_and_from_each_other():
    default = Settings()
    seen = {default}
    for name, preset in list(C.PRESETS.items())[1:]:
        s = Settings(**preset)
        assert s != default and s not in seen, name           # sonst hebt aktives_preset zwei Presets zugleich hervor und das spätere wäre nie erkennbar
        seen.add(s)


def test_every_preset_changes_the_settings_it_describes():
    default = C.PRESETS["Gleiche Geduld (Voreinstellung)"]
    changed = {name: {k for k in p if p[k] != default[k]} for name, p in C.PRESETS.items()}
    assert changed["Geduld ist Macht"] == {"d1"} and changed["Kurze Runden"] == {"dt"} and changed["Feste Kosten"] == {"c1"} and changed["Risikoscheu"] == {"curv"} and changed["Rasterfalle"] == {"g"}
    assert changed["Ultimatum (eine Runde)"] == {"t", "g", "rule"} and changed["Außenoption unter der Schwelle"] == {"o"} and changed["Außenoption über der Schwelle"] == {"o"}


def test_sliders_with_steps_have_their_default_on_the_grid():
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.step and spec.lo is not None:
            assert abs((spec.default - spec.lo) / spec.step - round((spec.default - spec.lo) / spec.step)) < 1e-9, state_key


def test_dt_options_are_unique_and_labelled():
    assert len(set(C.DT_OPTIONS)) == len(C.DT_OPTIONS) and set(C.DT_LABELS) == set(C.DT_OPTIONS)


@pytest.mark.parametrize("state_key,raw,expected", [
    ("d1_slider", "5", C.D_MAX), ("d1_slider", "-3", C.D_MIN), ("d1_slider", "0.917", 0.92), ("d1_slider", "abc", None), ("d1_slider", "nan", None),
    ("t_slider", "99999", C.T_MAX), ("t_slider", "0", C.T_MIN), ("t_slider", "x", None),
    ("g_select", "100", 100), ("g_select", "77", None), ("rule_radio", "R", "R"), ("rule_radio", "Z", None), ("first_radio", "2", 2), ("first_radio", "3", None),
    ("dt_select", str(1 / 24), 1 / 24), ("dt_select", "0.3", None), ("variant_radio", "C", "C"), ("variant_radio", "B", None),
    ("r1_slider", "1.23", 1.2), ("curv_slider", "0.1", C.CURV_MIN), ("rounds_slider", "123456", C.ROUNDS_MAX), ("scale_select", "10", 10.0), ("seed_input", "-5", 0),
])
def test_permalink_values_are_cleaned(state_key, raw, expected):
    assert P.clean_value(P.SETTING_SPECS[state_key], raw) == expected


def test_permalink_value_of_every_regler_round_trips():
    for state_key, spec in P.SETTING_SPECS.items():
        assert P.clean_value(spec, str(spec.default)) == spec.default, state_key
