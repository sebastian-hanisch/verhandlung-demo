"""App: jedes Preset, der Permalink, der ▶️-Knopf der Lernläufe, die Reglermaxima und die Verdikt-Zeile je Preset (DEMO-PLAYBOOK §1d und §11a Punkt 7)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import vhd_constants as C
import vhd_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning] + [c.value for c in at.caption])


def _expander(at, label):
    return next(e for e in at.expander if e.label == label)               # nie per Index (DEMO-PLAYBOOK §2)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_sets_its_values(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == C.PRESETS[name][key], (name, key)
    assert len(at.get("plotly_chart")) == 11 and len(at.tabs) == 8


def test_the_two_expanders_and_the_footer_are_found_by_title():
    at = _app()
    assert _expander(at, "📐 Mathematische Formulierung") and _expander(at, "⚠️ Annahmen, Grenzen und bekannte Probleme") and _expander(at, "So funktioniert Verhandeln mit abwechselnden Angeboten")


def test_permalink_is_loaded_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["d1"] = "5"
    at.query_params["t"] = "99999"
    at.query_params["g"] = "77"                                  # ungültige Auswahl: wird ignoriert
    at.query_params["rule"] = "R"
    at.query_params["dt"] = str(1 / 24)
    at.query_params["o"] = "0.437"
    at.run()
    assert not at.exception
    assert at.session_state["d1_slider"] == C.D_MAX and at.session_state["t_slider"] == C.T_MAX and at.session_state["g_select"] == C.DEFAULT_G
    assert at.session_state["rule_radio"] == "R" and at.session_state["dt_select"] == 1 / 24 and at.session_state["o_slider"] == 0.44


def test_metrics_follow_the_first_bidder():
    at = _app()
    assert [m.label for m in at.metric] == ["Verlader (Spieler 1)", "Spediteur (Spieler 2)", "Formel für den Verlader", "Raster (Verlader)"]
    assert at.metric[0].value == "52,6 %" and at.metric[1].value == "47,4 %" and at.metric[3].value == "50,0 %"
    at.sidebar.radio(key="first_radio").set_value(2).run()
    assert not at.exception and at.metric[0].value == "47,4 %" and at.metric[1].value == "52,6 %"


def test_the_learning_button_runs_each_game_and_each_rule():
    at = _app()
    assert "Noch kein Lauf mit diesen Einstellungen" in _texts(at)
    for game in C.GAMES:
        for rule in C.LEARN_RULES:
            at.sidebar.selectbox(key="game_select").set_value(game)
            at.sidebar.selectbox(key="learn_rule_select").set_value(rule)
            at.sidebar.slider(key="rounds_slider").set_value(2000).run()
            assert "Noch kein Lauf mit diesen Einstellungen" in _texts(at)
            at.button(key="learn_start").click().run()
            assert not at.exception, [str(e) for e in at.exception]
            text = _texts(at)
            assert f"{C.LEARN_RULES[rule]}, {C.GAMES[game]}, 2000 Runden" in text and "Rechenzeit dieses Laufs" in text
            assert len(at.get("plotly_chart")) == 12


def test_the_learning_result_disappears_when_the_settings_change():
    at = _app()
    at.sidebar.slider(key="rounds_slider").set_value(1000).run()
    at.button(key="learn_start").click().run()
    assert "Rechenzeit dieses Laufs" in _texts(at)
    at.sidebar.number_input(key="seed_input").set_value(36).run()
    assert "Rechenzeit dieses Laufs" not in _texts(at) and "Noch kein Lauf" in _texts(at)


def test_the_slider_maxima_run_through_the_whole_app():
    at = _app()
    at.sidebar.slider(key="t_slider").set_value(C.T_MAX)
    at.sidebar.select_slider(key="g_select").set_value(10000)
    at.sidebar.slider(key="d1_slider").set_value(C.D_MAX)
    at.sidebar.slider(key="d2_slider").set_value(C.D_MAX).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "T = 2000" in _texts(at) and "Raster G = 10000" in _texts(at)


def test_the_maximum_of_the_learning_rounds_runs():
    at = _app()
    at.sidebar.selectbox(key="game_select").set_value("zwei_runden")
    at.sidebar.slider(key="rounds_slider").set_value(C.ROUNDS_MAX).run()
    at.button(key="learn_start").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "50000 Runden" in _texts(at)


def test_every_regler_minimum_and_maximum_does_not_raise():
    for state_key, spec in P.SETTING_SPECS.items():
        values = list(spec.choices) if spec.choices is not None else [spec.lo, spec.hi]
        for value in values:
            at = _app()
            widget = next(w for art in ("slider", "select_slider", "radio", "selectbox", "number_input") for w in getattr(at.sidebar, art) if w.key == state_key)
            widget.set_value(value).run()
            assert not at.exception, (state_key, value, [str(e) for e in at.exception])


def test_all_minima_and_all_maxima_together():
    for pick in (min, max):
        at = _app()
        for state_key, spec in P.SETTING_SPECS.items():
            if spec.choices is None:
                widget = next(w for art in ("slider", "number_input") for w in getattr(at.sidebar, art) if w.key == state_key)
                widget.set_value(spec.lo if pick is min else spec.hi)
        at.run()
        assert not at.exception, (pick.__name__, [str(e) for e in at.exception])


# --- Verdikt-Zeilen je Preset (§11a Punkt 7): ein Preset ohne Effekt meldet keinen Effekt --------------------------------------------------------

EXPECTED = {
    "Gleiche Geduld (Voreinstellung)": ["Der Verlader bekommt 52,6 %", "Die Option ändert nichts", "Gleiche Kosten"],
    "Geduld ist Macht": ["Der Verlader bekommt 55,2 %", "eine Stufe mehr (δ₁ = 0,92)"],
    "Ultimatum (eine Runde)": ["Eine Runde (Ultimatum)", "bekommt 90,00 %", "T = 1 ist das Ultimatum"],
    "Kurze Runden": ["bekommt er 68,05 %", "Runden von 1 Stunde"],
    "Feste Kosten": ["Verlader hat die kleineren Kosten (0,05 < 0,10)", "der erste Anbieter bekommt 100,0 %"],
    "Außenoption unter der Schwelle": ["Die Option ändert nichts"],
    "Außenoption über der Schwelle": ["Die Option zählt: der Spediteur bekommt 60,0 %"],
    "Risikoscheu": ["Nash 62,6 %"],
    "Rasterfalle": ["bekommt 10,00 %", "9 (Anteil des ersten Anbieters von 10,0 % bis 90,0 %)"],
}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_each_preset_shows_the_expected_verdict_lines(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    text = _texts(at)
    for expected in EXPECTED[name]:
        assert expected in text, (name, expected)


def test_a_preset_without_effect_reports_no_effect():
    at = _app()
    at.sidebar.button(key="preset_Außenoption unter der Schwelle").click().run()
    text = _texts(at)
    assert "Die Option ändert nichts" in text and "Die Option zählt" not in text


def test_the_pdf_button_creates_the_download_and_it_goes_away_when_the_settings_change():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1
    at.sidebar.slider(key="t_slider").set_value(201).run()
    assert not at.get("download_button")
