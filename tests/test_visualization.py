"""Abbildungen: jede baut für jedes Preset und an den Reglergrenzen, alle Achsen sind gesperrt, die Daten stimmen mit der Auswertung überein."""

import pytest

import vhd_constants as C
import vhd_evaluation as ev
import vhd_visualization as V
from vhd_scenario import Settings

PRESETS = [Settings(**p) for p in C.PRESETS.values()]
GRID = ev.load_grid()


def _locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def _all_figures(s: Settings, run: dict):
    a = ev.analyse(s)
    rows = ev.short_rounds_rows(s.r1, s.r2, C.DT_OPTIONS)
    return [V.build_rounds(a), V.build_patience(s), V.build_horizon(s), V.build_short_rounds(rows, s), V.build_costs(s), V.build_outside(s), V.build_solutions(s),
            V.build_scale(ev.scale_rows(s.curv, s.threat)), V.build_grid(s), V.build_learning(run), V.build_convergence(ev.error_curves(t_max=80))]


@pytest.fixture(scope="module")
def run():
    return ev.learn_once("zwei_runden", "rm", 0.0, 1000, 1)


@pytest.mark.parametrize("s", PRESETS, ids=list(C.PRESETS))
def test_every_figure_builds_for_every_preset_and_is_locked(s, run):
    for fig in _all_figures(s, run):
        assert len(fig.data) >= 1 and _locked(fig)


def test_figures_at_the_regler_limits(run):
    for s in (Settings(d1=C.D_MIN, d2=C.D_MIN, t=C.T_MIN, g=10), Settings(d1=C.D_MAX, d2=C.D_MAX, t=C.T_MAX, g=10000, first=2), Settings(c1=C.COST_MAX, c2=C.COST_MAX, o=C.O_MAX, variant="C", curv=C.CURV_MAX, scale=0.1)):
        for fig in _all_figures(s, run):
            assert _locked(fig)


def test_rounds_bars_are_the_backward_values():
    a = ev.analyse(Settings(d1=0.5, d2=0.5, t=3))
    fig = V.build_rounds(a)
    assert list(fig.data[0].y) == [0.75, 0.5, 1.0] and list(fig.data[0].x) == [1, 2, 3]
    assert list(fig.data[0].marker.color) == [C.COLOR_P1, C.COLOR_P2, C.COLOR_P1]
    swapped = V.build_rounds(ev.analyse(Settings(d1=0.5, d2=0.5, t=3, first=2)))
    assert list(swapped.data[0].marker.color) == [C.COLOR_P2, C.COLOR_P1, C.COLOR_P2]


def test_patience_marker_sits_on_the_curve():
    s = Settings(d1=0.9, d2=0.9)
    fig = V.build_patience(s)
    marker = fig.data[2]
    assert marker.x == (0.9,) and marker.y[0] == pytest.approx(10 / 19)


def test_horizon_marker_only_inside_the_plot_range():
    assert len(V.build_horizon(Settings(t=10)).data) == 2
    assert len(V.build_horizon(Settings(t=ev.HORIZON_PLOT + 1)).data) == 1


def test_costs_figure_has_a_discount_line_only_for_costs_below_one():
    assert len(V.build_costs(Settings(c1=0.05, c2=0.10)).data) == 2 and len(V.build_costs(Settings(c1=1.2, c2=0.3)).data) == 1


def test_scale_figure_has_one_group_per_scale_and_four_series():
    fig = V.build_scale(ev.scale_rows(0.5, 0.1))
    assert [t.name for t in fig.data] == ["Nash", "Kalai–Smorodinsky", "egalitär", "utilitaristisch"] and list(fig.data[0].x) == ["k = 1,0", "k = 10,0", "k = 0,1"]


@pytest.mark.skipif(GRID is None, reason="data/negotiation_grid.json fehlt")
def test_learning_bands_figure_has_a_point_per_run():
    fig = V.build_learning_bands(GRID["learn"])
    runs = sum(len(t.y) for t in fig.data if t.marker.symbol != "line-ew-open")
    assert runs == 32 and _locked(fig)


def test_convergence_figure_uses_a_log_axis():
    fig = V.build_convergence(ev.error_curves(t_max=50))
    assert fig.layout.yaxis.type == "log" and len(fig.data) == 4
