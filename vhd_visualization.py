"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import vhd_constants as C
import vhd_evaluation as ev
from vhd_scenario import Settings


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height: int = 340, ytitle: str = "Anteil des Verladers", yrange=(0, 1.02)):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text=ytitle, range=list(yrange), tickformat=".0%")
    return lock_axes(fig)


def _half(fig):
    fig.add_hline(y=0.5, line=dict(color=C.COLOR_HALF, dash="dot", width=1), annotation_text="Halbe-halbe", annotation_position="bottom right")


def build_rounds(a: ev.Analysis, height: int = 300):
    """Der Wert des Anbieters der Runde t in der Rückwärtsinduktion (Anteil am Überschuss): wer die Verhandlung in Runde t eröffnete, bekäme diesen Anteil."""
    s = a.settings
    rounds = list(range(1, len(a.values) + 1))
    proposer = [s.first if t % 2 == 1 else 3 - s.first for t in rounds]
    colors = [C.COLOR_P1 if p == 1 else C.COLOR_P2 for p in proposer]
    labels = [C.NAMES[p] for p in proposer]
    fig = go.Figure(go.Bar(x=rounds, y=list(a.values), marker_color=colors, text=[f"{ev.pct(v, 0)}" for v in a.values], textposition="outside", hovertext=labels, hoverinfo="text+y", showlegend=False))
    fig.update_xaxes(title_text="Runde t (Anbieter: blau Verlader, rot Spediteur)", dtick=1)
    return _base(fig, height, "Wert des Anbieters der Runde", (0, 1.1))


def build_patience(s: Settings, height: int = 340):
    """Anteil des Verladers über seiner Geduld δ₁ (δ₂ fest), unendlicher Horizont."""
    curve = ev.patience_curve(s.d2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["d1"], y=curve["verlader_first"], mode="lines", name="Verlader bietet zuerst", line=dict(color=C.COLOR_P1, width=3)))
    fig.add_trace(go.Scatter(x=curve["d1"], y=curve["spediteur_first"], mode="lines", name="Spediteur bietet zuerst", line=dict(color=C.COLOR_P2, width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=[s.d1], y=[curve["verlader_first"][round(s.d1 * 100)]], mode="markers", name="gewählte Geduld", marker=dict(color=C.COLOR_DARK, size=11, symbol="diamond")))
    _half(fig)
    fig.update_xaxes(title_text="Geduld des Verladers δ₁", range=[0, 1])
    return _base(fig, height)


def build_horizon(s: Settings, height: int = 340):
    """Anteil des Verladers für T = 1 … 60: ungerade Horizonte über, gerade unter dem Grenzwert."""
    curve = ev.horizon_curve(s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["T"], y=curve["x1"], mode="lines+markers", name="Rückwärtsinduktion im Spiel mit T Runden", line=dict(color=C.COLOR_P1), marker=dict(size=6)))
    fig.add_hline(y=curve["limit"], line=dict(color=C.COLOR_FORMULA, dash="dash", width=2), annotation_text="Formel (unendlicher Horizont)", annotation_position="top right")
    if s.t <= len(curve["T"]):
        fig.add_trace(go.Scatter(x=[s.t], y=[curve["x1"][s.t - 1]], mode="markers", name="gewähltes T", marker=dict(color=C.COLOR_DARK, size=12, symbol="diamond")))
    _half(fig)
    fig.update_xaxes(title_text="Horizont T (Runden)")
    return _base(fig, height)


def build_convergence(curves: dict, height: int = 340):
    """Fehler gegen die Formel über T (logarithmisch) für mehrere Paare (δ₁; δ₂), E1."""
    fig = go.Figure()
    palette = [C.COLOR_P1, C.COLOR_P2, C.COLOR_FORMULA, C.COLOR_ALT]
    for color, (label, errors) in zip(palette, curves["series"].items()):
        fig.add_trace(go.Scatter(x=curves["T"], y=errors, mode="lines", name=label, line=dict(color=color)))
    fig.add_hline(y=C.E1_TOL, line=dict(color=C.COLOR_HALF, dash="dot"), annotation_text="10⁻⁶", annotation_position="top left")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(type="log", title_text="|Anteil − Formel|", range=[-12, 0])
    fig.update_xaxes(title_text="Horizont T (Runden)")
    return lock_axes(fig)


def build_short_rounds(rows: list, s: Settings, height: int = 340):
    """Anteil des Verladers über der Rundenlänge Δ (logarithmisch): zuerst oder zuletzt bietend, dazu Grenzwert und Halbe-halbe."""
    dts = [r["dt"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dts, y=[r["x_first"] for r in rows], mode="lines+markers", name="Verlader bietet zuerst", line=dict(color=C.COLOR_P1)))
    fig.add_trace(go.Scatter(x=dts, y=[r["x_second"] for r in rows], mode="lines+markers", name="Spediteur bietet zuerst", line=dict(color=C.COLOR_P2, dash="dash")))
    fig.add_hline(y=rows[0]["limit"], line=dict(color=C.COLOR_FORMULA, dash="dash", width=2), annotation_text="Grenzwert r₂/(r₁ + r₂)", annotation_position="top right")
    _half(fig)
    fig.update_xaxes(type="log", title_text="Rundenlänge Δ in Tagen (rechts länger)", autorange="reversed")
    return _base(fig, height)


def build_costs(s: Settings, height: int = 340):
    """Anteil des Verladers für T = 1 … 40 bei festen Kosten und, wo möglich, bei Diskont d = 1 − c."""
    curve = ev.costs_curve(s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["T"], y=curve["costs"], mode="lines+markers", name="feste Kosten je Runde", line=dict(color=C.COLOR_P1), marker=dict(size=6)))
    if curve["discount"] is not None:
        fig.add_trace(go.Scatter(x=curve["T"], y=curve["discount"], mode="lines+markers", name="Diskont d = 1 − c", line=dict(color=C.COLOR_ALT, dash="dash"), marker=dict(size=5)))
    _half(fig)
    fig.update_xaxes(title_text="Horizont T (Runden)")
    return _base(fig, height)


def build_outside(s: Settings, height: int = 340):
    """Anteil des Halters über der Außenoption o: Variante A, Variante C, Nash-Lösung mit Drohpunkt o."""
    curve = ev.outside_curve(s)
    st = ev.outside_state(s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["o"], y=curve["A"], mode="lines", name="Variante A (Erlös sofort)", line=dict(color=C.COLOR_P1, width=3)))
    fig.add_trace(go.Scatter(x=curve["o"], y=curve["C"], mode="lines", name="Variante C (Erlös eine Runde später)", line=dict(color=C.COLOR_P2, dash="dash")))
    fig.add_trace(go.Scatter(x=curve["o"], y=curve["nash"], mode="lines", name="Nash-Lösung mit Drohpunkt o", line=dict(color=C.COLOR_FORMULA, dash="dot")))
    fig.add_trace(go.Scatter(x=[s.o], y=[st["with"]], mode="markers", name=f"gewählt, T = {s.t}", marker=dict(color=C.COLOR_DARK, size=12, symbol="diamond")))
    fig.add_hline(y=curve["without"], line=dict(color=C.COLOR_HALF, dash="dot", width=1), annotation_text="ohne Option", annotation_position="bottom right")
    fig.update_xaxes(title_text="Außenoption o", range=[0, 1])
    return _base(fig, height, "Anteil des Halters (Antwortender der ersten Runde)")


def build_solutions(s: Settings, height: int = 320):
    """Anteil des Verladers nach den fünf Lösungskonzepten bei den gewählten Einstellungen."""
    sol = ev.solutions_for(s)
    names = {"nash": "Nash", "ks": "Kalai–Smorodinsky", "egalitarian": "egalitär", "utilitarian": "utilitaristisch", "half": "Halbe-halbe"}
    fig = go.Figure(go.Bar(x=[names[k] for k in sol], y=list(sol.values()), marker_color=[C.COLOR_P1, C.COLOR_FORMULA, C.COLOR_ALT, C.COLOR_P2, C.COLOR_HALF], text=[ev.pct(v) for v in sol.values()], textposition="outside"))
    return _base(fig, height, "Anteil des Verladers", (0, 1.12))


def build_scale(rows: list, height: int = 320):
    """E6: alle Lösungen für k = 0,1; 1; 10 nebeneinander (Anteil des Verladers)."""
    names = {"nash": "Nash", "ks": "Kalai–Smorodinsky", "egalitarian": "egalitär", "utilitarian": "utilitaristisch"}
    fig = go.Figure()
    for color, (key, label) in zip([C.COLOR_P1, C.COLOR_FORMULA, C.COLOR_ALT, C.COLOR_P2], names.items()):
        fig.add_trace(go.Bar(x=[f"k = {ev.de(r['k'], 1)}" for r in rows], y=[r[key] for r in rows], name=label, marker_color=color))
    fig.update_layout(barmode="group")
    return _base(fig, height, "Anteil des Verladers", (0, 1.05))


def build_grid(s: Settings, height: int = 340):
    """Anteil des Verladers im Rasterspiel und im stetigen Spiel über T = 1 … 60."""
    curve = ev.grid_curve(s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["T"], y=curve["continuous"], mode="lines+markers", name="stetige Angebote", line=dict(color=C.COLOR_P1), marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=curve["T"], y=curve["grid"], mode="lines+markers", name=f"Raster G = {s.g}, Regel {s.rule}", line=dict(color=C.COLOR_ALT, dash="dash"), marker=dict(size=5)))
    fig.add_hline(y=curve["limit"], line=dict(color=C.COLOR_FORMULA, dash="dot", width=2), annotation_text="Formel", annotation_position="top right")
    fig.update_xaxes(title_text="Horizont T (Runden)")
    return _base(fig, height)


def build_learning(run: dict, height: int = 320):
    """Verlauf eines Lernlaufs: erwarteter Anteil von Spieler 1 unter den aktuellen Strategien und Nash-Lücke, dazu das teilspielperfekte Ergebnis."""
    res = run["result"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(res.path_rounds), y=list(res.path_share1), mode="lines", name="erwarteter Anteil von Spieler 1", line=dict(color=C.COLOR_P1, width=3)))
    fig.add_trace(go.Scatter(x=list(res.path_rounds), y=list(res.path_gap), mode="lines", name="Nash-Lücke", line=dict(color=C.COLOR_P2, dash="dash")))
    fig.add_hline(y=run["spe"], line=dict(color=C.COLOR_FORMULA, dash="dot", width=2), annotation_text="teilspielperfekt", annotation_position="top right")
    fig.update_xaxes(title_text="Lernrunden")
    return _base(fig, height, "Anteil bzw. Lücke", (-0.02, 1.05))


def build_learning_bands(cells: list, height: int = 340):
    """E7: Anteil des Anbieters in den vier Läufen je Zelle (Punkte) und das teilspielperfekte Ergebnis je Spiel (Linie)."""
    fig = go.Figure()
    labels = []
    for cell in cells:
        label = f"{cell['game'].replace('zwei_runden', '2 Runden')} · {cell['rule']}" + (f" η₀ = {ev.de(cell['eta'], 1)}" if cell["rule"] == "hedge" else "")
        labels.append(label)
        fig.add_trace(go.Scatter(x=[label] * len(cell["shares"]), y=cell["shares"], mode="markers", marker=dict(color=C.COLOR_P1, size=9, opacity=0.6), showlegend=False,
                                 hovertemplate="Anteil %{y:.0%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[label], y=[cell["spe"]], mode="markers", marker=dict(color=C.COLOR_FORMULA, size=13, symbol="line-ew-open", line=dict(width=3)), showlegend=False,
                                 hovertemplate="teilspielperfekt %{y:.0%}<extra></extra>"))
    fig.update_xaxes(tickangle=-30)
    return _base(fig, height, "Anteil von Spieler 1 (blau: Läufe, grün: teilspielperfekt)")
