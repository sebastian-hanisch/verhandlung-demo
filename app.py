"""Verhandlung: wer Zeit hat, bekommt mehr - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei Betriebe (Verlader und Spediteur) teilen den Überschuss einer Tour und machen sich abwechselnd Angebote. Die Demo rechnet das Rubinstein-Gleichgewicht (Formel gegen Rückwärtsrechnen),
den Einfluss von Geduld, Horizont, Rundenlänge, festen Kosten, Außenoption und Angebotsraster, vergleicht Lösungskonzepte (mit Maßstabsprobe) und lässt Lernregeln gegeneinander spielen.

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import vhd_constants as C
import vhd_evaluation as ev
from vhd_layout import dunkles_design, kopf_css
from vhd_pdf_export import build_pdf
from vhd_presets import (
    SETTING_SPECS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    settings_from_state,
    sidebar_presets,
    sync_query_params,
)
from vhd_visualization import (
    build_convergence,
    build_costs,
    build_grid,
    build_horizon,
    build_learning,
    build_learning_bands,
    build_outside,
    build_patience,
    build_rounds,
    build_scale,
    build_short_rounds,
    build_solutions,
)

st.set_page_config(page_title="Verhandlung – Sebastian Hanisch", layout="wide")

de = ev.de
pct = ev.pct


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


@st.cache_data(show_spinner=False)
def _short_rows(r1, r2):
    return ev.short_rounds_rows(r1, r2, C.DT_OPTIONS)


@st.cache_data(show_spinner=False)
def _convergence_curves():
    return ev.error_curves()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in vhd_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🤝 Verhandlung: wer Zeit hat, bekommt mehr")
st.markdown(
    """
**Ein Verlader und ein Spediteur teilen den Überschuss einer Tour** (Wert für den Verlader minus Kosten des Spediteurs, hier auf 1 normiert; die Frachtrate ist dann Kosten plus Anteil mal Überschuss). Sie machen sich
**abwechselnd Angebote**, und jede Runde Warten kostet beide etwas. Die Fragen: **Was bestimmt die Aufteilung, wenn man sich über Runden hinweg einigt? Was ist Geduld wert? Und wie ändern Horizont, Rundenlänge, feste
Kosten, eine Außenoption oder ein grobes Angebotsraster das Ergebnis?** Die Demo gehört zur Linie zu Verhandlung, Auktion und Vertrag; Shapley und Kern teilen einen Gewinn ohne Ablauf, die
Myerson–Satterthwaite-Demo hat private Werte, hier kennen beide alles.

Weiter unten: acht Reiter mit Experimenten (**🔬**), **🚧 Wo die Annahmen enden**, der Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und der Expander **📐 Mathematische Formulierung**.
**Eigennutz, Geduld und Nutzenfunktionen sind Modellannahmen, keine Aussagen über echte Verhandler.**
"""
)

with st.expander("So funktioniert Verhandeln mit abwechselnden Angeboten", expanded=True):
    st.markdown(
        r"""
Alle Kennzahlen sind **Anteile am Überschuss der Tour** (Bezugsmenge: der ganze Überschuss = 100 %).

| Begriff | Was er hier bedeutet |
|---|---|
| **Abwechselnde Angebote, Horizont T** | In Runde 1 bietet der erste Anbieter, in Runde 2 der zweite usw.; wer antwortet, nimmt an oder lehnt ab. Nach T Runden ist Schluss (T = 1: **Ultimatum**). Im Gleichgewicht einigen sich beide sofort in Runde 1; die späteren Runden bestimmen den Preis dafür. |
| **Geduld δ, Diskontfaktor** | Eine Einigung in Runde t zählt mit δ^(t−1): je kleiner δ, desto mehr kostet das Warten. **Feste Kosten** c ziehen stattdessen c je Runde ab. |
| **Rubinstein-Gleichgewicht** | Das teilspielperfekte Gleichgewicht bei unendlichem Horizont: der erste Anbieter bekommt (1 − δ₂)/(1 − δ₁δ₂). Mehr Geduld des ersten Anbieters vergrößert seinen Anteil, mehr Geduld des anderen verkleinert ihn. |
| **Rückwärtsrechnen** | Die letzte Runde zuerst: dort bekommt der Anbieter alles; davor muss er dem Antwortenden so viel bieten, wie dieser als Anbieter der nächsten Runde (nach Abzug der Wartekosten) bekäme. |
| **Raster G, Gleichstand** | Angebote nur in Schritten von 1/G. Bei genau gleichem Wert nimmt der Antwortende an (Regel A) oder lehnt ab (Regel R); dazu zieht der Anbieter bei gleicher Auszahlung das angenommene Angebot vor. |
| **Außenoption o** | Der Antwortende der ersten Runde kann aussteigen und bekommt o. Sie zählt erst, wenn o über seinem Rubinstein-Anteil liegt (Variante A); dann bekommt er genau o. |
| **Nash-Lösung** | Die Aufteilung, die das Produkt der Nutzengewinne gegenüber dem Drohpunkt maximiert. Für Runden, die gegen null gehen, ist sie (ohne Drohpunkt, mit den Gewichten 1/Verlustrate) der Grenzwert des Rubinstein-Gleichgewichts. |
| **Kalai–Smorodinsky** | Die Aufteilung, in der beide denselben Anteil ihres Idealgewinns bekommen (Idealgewinn: das Beste, was ihnen die Einigungen erlauben). |
| **Egalitär, utilitaristisch** | Egalitär: gleiche Gewinne. Utilitaristisch: größte Summe der Nutzen. Beide hängen am **Maßstab**, in dem man den Nutzen misst; Nash und Kalai–Smorodinsky nicht (Experiment 6). |
| **Halbe-halbe (Shapley-Wert)** | Bei zwei Spielern teilt der Shapley-Wert den Überschuss in der Mitte, ohne Ablauf, Geduld oder Nutzenform. |
| **Nash- und teilspielperfektes Gleichgewicht** | Ein Nash-Gleichgewicht darf mit unglaubwürdigen Drohungen arbeiten („ich lehne alles unter 90 % ab“); teilspielperfekt verlangt, dass jede Drohung in jeder Runde auch ausgeführt würde. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Geduld und Ablauf")
    st.slider("Geduld des Verladers δ₁", *bounds("d1_slider"), step=0.01, key="d1_slider", format="%.2f", help="Diskontfaktor je Runde: 0,9 heißt, eine Runde später zählt der Anteil nur noch 90 %. Wirkt in der Hauptansicht und in den Reitern Geduld, Zeit, Außenoption und Raster. Höchstwert 0,99: bei 0,999 reichten 2 000 Runden nicht zur Konvergenz.")
    st.slider("Geduld des Spediteurs δ₂", *bounds("d2_slider"), step=0.01, key="d2_slider", format="%.2f", help="Diskontfaktor des Spediteurs; wirkt wie δ₁.")
    st.slider("Horizont T (Runden)", *bounds("t_slider"), step=1, key="t_slider", help="Nach T Runden ist Schluss; T = 1 ist das Ultimatum. Wirkt in der Hauptansicht und in den Reitern Zeit, Feste Kosten, Außenoption und Raster.")
    st.radio("Wer bietet zuerst", list(C.FIRSTS), format_func=C.FIRSTS.get, key="first_radio", help="Der erste Anbieter bietet in den ungeraden Runden. Wirkt in der Hauptansicht und in den Reitern Zeit, Feste Kosten, Außenoption und Raster; die Reiter Geduld und Kurze Runden zeigen beide Reihenfolgen.")
    st.header("🔲 Raster")
    st.select_slider("Angebotsraster G (Schritte 1/G)", options=C.G_OPTIONS, key="g_select", help="Angebote nur als Vielfache von 1/G. Wirkt im Reiter Raster und Gleichstand.")
    st.radio("Gleichstand beim Antworten", list(C.RULES), format_func=C.RULES.get, key="rule_radio", help="Was der Antwortende tut, wenn ihm das Angebot genau so viel bringt wie Ablehnen. Wirkt im Reiter Raster und Gleichstand.")
    st.header("💸 Feste Kosten")
    st.slider("Kosten je Runde, Verlader c₁", *bounds("c1_slider"), step=0.01, key="c1_slider", format="%.2f", help="Jede Runde Warten kostet den Verlader c₁ vom Überschuss. Wirkt im Reiter Feste Kosten.")
    st.slider("Kosten je Runde, Spediteur c₂", *bounds("c2_slider"), step=0.01, key="c2_slider", format="%.2f", help="Kosten je Runde des Spediteurs. Wirkt im Reiter Feste Kosten.")
    st.header("⏱️ Kurze Runden")
    st.slider("Verlustrate des Verladers r₁ (je Tag)", *bounds("r1_slider"), step=0.1, key="r1_slider", format="%.1f", help="Geduld als Rate: δ = exp(−r·Δ) bei Rundenlänge Δ in Tagen. Wirkt im Reiter Kurze Runden.")
    st.slider("Verlustrate des Spediteurs r₂ (je Tag)", *bounds("r2_slider"), step=0.1, key="r2_slider", format="%.1f", help="Verlustrate des Spediteurs. Wirkt im Reiter Kurze Runden.")
    st.select_slider("Rundenlänge Δ", options=C.DT_OPTIONS, format_func=lambda v: C.DT_LABELS[v], key="dt_select", help="Wie lange eine Runde dauert. Wirkt im Reiter Kurze Runden.")
    st.header("🚪 Außenoption")
    st.slider("Wert der Außenoption o", *bounds("o_slider"), step=0.01, key="o_slider", format="%.2f", help="Was der Antwortende der ersten Runde bekäme, wenn er aussteigt (Anteil am Überschuss). Wirkt im Reiter Außenoption.")
    st.radio("Variante der Außenoption", list(C.VARIANTS), format_func=C.VARIANTS.get, key="variant_radio", help="A: der Erlös o kommt sofort nach der Ablehnung; C: eine Runde später (zählt δ·o). Variante B ist nicht angeboten: sie ist in 2 400 von 2 400 gemessenen Zellen gleich A. Wirkt im Reiter Außenoption.")
    st.header("⚖️ Nutzen und Lösungskonzepte")
    st.slider("Krümmung des Nutzens des Spediteurs a", *bounds("curv_slider"), step=0.05, key="curv_slider", format="%.2f", help="u₂ = k·(1 − x)^a: a < 1 konkav (in der Nash-Lösung risikoscheu, im Rubinstein-Spiel wie größere Ungeduld), a = 1 linear, a > 1 nicht konvexe Menge. Wirkt im Reiter Lösungskonzepte.")
    st.select_slider("Maßstab k des Nutzens des Spediteurs", options=C.SCALE_OPTIONS, key="scale_select", help="Der Nutzen des Spediteurs wird mit k multipliziert (der Drohpunkt mit). Wirkt im Reiter Lösungskonzepte.")
    st.slider("Drohpunkt des Spediteurs (Anteil von k)", *bounds("threat_slider"), step=0.05, key="threat_slider", format="%.2f", help="Was der Spediteur ohne Einigung bekäme, als Anteil von k; der Verlader hat 0. Wirkt im Reiter Lösungskonzepte.")
    st.header("🧪 Lernende Verhandler")
    st.selectbox("Spiel", list(C.GAMES), format_func=C.GAMES.get, key="game_select", help="Ultimatum: Angebot und Annahmeschwelle in Zehnteln. Zwei Runden: dazu Gegenangebot und Schwelle für Runde 2. Wirkt im Reiter Lernende Verhandler.")
    st.selectbox("Lernregel", list(C.LEARN_RULES), format_func=C.LEARN_RULES.get, key="learn_rule_select", help="Hedge gewichtet nach der Summe der bisherigen Auszahlungen, Regret Matching nach den positiven Bedauern. Wirkt im Reiter Lernende Verhandler.")
    st.select_slider("Lernrate η₀ (nur Hedge)", options=C.ETA_OPTIONS, key="eta_select", help="Hedge lernt mit η₀/√t; bei Regret Matching ohne Wirkung (dort wird die Rate nicht benutzt). Wirkt im Reiter Lernende Verhandler.")
    st.slider("Lernrunden", *bounds("rounds_slider"), step=1000, key="rounds_slider", help="Die Rechenzeit wächst linear mit den Runden. Wirkt im Reiter Lernende Verhandler.")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

settings = settings_from_state()
sync_query_params({key: st.session_state[key] for key in SETTING_SPECS})

a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“ (Vorzeichen-Falle, DEMO-PLAYBOOK §2); der Anteil ist keine Güte, deshalb keine Farbe.
m1.metric("Verlader (Spieler 1)", pct(a.x1), delta=f"{ev.pp(a.x1 - 0.5)} gegen Halbe-halbe", delta_color="off", help="Anteil am Überschuss der Tour bei Horizont T mit den Einstellungen der Seitenleiste.")
m2.metric("Spediteur (Spieler 2)", pct(a.x2), delta=f"{ev.pp(a.x2 - 0.5)} gegen Halbe-halbe", delta_color="off")
m3.metric("Formel für den Verlader", pct(a.limit_x1), delta=f"Abstand bei T = {settings.t}: {ev.sci(abs(a.x1 - a.limit_x1)) if a.x1 != a.limit_x1 else '0'}", delta_color="off", help="Rubinstein-Formel (1 − δ₂)/(1 − δ₁δ₂) für den ersten Anbieter, bei unendlichem Horizont.")
grid_state = ev.grid_state(settings)
m4.metric("Raster (Verlader)", pct(ev.to_player1(settings.first, grid_state["share_first"])), delta=f"G = {settings.g}, Regel {settings.rule}", delta_color="off", help="Anteil des Verladers im Rasterspiel mit Angeboten in Schritten von 1/G.")
st.plotly_chart(build_rounds(a), width="stretch", key="main_rounds")
st.caption(f"Rückwärtsrechnen: der Balken der Runde t ist der Anteil, den der Anbieter dieser Runde bekäme, wenn die Verhandlung dort begänne (bis Runde {len(a.values)}). Runde 1 ist das Ergebnis; "
           "die Balken wechseln, weil jeder dem Antwortenden das gibt, was dieser als nächster Anbieter nach Abzug der Wartekosten bekäme.")
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="verhandlung-demo.pdf", mime="application/pdf", key="pdf_download")

st.markdown("## ❓ Kernfrage: Wer bekommt wie viel, und was kostet Warten?")
kind, text = ev.verdict_main(a)
getattr(st, kind)(text)

st.markdown("---")
st.caption("Jeder Reiter nennt, welche Regler der Seitenleiste in ihm wirken; die Hauptansicht oben folgt Geduld, Horizont, Reihenfolge, Raster und Gleichstand.")

grid = _grid()
if grid is None:
    st.warning("Die vorgerechneten Messreihen (data/negotiation_grid.json) fehlen; sie entstehen mit `python tools/make_grid.py`. Die Experimente zeigen sie.")

tab_names = ["⏳ Geduld ist Macht", "🔄 Zeit und Horizont", "⚡ Kurze Runden", "💸 Feste Kosten", "🚪 Außenoption", "⚖️ Lösungskonzepte", "🔲 Raster und Gleichstand", "🧪 Lernende Verhandler"]
tabs = st.tabs(tab_names)

# --- Reiter 1: Geduld ist Macht ------------------------------------------------------------------------------------------------------------

with tabs[0]:
    st.caption("Wirkt: Geduld des Spediteurs δ₂ (feste Kurve) und des Verladers δ₁ (Marke); Verlader bietet zuerst, unendlicher Horizont.")
    st.plotly_chart(build_patience(settings), width="stretch", key="patience_chart")
    st.caption("Blau: Anteil des Verladers, wenn er zuerst bietet; rot gestrichelt: wenn der Spediteur zuerst bietet; grau gepunktet: Halbe-halbe (Shapley-Wert), der Geduld nicht kennt.")
    kind, text = ev.verdict_patience(settings)
    getattr(st, kind)(text)
    st.subheader("🔬 Experiment 0: Der Wert der Geduld wächst")
    st.dataframe([{"Geduld des Verladers δ₁": de(r["d1"], 2), "Verlader zuerst": pct(r["first"], 2), "Spediteur zuerst": pct(r["second"], 2), "Halbe-halbe": "50,0 %", "Steigung dx₁/dδ₁": de(r["slope"], 2)}
                  for r in ev.patience_rows(settings.d2)], hide_index=True, width="stretch")
    st.caption(f"Bei δ₂ = {de(settings.d2, 2)}: je geduldiger der Verlader schon ist, desto mehr bringt eine weitere Stufe. Der Shapley-Wert bleibt in jeder Zeile bei 50,0 %.")

# --- Reiter 2: Zeit und Horizont ----------------------------------------------------------------------------------------------------------

with tabs[1]:
    st.caption("Wirkt: Geduld δ₁ und δ₂, Horizont T (Marke) und wer zuerst bietet.")
    st.plotly_chart(build_horizon(settings), width="stretch", key="horizon_chart")
    kind, text = ev.verdict_horizon(settings)
    getattr(st, kind)(text)
    st.subheader("🔬 Experiment 1: Die Formel gegen das Rückwärtsrechnen")
    if grid is not None:
        st.caption("Fehler |Anteil bei Horizont T − Formel| für vier Paare (δ₁; δ₂), exakt gerechnet bis T = 400; in der Tabelle die Messreihe bis T = 2 000 (Gleitkomma) und T = 500 in Brüchen.")
        st.plotly_chart(build_convergence(_convergence_curves()), width="stretch", key="e1_chart")
        st.dataframe([{"(δ₁; δ₂)": f"({de(r['d1'], 2)}; {de(r['d2'], 2)})", "Formel": pct(r["limit"], 2), "T = 10": pct(r["x10"], 2), "T = 11": pct(r["x11"], 2), "kleinstes T, Fehler < 10⁻⁶": r["settled"],
                       "vorhergesagt": round(r["predicted"]), "Fehler bei T = 500 (Brüche)": ev.sci(r["err500"]) if r["err500"] else "0"} for r in grid["settle"]], hide_index=True, width="stretch")
        kind, text = ev.verdict_e1(grid["settle"])
        getattr(st, kind)(text)

# --- Reiter 3: Kurze Runden ---------------------------------------------------------------------------------------------------------------

with tabs[2]:
    st.caption("Wirkt: Verlustraten r₁, r₂ und Rundenlänge Δ (Marke in der Tabelle); beide Reihenfolgen werden gezeigt.")
    rows = _short_rows(settings.r1, settings.r2)
    st.plotly_chart(build_short_rounds(rows, settings), width="stretch", key="short_chart")
    kind, text = ev.verdict_short_rounds(settings)
    getattr(st, kind)(text)
    st.dataframe([{"Rundenlänge Δ": C.DT_LABELS[r["dt"]], "Verlader zuerst": pct(r["x_first"], 2), "Spediteur zuerst": pct(r["x_second"], 2), "Vorteil des Zuerstbietens": ev.pp(r["edge"], 2), "Grenzwert": pct(r["limit"], 2)} for r in rows],
                 hide_index=True, width="stretch")
    st.subheader("🔬 Experiment 3: Runden gegen null, der Grenzwert ist die Nash-Lösung")
    if grid is not None:
        st.caption("Fünf Paare von Verlustraten; Abstand zum Grenzwert geteilt durch die Rundenlänge gegen die Handentwicklung r₁r₂/(2(r₁ + r₂)).")
        table = []
        for key, rs in grid["rounds"].items():
            r1, r2 = (float(x) for x in key.split("|"))
            for r in rs:
                if r["dt"] in (0.1, 0.01, 0.001, 0.0001):
                    table.append({"(r₁; r₂)": f"({de(r1, 1)}; {de(r2, 1)})", "Δ": de(r["dt"], 4), "Verlader zuerst": pct(r["x_first"], 3), "Grenzwert": pct(r["limit"], 3), "Abstand/Δ": de(r["rate"], 6), "Handentwicklung": de(r["rate_hand"], 6)})
        st.dataframe(table, hide_index=True, width="stretch")
        kind, text = ev.verdict_e3(grid["rounds"])
        getattr(st, kind)(text)

# --- Reiter 4: Feste Kosten ---------------------------------------------------------------------------------------------------------------

with tabs[3]:
    st.caption("Wirkt: Kosten je Runde c₁, c₂, Horizont T und wer zuerst bietet.")
    st.plotly_chart(build_costs(settings), width="stretch", key="costs_chart")
    kind, text = ev.verdict_costs(settings)
    getattr(st, kind)(text)
    st.subheader("🔬 Experiment 4: Feste Kosten machen das Ergebnis „alles oder fast nichts“")
    if grid is not None:
        st.caption("Neun Fälle (Kosten des ersten, des zweiten Anbieters): Handformel gegen Rückwärtsrechnen bei T = 1 000 und 1 001, daneben der Diskont d = 1 − c (nur bei Kosten unter 1).")
        st.dataframe([{"Kosten (erster; zweiter)": f"({de(r['c_first'], 2)}; {de(r['c_second'], 2)})", "Handformel (erster)": "nicht eindeutig" if r["pred_first"] is None else pct(r["pred_first"]),
                       "T = 1 000": pct(r["t1000_first"]), "T = 1 001": pct(r["t1001_first"]), "Diskont d = 1 − c": "–" if r["discount"] is None else pct(r["discount"])} for r in grid["costs"]], hide_index=True, width="stretch")
        kind, text = ev.verdict_e4(grid["costs"])
        getattr(st, kind)(text)

# --- Reiter 5: Außenoption ------------------------------------------------------------------------------------------------------------------

with tabs[4]:
    st.caption("Wirkt: Außenoption o und Variante, Geduld, Horizont und wer zuerst bietet; die Option hält der Antwortende der ersten Runde.")
    st.plotly_chart(build_outside(settings), width="stretch", key="outside_chart")
    kind, text = ev.verdict_outside(settings)
    getattr(st, kind)(text)
    st.subheader("🔬 Experiment 5: Die Option zählt erst über der Schwelle")
    if grid is not None:
        st.caption("Zwei Lesarten von „Außenoption“ (Halter, Geduld 0,9999 beider Seiten): Nash-Lösung mit Drohpunkt o gegen Ausstiegsprinzip (Variante A).")
        st.dataframe([{"Außenoption o": de(r["o"], 2), "Nash mit Drohpunkt o": pct(r["nash"], 1), "Ausstiegsprinzip": pct(r["exit"], 1)} for r in grid["nash_vs_exit"]], hide_index=True, width="stretch")
        kind, text = ev.verdict_e5(grid["outside_scan"], grid["outside_check"], grid["nash_vs_exit"])
        getattr(st, kind)(text)

# --- Reiter 6: Lösungskonzepte -----------------------------------------------------------------------------------------------------------

with tabs[5]:
    st.caption("Wirkt: Krümmung a, Maßstab k und Drohpunkt; das Rubinstein-Ergebnis daneben folgt der Geduld δ₁, δ₂.")
    st.plotly_chart(build_solutions(settings), width="stretch", key="solutions_chart")
    kind, text = ev.verdict_solutions(settings)
    getattr(st, kind)(text)
    curved = ev.curved_rubinstein_row(settings.d1, settings.d2, settings.curv)
    st.caption(f"Rubinstein bei Krümmung a = {de(settings.curv, 2)} und Geduld ({de(settings.d1, 2)}; {de(settings.d2, 2)}), Verlader bietet zuerst, linearer Nutzen des Verladers: {pct(curved['rubinstein'], 2)} "
               f"(der Diskont des Spediteurs wirkt als δ₂^(1/a) = {de(curved['effective_d2'], 4)}); Nash-Lösung ohne Drohpunkt {pct(curved['nash'], 2)}. Bei Rundenlänge gegen null und gleicher Rate nähert sich das Rubinstein-Ergebnis der Nash-Lösung.")
    st.subheader("🔬 Experiment 6: Maßstabsprobe")
    if grid is not None:
        e6_a = st.selectbox("Krümmung a für die Tabelle", list(C.E6_AS), index=1, key="e6_a", format_func=lambda v: de(v, 2))
        rows6 = sorted([r for r in grid["scale"] if abs(r["a"] - e6_a) < 1e-9], key=lambda r: r["k"])
        st.caption("Der Nutzen des Spediteurs wird mit k multipliziert, der Drohpunkt (Anteil 0,1 von k) wächst mit; die Verhandlungsmenge hat dieselbe Form. Anteile des Verladers. Bei Krümmung 1 und k = 1 ist die Summe der Nutzen konstant: Jeder Anteil ist utilitaristisch optimal, die Demo wählt per Regel den kleinsten (0,0 %).")
        st.plotly_chart(build_scale(rows6), width="stretch", key="e6_chart")
        st.dataframe([{"Maßstab k": de(r["k"], 1), "Nash": pct(r["nash"], 2), "Kalai–Smorodinsky": pct(r["ks"], 2), "egalitär": pct(r["egalitarian"], 2), "utilitaristisch": pct(r["utilitarian"], 2), "Halbe-halbe": "50,00 %"} for r in rows6],
                     hide_index=True, width="stretch")
        kind, text = ev.verdict_e6(grid["scale"])
        getattr(st, kind)(text)

# --- Reiter 7: Raster und Gleichstand -------------------------------------------------------------------------------------------------------

with tabs[6]:
    st.caption("Wirkt: Raster G, Gleichstandsregel, Geduld, Horizont und wer zuerst bietet.")
    st.plotly_chart(build_grid(settings), width="stretch", key="grid_chart")
    kind, text = ev.verdict_grid(settings)
    getattr(st, kind)(text)
    st.subheader("🔬 Experiment 2: Die Rasterfrage")
    if grid is not None:
        st.caption("Stationäre Gleichgewichte des Rasterspiels mit sofortiger Einigung (Anteil des ersten Anbieters), Geduld 0,9 beider Seiten; die Formel sagt 52,6 %.")
        st.dataframe([{"Raster G": r["g"], "Regel": r["rule"], "Gleichgewichte": r["count"], "Anteil von": pct(r["lo"], 1), "bis": pct(r["hi"], 1)} for r in grid["fixed"] if (r["d1"], r["d2"]) == (0.9, 0.9)],
                     hide_index=True, width="stretch")
        kind, text = ev.verdict_e2(grid["bound3"], grid["bound_more_a"], grid["bound_more_b"], grid["rule_gap"])
        getattr(st, kind)(text)

# --- Reiter 8: Lernende Verhandler ---------------------------------------------------------------------------------------------------------

with tabs[7]:
    st.markdown("**Nash heißt nicht teilspielperfekt.** Zwei Lernregeln spielen wiederholt gegeneinander und landen in Nash-Gleichgewichten, in der Messreihe unten aber in keinem Lauf beim teilspielperfekten Ergebnis der Regel A (ein Regret-Matching-Lauf im Ultimatum landet bei 90 %, dem Ergebnis der Regel R).")
    st.caption("Wirkt: Spiel, Lernregel, Lernrate, Lernrunden und Zufalls-Seed. Rechenzeit wächst linear mit den Runden; ein Lauf läuft hinter dem Knopf.")
    learn_key = (settings.game, settings.learn_rule, settings.eta, settings.rounds, settings.seed)
    if st.button("▶️ Lernlauf starten", key="learn_start"):
        with st.spinner(f"Lerne {settings.rounds} Runden …"):
            started = time.perf_counter()
            run = ev.learn_once(settings.game, settings.learn_rule, settings.eta, settings.rounds, settings.seed)
            st.session_state["learn_result"] = (learn_key, run, time.perf_counter() - started)
    stored = st.session_state.get("learn_result")
    if stored and stored[0] == learn_key:
        st.plotly_chart(build_learning(stored[1]), width="stretch", key="learn_chart")
        kind, text = ev.verdict_learning(stored[1])
        getattr(st, kind)(text)
        st.caption(f"Rechenzeit dieses Laufs: {de(stored[2], 2)} s. Reine Nash-Gleichgewichte des Spiels: {stored[1]['pure_nash']}.")
    else:
        st.info("Noch kein Lauf mit diesen Einstellungen: „Lernlauf starten“ rechnet ihn.")
    st.subheader("🔬 Experiment 7: Vier Zufalls-Seeds je Lernregel")
    if grid is not None:
        cells = grid["learn"]
        st.caption(f"{len(cells)} Zellen mal {len(C.GRID_LEARN_SEEDS)} Zufalls-Seeds, je {C.GRID_LEARN_ROUNDS} Runden, letzter Iterierter (kein Zeitmittel). Anteil des Anbieters (Spieler 1); teilspielperfekt wäre bei Regel A 100 % im Ultimatum und 10 % in zwei Runden.")
        st.plotly_chart(build_learning_bands(cells), width="stretch", key="e7_chart")
        st.dataframe([{"Spiel": C.GAMES[c["game"]], "Lernregel": C.LEARN_RULES[c["rule"]], "Lernrate η₀": de(c["eta"], 1) if c["rule"] == "hedge" else "–",
                       "Anteil von Spieler 1 (kleinster bis größter)": f"{pct(min(c['shares']), 1)} bis {pct(max(c['shares']), 1)}", "größte Nash-Lücke": de(max(c["gaps"]), 3), "teilspielperfekt": pct(c["spe"], 0)} for c in cells],
                     hide_index=True, width="stretch")
        kind, text = ev.verdict_e7(cells)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Vollständige Information** | Kennt nur eine Seite den Wert, ändert sich das Ergebnis; die Demo rechnet das nicht | Demo Myerson–Satterthwaite (private Werte, ohne Ablauf in der Zeit) |
| **Zwei Parteien** | Mit mehr Parteien (Koalitionsverhandlung) gelten andere Modelle | – |
| **Linearer Nutzen (quasilinear)** | Mit Krümmung ändert sich das Ergebnis (Reiter Lösungskonzepte); die Wahl der Nutzenfunktion ist nicht gemessen | – |
| **Geduld als Diskontfaktor oder feste Kosten** | Beide Modelle geben verschiedene Ergebnisse (Experiment 4); welches zu einer echten Standzeit passt, sagt die Demo nicht | – |
| **Keine Daten echter Verhandler** | Die Experimente zeigen das Modell, nicht das Verhalten von Betrieben | – |
| **Nash-Verhandlung im Maschinellen Lernen** | Nash-Lösung und Maßstabsinvarianz kommen auch als Gewichtung von Zielen beim Mehraufgaben-Lernen vor (Nash-MTL, Navon u. a., ICML 2022, nicht im Volltext gelesen); dort nur als Einordnung | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(C.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Der Überschuss der Tour ist 1; Spieler 1 (Verlader) und Spieler 2 (Spediteur) teilen ihn. Der erste Anbieter bietet in den ungeraden Runden, der zweite in den geraden. Eine Einigung in Runde $t$ zählt mit
$\delta_i^{\,t-1}$ (Geduld $\delta_i$) bzw. mit $-c_i (t-1)$ (feste Kosten).

| Symbol | Bedeutung |
|---|---|
| $\delta_1, \delta_2$ | Diskontfaktoren (Geduld) des ersten und des zweiten Anbieters |
| $T$, $v_t$ | Horizont; Wert des Anbieters der Runde $t$ im Rückwärtsrechnen |
| $c_i$ | feste Kosten je Runde |
| $o$, $x_2^*$ | Außenoption; Rubinstein-Anteil des Antwortenden ohne Option |
| $G$ | Raster der Angebote (Schritte $1/G$) |
| $a$, $k$, $d_2$ | Krümmung, Maßstab und Drohpunkt des Spediteurs |

**Rubinstein-Gleichgewicht** (unendlicher Horizont, teilspielperfekt):

$$ x_1 = \frac{1-\delta_2}{1-\delta_1\delta_2}, \qquad x_2 = \frac{\delta_2\,(1-\delta_1)}{1-\delta_1\delta_2} $$

**Rückwärtsrechnen** im Spiel mit $T$ Runden ($r(t)$ ist der Antwortende der Runde $t$):

$$ v_T = 1, \qquad v_t = 1 - \delta_{r(t)}\, v_{t+1} $$

**Abstand zur Formel** mit $q = \delta_1\delta_2$ und $T = 2m$ bzw. $T = 2m+1$:

$$ v_1(T) - x_1 = \begin{cases} -\dfrac{1-\delta_2}{1-q}\, q^{m} & T = 2m \\[2mm] +\dfrac{\delta_2\,(1-\delta_1)}{1-q}\, q^{m} & T = 2m+1 \end{cases} $$

**Kurze Runden** ($\delta_i = e^{-r_i \Delta}$) mit Grenzwert und Handentwicklung:

$$ x_1 = \frac{r_2}{r_1 + r_2}\Big(1 + \frac{r_1\,\Delta}{2} + O(\Delta^2)\Big) \;\longrightarrow\; \frac{r_2}{r_1+r_2} \quad (\Delta \to 0) $$

**Feste Kosten** (Auszahlung $x - c_i (t-1)$):

$$ A_T = 1, \qquad A_t = 1 - \max\big(0,\; A_{t+1} - c_{r(t)}\big) $$

**Außenoption** (Variante A: Erlös sofort; Variante C: Erlös eine Runde später):

$$ x_2 = \max\big(o,\; x_2^*\big), \qquad x_2^{C} = \max\big(\delta_2\, o,\; x_2^*\big) $$

**Raster:** der Antwortende mit Fortsetzungswert $w$ nimmt ein Angebot $k/G$ an, wenn $k/G \ge w$ (Regel A) bzw. $k/G > w$ (Regel R); das kleinste annehmbare Angebot ist

$$ k = \lceil G\,w \rceil \;\;(\text{A}), \qquad k = \lfloor G\,w \rfloor + 1 \;\;(\text{R}), \qquad \text{Schranke aus der Messung: } \big|x_G - x_1\big| \le \frac{1}{(1-\delta_1\delta_2)\,G} $$

**Lösungskonzepte** auf der Menge $u_1 = x$, $u_2 = k\,(1-x)^a$ mit Drohpunkt $(0, d_2)$:

$$ \text{Nash: } \max_x\; x\,\big(u_2(x) - d_2\big), \qquad \text{egalitär: } u_1 = u_2 - d_2, \qquad \text{utilitaristisch: } \max_x\; u_1 + u_2 $$

$$ \text{Kalai–Smorodinsky: } \frac{u_1}{\bar u_1} = \frac{u_2 - d_2}{\bar u_2 - d_2}, \qquad \bar u_1 = 1 - \big(d_2/k\big)^{1/a},\; \bar u_2 = k $$

**Nash ohne Drohpunkt** und **Rubinstein bei gekrümmtem Nutzen** (der Diskont des Spediteurs wirkt als $\delta_2^{1/a}$):

$$ x_{\text{Nash}} = \frac{1}{1+a}, \qquad x_1 = \frac{1-\delta_2^{1/a}}{1-\delta_1\,\delta_2^{1/a}} $$

**Lernregeln:** Hedge mit Lernrate $\eta_t = \eta_0/\sqrt{t}$ und Regret Matching mit den kumulierten Bedauern $R_t(i)$:

$$ p_t(i) \;\propto\; \exp\Big(\eta_t \sum_{s<t} u_s(i)\Big), \qquad p_t(i) \;\propto\; \max\big(R_t(i),\, 0\big) $$

**Quellen.** Die Formeln sind durch die unabhängigen Rechenwege der Tests (Brüche, lineares Gleichungssystem, geschlossene Summe, Aufzählung aller Rasterangebote, scipy) und die Messungen der Experimente gestützt. Literaturangaben
(Nash 1950; Rubinstein 1982; Binmore, Rubinstein, Wolinsky 1986; Kalai und Smorodinsky 1975; Kalai 1977) wurden für diese Demo nicht im Volltext gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
