# Verhandlung: wer Zeit hat, bekommt mehr – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-verhandlung-demo.streamlit.app/)**

Konzepte-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Teil der Linie zu Verhandlung, Auktion und Vertrag. Ein **Verlader** und ein **Spediteur** teilen den Überschuss einer Tour
(Wert für den Verlader minus Kosten des Spediteurs, hier auf 1 normiert; die Frachtrate ist Kosten plus Anteil des Spediteurs mal Überschuss) und machen sich **abwechselnd Angebote**; jede Runde Warten kostet beide etwas. Die Demo rechnet das
Rubinstein-Gleichgewicht (Formel gegen unabhängiges Rückwärtsrechnen), zeigt, was Geduld, Horizont, Rundenlänge, feste Kosten, eine Außenoption und ein grobes Angebotsraster ändern, vergleicht Lösungskonzepte mit einer
Maßstabsprobe und lässt zwei Lernregeln gegeneinander spielen. Nachbarn im Portfolio: `shapley-demo` und `kern-demo` teilen einen Gewinn **ohne Ablauf** (bei zwei Spielern ist der Shapley-Wert Halbe-halbe und kennt weder Geduld
noch Nutzenform); `myerson-satterthwaite-demo` hat **private** Werte; hier kennen beide alles. **Eigennutz, Geduld und Nutzenfunktionen sind Modellannahmen, keine Aussagen über echte Verhandler.**

## Warum dieses Problem

Wer eine Frachtrate aushandelt, teilt einen Überschuss, und wer geduldiger wird, bekommt mehr; ein kleiner Vorsprung an Geduld gleicht den Vorteil des Zuerstbietens nicht aus (Geduld 0,90 und 0,905: der Verlader bekommt 51,2 %). Befund (Geduld beider Seiten 0,9, 200 Runden, stetige Angebote, Verlader bietet zuerst, linearer Nutzen): der Verlader bekommt
**52,6 %** des Überschusses, der Spediteur 47,4 %; die Formel (1 − δ₂)/(1 − δ₁δ₂) gibt 10/19. Der Wert der Geduld wächst: bei Geduld des Spediteurs 0,9 steigt der Anteil des Verladers, wenn seine Geduld von 0,90 auf 0,91 wächst,
von 52,6 % auf **55,2 %** (+2,62 Prozentpunkte), und die Steigung des Anteils nach der Geduld liegt bei Geduld 0,5 bei 0,30, bei 0,9 bei 2,49 und bei 0,99 bei 7,58.
Der Shapley-Wert (Halbe-halbe) bleibt dabei bei 50,0 %.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`verhandeln-planung/STUECK-B-verhandlung.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte; alle Zahlen dieser README sind **neu gemessen**. Wo die Vormessung dasselbe maß, stimmen die Zahlen überein, mit diesen Ausnahmen und Ergänzungen:

- **Kalai–Smorodinsky und utilitaristisch** rechnen hier auf der Menge der Einigungen, in denen der Spediteur seinen Drohpunkt erreicht (Idealpunkt: der größte Nutzen auf dieser Menge); die Vormessung nahm den Idealpunkt der ganzen Menge. Ohne Drohpunkt ist das dasselbe, mit Drohpunkt über null unterscheiden sich die Zahlen. Bei gleicher Summe (Krümmung 1, Maßstab 1) gilt der kleinste Anteil des Verladers.
- **Geduld höchstens 0,99** (Plan: 0,999): bei 0,999 reichen 2 000 Runden nicht zur Konvergenz; bei 0,99 brauchen die Runden bis zum Fehler unter 10⁻⁶ schon 1307 (vorhergesagt 1375).
- **Lernläufe** nutzen den Zufallsgenerator SplitMix64 (wörtliche Kopie aus `pearson-demo`) statt NumPy; die Zahlen der Vormessung gelten deshalb nicht, die Bänder unten sind neu gemessen.
- **Variante B der Außenoption** wird nicht als Regler angeboten: sie ist in 0 von 2400 Zellen anders als Variante A (siehe Befunde). Der Test hält das fest.
- **Nicht gebaut:** der Reiter zu Hin- und Rückfahrt mit Flussungleichgewicht (siehe „Bewusst nicht umgesetzt“). Die Hypothese H10 des Plans ist deshalb hier **nicht geprüft**.
- **Gleichwertige Mutanten:** siehe Verifikation.

## Modell

Der Überschuss der Tour ist 1; Spieler 1 (Verlader) und Spieler 2 (Spediteur) teilen ihn. In Runde 1 bietet der **erste Anbieter**, in Runde 2 der zweite usw.; wer antwortet, nimmt an oder lehnt ab. Nach **T Runden** ist Schluss
(T = 1: Ultimatum). Wartekosten: ein **Diskontfaktor δ** je Runde (eine Einigung in Runde t zählt mit δ^(t−1)) oder **feste Kosten** c je Runde (Auszahlung x − c·(t − 1)). Das Gleichgewicht ist das teilspielperfekte, gerechnet durch
**Rückwärtsrechnen** (`vhd_rubinstein.py`): in der letzten Runde bekommt der Anbieter alles, davor muss er dem Antwortenden so viel bieten, wie dieser als nächster Anbieter nach Abzug der Wartekosten bekäme. Im Gleichgewicht
einigen sich beide sofort in Runde 1; die späteren Runden bestimmen den Preis dafür. Bei unendlichem Horizont gilt die Formel (1 − δ₂)/(1 − δ₁δ₂) für den ersten Anbieter; der Abstand bei Horizont T ist in geschlossener Form
durch die geometrische Summe gegeben (📐-Expander der App, getestet gegen exaktes Rückwärtsrechnen in Brüchen).

- **Raster** (`vhd_raster.py`): Angebote nur als Vielfache von 1/G, gerechnet **ganzzahlig in Brüchen**, nie in Gleitkomma. Gleichstand in drei Ebenen: der Antwortende nimmt bei gleichem Wert an (Regel A) oder lehnt ab (Regel R); der Anbieter zieht bei
  gleicher Auszahlung das angenommene Angebot vor; unter gleich guten angenommenen Angeboten gilt das kleinste. Dazu die stationären Gleichgewichte mit sofortiger Einigung.
- **Außenoption** (`vhd_outside.py`): der Antwortende der ersten Runde kann nach einer Ablehnung aussteigen. Variante A: Erlös o sofort; Variante C: eine Runde später (δ·o). Sie zählt erst über der Schwelle x₂* = δ₂(1 − δ₁)/(1 − δ₁δ₂) (Variante A) bzw. x₂*/δ₂ (Variante C).
- **Kurze Runden:** δᵢ = exp(−rᵢ·Δ) bei Rundenlänge Δ; Grenzwert für Δ → 0 ist r₂/(r₁ + r₂) für den ersten Anbieter Verlader (Verlustraten r₁, r₂).
- **Lösungskonzepte** (`vhd_solutions.py`): Verhandlungsmenge u₁ = x, u₂ = k·(1 − x)^a, Drohpunkt (0, d₂). Nash, Kalai–Smorodinsky, egalitär, utilitaristisch und Halbe-halbe, ohne scipy zur Laufzeit (Bisektion auf Funktionen, die auf dem Suchintervall monoton sind); Rubinstein bei gekrümmtem Nutzen über den wirksamen Diskont δ₂^(1/a).
- **Lernende Verhandler** (`vhd_learning.py`): Hedge (Lernrate η₀/√t) und Regret Matching im Ultimatumspiel (Zehntelraster) und im Spiel mit zwei Runden (Diskont 0,9). Pro Runde ziehen beide Spieler eine Aktion (SplitMix64, fester Zufalls-Seed); ausgewiesen wird der letzte Iterierte.

## Methodik

- **Kleine Einheiten mit Mini-Instanzen von Hand** (DEMO-PLAYBOOK §1b, `tests/conftest.py`): Formel, Rückwärtsrechnen, Kosten, kleinstes annehmbares Rasterangebot, Raster-Rückwärtsrechnen, Fixpunkte, Außenoption, jedes Lösungskonzept, Lernregeln und Nash-Lücke.
- **Orakel** (`tests/test_oracle_*.py`, anderer Rechenweg): Formel gegen das lineare Gleichungssystem A₁ = 1 − δ₂A₂, A₂ = 1 − δ₁A₁ (`numpy.linalg.solve`), gegen die geschlossene Summe und gegen exaktes Rückwärtsrechnen in Brüchen (T = 500, 501); Raster gegen die Aufzählung **aller** Rasterangebote; Fixpunkte gegen
  die Prüfung aller (A₁, A₂)-Paare; feste Kosten gegen die Aufzählung aller Rasterangebote und die Fallunterscheidung von Hand; Außenoption gegen einen Spielbaum mit allen Entscheidungen und die Handformeln; Nash und Kalai–Smorodinsky gegen `scipy.optimize` und ein Gitter mit 400 001 Punkten;
  Rubinstein bei gekrümmtem Nutzen gegen das Rückwärtsrechnen direkt in Nutzen; Lernregeln gegen eine Schleifenfassung ohne NumPy mit demselben Zufallsstrom; reine Nash-Gleichgewichte gegen die Prüfung jedes Aktionspaars.
- **Eigenschaftstests:** Spieler umbenennen, positiv-affine Umrechnung der Nutzen (Nash und Kalai–Smorodinsky unverändert, egalitär nicht), Monotonie in der Geduld bei jedem Horizont, Anteile summieren zu 1, gleicher Seed gleiche Ausgabe.
- **Experimente** (Reiter der App, jeweils live und als vorgerechnete Messreihe): (0) Wert der Geduld, (1) Formel gegen Rückwärtsrechnen, (2) die Rasterfrage, (3) kurze Runden, (4) feste Kosten gegen Diskont, (5) Außenoption, (6) Maßstabsprobe, (7) Lernende.
- **Vorgerechnete Messreihen** (`tools/make_grid.py`, `data/negotiation_grid.json`): alles bis auf die Lernläufe ist deterministisch (Brüche oder Formeln); die Lernläufe benutzen 4 feste Zufalls-Seeds und 20 000 Runden je Lauf. `tests/test_grid.py` rechnet eine Probe nach.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Tabellen Zeile für Zeile in der Rundung der App). Jede Zahl nennt ihre Bedingung; Anteile sind Anteile am Überschuss der Tour.

| Hypothese | Ergebnis | Befund (Bedingung) |
|---|---|---|
| **H1** Die Formel (1 − δ₂)/(1 − δ₁δ₂) ist der Grenzwert des Rückwärtsrechnens; der Fehler fällt wie (δ₁δ₂)^(T/2) | Ja | exakt in Brüchen für T = 1 bis 40 gleich der geschlossenen Summe; Fehler bei T = 500 und 501 unter 10⁻²⁰ (Paare (0,9; 0,9), (0,95; 0,8), (0,5; 0,9), (0,8; 0,95)); kleinstes T für Fehler < 10⁻⁶: 126 gemessen, 131 vorhergesagt bei (0,9; 0,9) |
| **H2** Ungerade T liegen nicht unter, gerade nicht über dem Grenzwert; T = 1 ist das Ultimatum | Ja | Geduld (0,9; 0,9), stetig: T = 10 gibt 34,28 %, T = 11 69,15 %, Grenzwert 52,63 %; die geschlossene Form des Abstands (📐-Expander) gilt exakt in Brüchen für 60 Zufallspaare mit Geduld 0 bis 0,98 und T = 1 bis 14 |
| **H3** Raster: Fehler ≤ 1/((1 − δ₁δ₂)·G); mehrere stationäre Gleichgewichte; hängt an der Gleichstandsregel | Ja, mit Ausnahmen | Schranke in 196 von 200 Zellen; die 4 Ausnahmen sind δ₁ = δ₂ = 0,99, G = 10 000 bei T = 400 und 401 (T noch nicht konvergiert); bis zu 9 stationäre Gleichgewichte; Regel A gegen R: größter Unterschied über T = 1 bis 60 in allen 12 Zellen (4 Paare, G = 10, 100, 1 000) genau 1/G |
| **H4** Δ → 0 mit gleicher Rate: Anteil → r₂/(r₁ + r₂) (Nash mit Gewichten 1/r); der Vorteil des Zuerstbietens sinkt linear in Δ | Ja | (1; 2) je Tag, Verlader bietet zuerst: Runden von 1 Tag 91,00 %, 1 Stunde 68,05 %, 1 Minute 66,69 %, Grenzwert 66,67 %; Vorteil bei Raten (1; 1) von 0,0500 (Δ = 0,1) auf 0,000500 (Δ = 0,001) |
| **H5** Feste Kosten: wer weniger Kosten hat und zuerst bietet, bekommt bei langem Horizont genau 1; hat der erste die größeren, bekommt er die Kosten des anderen | Ja, enger gefasst als im Plan („fast alles“ wird „genau 1“) | 8 von 8 Fällen mit ungleichen Kosten (T = 1 000 und 1 001); bei gleichen Kosten kein eindeutiges Ergebnis: ungerades T gibt dem ersten Anbieter alles, gerades T nur c |
| **H6** Die Außenoption zählt erst über dem Rubinstein-Anteil des Antwortenden, dann gilt o; Variante C verschiebt die Schwelle | Ja | Geduld (0,9; 0,9): Schwelle 0,4737 (A), 0,5263 (C); Variante B in 0 von 2400 Zellen anders als A |
| **H7** Nash und Kalai–Smorodinsky sind maßstabsinvariant, egalitär und utilitaristisch nicht | Ja | Maßstab k = 0,1; 1; 10, Drohpunkt 0,1·k, vier Krümmungen: Nash und Kalai–Smorodinsky gleich auf 9 Stellen; egalitär bei Krümmung 0,5: 8,6 % (k = 0,1), 56,2 % (k = 1), 96,2 % (k = 10) |
| **H8** Rubinstein bei Δ → 0 und Krümmung a nähert sich der Nash-Lösung; ein stärker gekrümmter Nutzen des Spediteurs verschiebt zugunsten des Verladers | Ja | Abstand bei Δ = 0,001 unter 5·10⁻⁴ für a = 1; 0,5; 0,25; Krümmung 0,5 und Drohpunkt 0,1: Nash 62,6 % statt 45,0 % bei linearem Nutzen. „Risikoscheu“ passt nur für die Nash-Lösung mit Lotterien; im Rubinstein-Spiel gibt es kein Risiko, dort wirkt die Krümmung wie größere Ungeduld (wirksamer Diskont δ₂^(1/a)) |
| **H9** Lernende landen im Nash-, nicht im teilspielperfekten Gleichgewicht | Ja, als Bandbreite | 32 Läufe, kein Lauf beim teilspielperfekten Anteil der Regel A (Ultimatum 100 %, zwei Runden 10 %), alle mit Nash-Lücke höchstens 0,024; ein Regret-Matching-Lauf im Ultimatum (Zufalls-Seed 3) landet bei 90 %, dem teilspielperfekten Ergebnis der Regel R (Ablehnung bei Gleichstand), die übrigen Läufe bei 70 % bis 80 % beziehungsweise 18 % bis 60 % und sind nicht teilspielperfekt; die Anteile hängen an Lernrate und Zufalls-Seed (Tabelle) |
| **H10** Das Verhältnis der Raten Hin/Rück hängt linear am inversen Flussverhältnis | nicht geprüft | der Reiter ist nicht gebaut |
| **H11** Die Nash-Lösung mit Drohpunkt und das Ausstiegsprinzip teilen gleich auf | Nein | bei o = 0,2 und Geduld 0,9999 beider Seiten gibt Nash dem Halter 60,0 %, das Ausstiegsprinzip 50,0 % (Tabelle) |

**Wert der Geduld** (Experiment 0; Geduld des Spediteurs 0,9, Verlader bietet zuerst bzw. Spediteur bietet zuerst, unendlicher Horizont; Steigung des Anteils nach δ₁):

| Geduld des Verladers δ₁ | Verlader zuerst | Spediteur zuerst | Halbe-halbe | Steigung |
|---|---|---|---|---|
| 0,00 | 10,0 % | 0,0 % | 50,0 % | 0,09 |
| 0,50 | 18,2 % | 9,1 % | 50,0 % | 0,30 |
| 0,80 | 35,7 % | 28,6 % | 50,0 % | 1,15 |
| 0,90 | 52,6 % | 47,4 % | 50,0 % | 2,49 |
| 0,99 | 91,7 % | 90,8 % | 50,0 % | 7,58 |

**Formel gegen Rückwärtsrechnen** (Experiment 1; Gleitkomma bis T = 2 000, Fehler gegen die Formel; „kleinstes T“ heißt: der Fehler bleibt für alle größeren T bis 2 000 unter 10⁻⁶; Vorhersage aus (δ₁δ₂)^(T/2) < 10⁻⁶):

| (δ₁; δ₂) | Formel | T = 10 | T = 11 | kleinstes T | vorhergesagt |
|---|---|---|---|---|---|
| 0,90; 0,90 | 52,63 % | 34,28 % | 69,15 % | 126 | 131 |
| 0,99; 0,99 | 50,25 % | 4,80 % | 95,24 % | 1307 | 1375 |
| 0,95; 0,80 | 83,33 % | 62,20 % | 87,56 % | 99 | 101 |
| 0,50; 0,90 | 18,18 % | 17,85 % | 19,69 % | 36 | 35 |

**Die Rasterfrage** (Experiment 2). Zählung der Schranke |Anteil im Raster − Formel| ≤ 1/((1 − δ₁δ₂)·G) über beide Gleichstandsregeln (Anteil des ersten Anbieters; die Schranke ist eine Hypothese aus dieser Messung, kein Satz):

| Zellen | Anzahl | Schranke eingehalten | größtes Verhältnis |
|---|---|---|---|
| 3 Paare, G = 10 bis 10 000, beide Regeln, T = 600 und 601 | 60 | 60 | 0,82 |
| 5 weitere Paare, G = 1 000 und 10 000, T = 400, 401, 2 000, 2 001 | 80 | 76 | 1,81 |
| 5 weitere Paare, G = 10, 50, 100, T = 400, 401 | 60 | 60 | 0,98 |

Stationäre Gleichgewichte des Rasterspiels mit sofortiger Einigung, Geduld (0,9; 0,9), Anzahl und Anteil des ersten Anbieters von … bis … (die Formel sagt 52,6 %); stetig ist das Gleichgewicht eindeutig:

| Raster G | Regel A | Regel R |
|---|---|---|
| 10 | 9 (10 % bis 90 %) | 9 (10 % bis 90 %) |
| 20 | 2 (50 % bis 55 %) | 1 (50 % bis 50 %) |
| 50 | 7 (46 % bis 58 %) | 9 (44 % bis 60 %) |
| 100 | 6 (50 % bis 55 %) | 4 (51 % bis 54 %) |

Das Ultimatum auf dem Raster mit Zehnteln und Geduld 0,9: Regel A gibt dem Anbieter 100,0 %, Regel R 90,0 % (G = 100: 99,0 %). **Rasterfalle** (G = 10, Geduld 0,9, Regel A): der Verlader bekommt 10,0 % bei T = 200 (gerade) und 90,0 % bei
T = 41 (ungerade), statt 52,6 % im stetigen Spiel; mit G = 100 und Regel A bei T = 200 sind es 50,0 %.

**Kurze Runden** (Experiment 3; Verlader bietet zuerst; Abstand zum Grenzwert geteilt durch Δ gegen die Handentwicklung r₁r₂/(2(r₁ + r₂)); Raten (r₁; r₂) je Tag):

| (r₁; r₂) | Δ | Verlader zuerst | Grenzwert | Abstand/Δ | Handentwicklung |
|---|---|---|---|---|---|
| 1,0; 1,0 | 0,1000 | 52,498 % | 50,000 % | 0,249792 | 0,250000 |
| 1,0; 1,0 | 0,0100 | 50,250 % | 50,000 % | 0,249998 | 0,250000 |
| 1,0; 1,0 | 0,0010 | 50,025 % | 50,000 % | 0,250000 | 0,250000 |
| 1,0; 1,0 | 0,0001 | 50,002 % | 50,000 % | 0,250000 | 0,250000 |
| 1,0; 2,0 | 0,1000 | 69,939 % | 66,667 % | 0,327237 | 0,333333 |
| 1,0; 2,0 | 0,0100 | 66,999 % | 66,667 % | 0,332772 | 0,333333 |
| 1,0; 2,0 | 0,0010 | 66,700 % | 66,667 % | 0,333278 | 0,333333 |
| 1,0; 2,0 | 0,0001 | 66,670 % | 66,667 % | 0,333328 | 0,333333 |
| 2,0; 1,0 | 0,1000 | 36,717 % | 33,333 % | 0,338321 | 0,333333 |
| 2,0; 1,0 | 0,0100 | 33,667 % | 33,333 % | 0,333883 | 0,333333 |
| 2,0; 1,0 | 0,0010 | 33,367 % | 33,333 % | 0,333389 | 0,333333 |
| 2,0; 1,0 | 0,0001 | 33,337 % | 33,333 % | 0,333339 | 0,333333 |
| 1,0; 4,0 | 0,1000 | 83,788 % | 80,000 % | 0,378797 | 0,400000 |
| 1,0; 4,0 | 0,0100 | 80,398 % | 80,000 % | 0,397987 | 0,400000 |
| 1,0; 4,0 | 0,0010 | 80,040 % | 80,000 % | 0,399800 | 0,400000 |
| 1,0; 4,0 | 0,0001 | 80,004 % | 80,000 % | 0,399980 | 0,400000 |
| 0,5; 3,0 | 0,1000 | 87,765 % | 85,714 % | 0,205115 | 0,214286 |
| 0,5; 3,0 | 0,0100 | 85,928 % | 85,714 % | 0,213390 | 0,214286 |
| 0,5; 3,0 | 0,0010 | 85,736 % | 85,714 % | 0,214196 | 0,214286 |
| 0,5; 3,0 | 0,0001 | 85,716 % | 85,714 % | 0,214277 | 0,214286 |

Beispiel Raten (1; 2) je Tag: Runden von einem Tag geben dem Verlader 91,00 %, bietet der Spediteur zuerst nur 33,48 %; bei einer Stunde 68,05 %, bei einer Minute 66,69 %; Grenzwert 66,67 %, nicht Halbe-halbe.

**Feste Kosten gegen Diskont** (Experiment 4; Kosten (erster Anbieter; zweiter Anbieter); Anteil des ersten Anbieters; Diskont d = 1 − c in derselben Zeile; „Handformel“ nur bei ungleichen Kosten):

| Kosten | Handformel | T = 1 000 | T = 1 001 | Diskont d = 1 − c |
|---|---|---|---|---|
| 0,05; 0,10 | 100,0 % | 100,0 % | 100,0 % | 69,0 % |
| 0,10; 0,05 | 5,0 % | 5,0 % | 5,0 % | 34,5 % |
| 0,02; 0,30 | 100,0 % | 100,0 % | 100,0 % | 95,5 % |
| 0,30; 0,02 | 2,0 % | 2,0 % | 2,0 % | 6,4 % |
| 0,20; 0,20 | nicht eindeutig | 20,0 % | 100,0 % | 55,6 % |
| 0,00; 0,05 | 100,0 % | 100,0 % | 100,0 % | 100,0 % |
| 0,05; 0,00 | 0,0 % | 0,0 % | 0,0 % | 0,0 % |
| 0,60; 0,70 | 100,0 % | 100,0 % | 100,0 % | 79,5 % |
| 1,20; 0,30 | 30,0 % | 30,0 % | 30,0 % | – |

Feste Kosten machen das Ergebnis „alles oder fast nichts“, der Diskont macht es stetig: bei Kosten (0,05; 0,10) gibt der Diskont 69,0 % statt 100 %, bei (0,10; 0,05) 34,5 % statt 5,0 %, bei (0,02; 0,30) 95,5 % statt 100 %.

**Außenoption** (Experiment 5; Geduld (0,9; 0,9), Anteil des Halters bei T = 1 001): die Schwelle der Variante A ist 0,4737, die der Variante C 0,5263:

| Außenoption o | Variante A | Variante C |
|---|---|---|
| 0,40 | 47,4 % | 47,4 % |
| 0,50 | 50,0 % | 47,4 % |
| 0,60 | 60,0 % | 54,0 % |
| 0,80 | 80,0 % | 72,0 % |
| 0,95 | 95,0 % | 85,5 % |

Zwei Lesarten von „Außenoption“ (Anteil des Halters bei Geduld 0,9999 beider Seiten): Nash-Lösung mit Drohpunkt o gegen das Ausstiegsprinzip (Variante A). Rückwärtsrechnen gegen Handformel bei T = 1 000 und 1 001: größte Abweichung 2,2·10⁻⁵
(Geduld 0,99 ist dabei noch nicht ganz konvergiert).

| Außenoption o | Nash mit Drohpunkt o | Ausstiegsprinzip |
|---|---|---|
| 0,00 | 50,0 % | 50,0 % |
| 0,20 | 60,0 % | 50,0 % |
| 0,40 | 70,0 % | 50,0 % |
| 0,50 | 75,0 % | 50,0 % |
| 0,60 | 80,0 % | 60,0 % |
| 0,80 | 90,0 % | 80,0 % |
| 0,95 | 97,5 % | 95,0 % |

**Maßstabsprobe** (Experiment 6; Anteil des Verladers; Nutzen des Spediteurs mal k, Drohpunkt 0,1·k; Nutzen u₂ = k·(1 − x)^a; bei Krümmung 1 und k = 1 ist die Summe der Nutzen überall gleich, der kleinste Anteil gilt):

| Krümmung a | Maßstab k | Nash | Kalai–Smorodinsky | egalitär | utilitaristisch |
|---|---|---|---|---|---|
| 1,00 | 0,1 | 45,0 % | 45,0 % | 8,2 % | 90,0 % |
| 1,00 | 1,0 | 45,0 % | 45,0 % | 45,0 % | 0,0 % (beliebig, Summe konstant) |
| 1,00 | 10,0 | 45,0 % | 45,0 % | 81,8 % | 0,0 % |
| 0,50 | 0,1 | 62,6 % | 59,2 % | 8,6 % | 99,0 % |
| 0,50 | 1,0 | 62,6 % | 59,2 % | 56,2 % | 75,0 % |
| 0,50 | 10,0 | 62,6 % | 59,2 % | 96,2 % | 0,0 % |
| 0,25 | 0,1 | 77,4 % | 70,7 % | 8,8 % | 99,3 % |
| 0,25 | 1,0 | 77,4 % | 70,7 % | 66,2 % | 84,3 % |
| 0,25 | 10,0 | 77,4 % | 70,7 % | 99,8 % | 0,0 % |
| 2,00 | 0,1 | 28,7 % | 29,8 % | 7,5 % | 68,4 % |
| 2,00 | 1,0 | 28,7 % | 29,8 % | 33,8 % | 0,0 % |
| 2,00 | 10,0 | 28,7 % | 29,8 % | 60,0 % | 0,0 % |

**Rubinstein bei gekrümmtem Nutzen** (Anteil des Verladers, Verlader bietet zuerst, unendlicher Horizont; der Diskont des Spediteurs wirkt als δ₂^(1/a); Nash ohne Drohpunkt 1/(1 + a)); bei Krümmung 0,5 und Geduld (0,9; 0,9) 70,11 %, die Nash-Lösung ohne Drohpunkt 66,7 %:

| Krümmung a | Geduld (beide) | wirksamer Diskont | Rubinstein | Nash |
|---|---|---|---|---|
| 1,00 | 0,90 | 0,9000 | 52,63 % | 50,00 % |
| 1,00 | 0,99 | 0,9900 | 50,25 % | 50,00 % |
| 0,50 | 0,90 | 0,8100 | 70,11 % | 66,67 % |
| 0,50 | 0,99 | 0,9801 | 67,00 % | 66,67 % |
| 0,25 | 0,90 | 0,6561 | 83,98 % | 80,00 % |
| 0,25 | 0,99 | 0,9606 | 80,40 % | 80,00 % |

Abstand Rubinstein minus Nash bei gleicher Rate und Rundenlänge Δ (Tage):

| Krümmung a | Δ | Abstand |
|---|---|---|
| 1,00 | 0,100 | 0,02498 |
| 1,00 | 0,010 | 0,00250 |
| 1,00 | 0,001 | 0,00025 |
| 0,50 | 0,100 | 0,03272 |
| 0,50 | 0,010 | 0,00333 |
| 0,50 | 0,001 | 0,00033 |
| 0,25 | 0,100 | 0,03788 |
| 0,25 | 0,010 | 0,00398 |
| 0,25 | 0,001 | 0,00040 |

**Lernende Verhandler** (Experiment 7; Anteil des Anbieters (Spieler 1) nach 20 000 Runden, letzter Iterierter, 4 Zufalls-Seeds je Zeile, Bereich kleinster bis größter; größte Nash-Lücke; teilspielperfekt nach Regel A: Ultimatum 100 %, zwei Runden 10 %). **Nash heißt nicht teilspielperfekt:**
32 Läufe, kein Lauf beim teilspielperfekten Anteil der Regel A; ein Regret-Matching-Lauf im Ultimatum liegt bei 90 %, dem Ergebnis der Regel R (Ablehnung bei Gleichstand); im Ultimatumspiel gibt es 13 reine Nash-Gleichgewichte (Regret Matching: 70 % bis 90 %), im Spiel mit zwei Runden hängt der Anteil an Lernrate und Zufalls-Seed (Hedge mit Lernrate 2: 18,2 % bis 18,3 %; Hedge mit Lernrate 8: 20 % bis 60 %; Regret Matching: 20 % bis 60 %). Das gilt für diese zwei Lernregeln in diesen zwei
Spielen und ist kein Befund darüber, wie echte Verhandler handeln.

| Spiel | Lernregel | Lernrate η₀ | Anteil von Spieler 1 | größte Nash-Lücke | teilspielperfekt |
|---|---|---|---|---|---|
| Ultimatumspiel (eine Runde) | Hedge (Entropie-Gewichte) | 0,5 | 80,0 % bis 80,0 % | 0,000 | 100 % |
| Ultimatumspiel (eine Runde) | Hedge (Entropie-Gewichte) | 2,0 | 80,0 % bis 80,0 % | 0,000 | 100 % |
| Ultimatumspiel (eine Runde) | Hedge (Entropie-Gewichte) | 8,0 | 70,0 % bis 80,0 % | 0,000 | 100 % |
| Ultimatumspiel (eine Runde) | Regret Matching | – | 70,0 % bis 90,0 % | 0,000 | 100 % |
| Zwei Runden (Diskont 0,9) | Hedge (Entropie-Gewichte) | 0,5 | 19,4 % bis 20,2 % | 0,024 | 10 % |
| Zwei Runden (Diskont 0,9) | Hedge (Entropie-Gewichte) | 2,0 | 18,2 % bis 18,3 % | 0,007 | 10 % |
| Zwei Runden (Diskont 0,9) | Hedge (Entropie-Gewichte) | 8,0 | 20,0 % bis 60,0 % | 0,006 | 10 % |
| Zwei Runden (Diskont 0,9) | Regret Matching | – | 20,0 % bis 60,0 % | 0,000 | 10 % |

## Ehrliche Grenzen

**Annahmen**

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

Negative und gemischte Befunde stehen oben in der Tabelle: die Schranke für das Raster gilt in 196 von 200 Zellen und nicht in allen; die Hypothese H11 ist widerlegt; die Lernergebnisse sind nur als Bänder belastbar; H10 ist nicht geprüft.
Nash-Lösung und Maßstabsinvarianz kommen auch als Gewichtung von Zielen beim Mehraufgaben-Lernen vor (Nash-MTL, Navon u. a., ICML 2022); das ist hier nur eine Einordnung, nicht im Volltext gelesen und nicht nachgebaut.

## Verifikation

- **Kern:** jede Einheit gegen Handrechnung auf den Mini-Instanzen von `tests/conftest.py`; Randfälle (T = 1 und 2, Geduld 0, Gleichstand, Rasterfeinheit G = 1, Optionswerte an den Grenzen, Reglermaxima T = 2 000 und G = 10 000).
- **Orakel und Eigenschaften:** siehe Methodik (`tests/test_oracle_rubinstein.py`, `test_oracle_raster.py`, `test_oracle_outside.py`, `test_oracle_solutions.py`, `test_oracle_learning.py`, `test_eigenschaften_verhandlung.py`); die Kopie `vhd_rng.py` hat einen Kopie-Test (`tests/test_copies.py`).
- **Aussagen:** `tests/test_claims.py` hält die Zahlen dieser README fest (Tabellen Zeile für Zeile, Zahlen im Fließtext als Zeichenketten in der Rundung der App, die Preset-Hilfen als Teilsätze); `tests/test_verdicts.py` prüft die Zahlen im Text der Verdikte an von Hand gesetzten Werten; `tests/test_grid.py` prüft Aufbau, Zählungen und eine Probe der vorgerechneten Messreihen.
- **Oberfläche:** AppTest für jedes Preset, den Permalink, jeden Knopf (auch den ▶️-Lernlauf je Spiel und Lernregel), jeden Regler am Minimum und Maximum einzeln und alle zusammen, die Reglermaxima T = 2 000, G = 10 000 und 50 000 Lernrunden; Layout (Schnellstart in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben, jede Änderung eines Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
- **Tests:** 349 Tests, die Suite läuft lokal im Test-venv in rund 2,5 Minuten.
- **Mutationsprüfung** des Kerns (`vhd_rubinstein.py`, `vhd_raster.py`, `vhd_outside.py`, `vhd_solutions.py`, `vhd_learning.py`; ohne die wörtliche Kopie `vhd_rng.py`): 352 Stellen, 347 gefangen, 5 gleichwertig, 0 offen (Zustand in `tests/mutation_state.json`). Gleichwertig sind: eine zusätzliche Schleifenrunde in `grid_fixed_points` (ein Anteil über 1 kann kein Gleichgewicht sein), der Zweig „aussteigen statt bieten“ der Variante B in `outside_backward` (er greift nie, weil der Halter dort immer mindestens o bekommt; getestet) und drei Stellen in der Bisektion (Gleichheit bei exakter Nullstelle, Gleichheit der Randwerte, 199 statt 200 Schritte). Auswertung, Verdikte und Oberfläche sind nicht mutiert; für sie stehen `test_verdicts.py`, `test_claims.py` und die AppTests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht, acht Reiter mit Experimenten, Annahmen, Mathematik |
| `data/negotiation_grid.json` | vorgerechnete Messreihen (E0 bis E7) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, Plotly, fpdf2 |
| `tools/make_grid.py` | rechnet die Messreihen vor |
| `vhd_constants.py` | Regler-Grenzen, Auswahlmöglichkeiten, Raster der Messreihen, Farben, Presets, Annahmentext |
| `vhd_evaluation.py` | Kennzahlen je Reiter, Messreihen-Erzeuger, Verdikte (reine Funktionen) |
| `vhd_layout.py` | Fester Kopf (Stylesheet) |
| `vhd_learning.py` | Lernende Verhandler: Spiele, Hedge, Regret Matching, Nash-Lücke |
| `vhd_outside.py` | Außenoption: Rückwärtsrechnen, Schwellen, Handformeln |
| `vhd_pdf_export.py` | PDF-Export (fpdf2) |
| `vhd_presets.py` | Permalink (Begrenzen, Einrasten), Presets (Schnellstart, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `vhd_raster.py` | Rasterspiel in Brüchen, stationäre Gleichgewichte, Schranke |
| `vhd_rng.py` | wörtliche Kopie von `pearson-demo/prs_rng.py` (SplitMix64) |
| `vhd_rubinstein.py` | Formel, Rückwärtsrechnen, feste Kosten, kurze Runden |
| `vhd_scenario.py` | Einstellungen (Settings) und Rollen |
| `vhd_solutions.py` | Nash, Kalai–Smorodinsky, egalitär, utilitaristisch, Rubinstein bei gekrümmtem Nutzen |
| `vhd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Mini-Instanzen von Hand, Orakel, Eigenschaften, Aussagen, Messreihen, Randfälle, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Kopie-Test |

## Bewusst nicht umgesetzt

- **Hin- und Rückfahrt mit Flussungleichgewicht** (Dreiparteien-Modell, Raten für Hin- und Rückfahrt): der Reiter ist nicht gebaut; die Entscheidung fiel erst nach den Kern-Reitern, und die Modellannahmen (Wert je Sendung gleich, Rückfahrt leer ohne Rückfracht) belegen die Aussage nicht.
- **Private Information** (Verhandeln mit Geheimnissen, Verzögerung bei einseitiger privater Information): nicht gebaut; `myerson-satterthwaite-demo` rechnet private Werte ohne Ablauf in der Zeit.
- **Mehrparteien-Verhandlung** (Koalitionsverhandlung) und **Daten echter Verhandler**: nicht gebaut, keine eigenen Daten an Menschen.
- **Variante B der Außenoption** als Regler: sie ist in 0 von 2400 gemessenen Zellen gleich Variante A.
- **Q-Learning und weitere Lernregeln** für die Lernenden: nur Hedge und Regret Matching.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Suite hat 349 Tests. Die Messreihen entstehen mit `python tools/make_grid.py` (rund 3 s auf acht Prozessen); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Rechnung oder Konstanten die Messreihen überholt haben. Lokal läuft Python 3.14, die CI Python 3.12; unter 3.12 läuft die Suite erst in der CI (lokal nicht geprüft).

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Regeln setzen, verhandeln, kontrollieren](https://sebastianhanisch.net/konzepte-verhandeln.html).
