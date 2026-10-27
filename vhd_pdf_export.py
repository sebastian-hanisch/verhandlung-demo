"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Ergebnis der Hauptansicht und die Verdikte der Reiter. Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import vhd_constants as C
import vhd_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen, hochgestellte Ziffern)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "δ": "delta", "α": "alpha", "β": "beta", "θ": "theta",
             "φ": "phi", "η": "eta", "Δ": "Delta", "₁": "1", "₂": "2", "²": "^2", "⁻": "-", "⁵": "5", "⁶": "6", "⁸": "8", "⁹": "9", "¹": "1", "⁰": "0", "⁴": "4", "⁷": "7", "³": "3", "½": "1/2",
             "…": "...", "ˆ": "", "Σ": "Summe"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Verhandlung: wer Zeit hat, bekommt mehr"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Geduld Verlader {s.d1:g}, Spediteur {s.d2:g}; Horizont T = {s.t}; es bietet zuerst: {C.FIRSTS[s.first]}; Raster G = {s.g}, Regel {s.rule}.")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Ergebnis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Verlader {ev.pct(a.x1)}, Spediteur {ev.pct(a.x2)} des Ueberschusses; Formel (unendlicher Horizont) fuer den Verlader {ev.pct(a.limit_x1)}.")
    _para(pdf, 5.5, ev.verdict_main(a)[1])
    for title, verdict in (("Geduld", ev.verdict_patience), ("Zeit und Horizont", ev.verdict_horizon), ("Kurze Runden", ev.verdict_short_rounds), ("Feste Kosten", ev.verdict_costs),
                           ("Aussenoption", ev.verdict_outside), ("Loesungskonzepte", ev.verdict_solutions), ("Raster und Gleichstand", ev.verdict_grid)):
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        _para(pdf, 5.5, verdict(s)[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Modellrechnung mit vollstaendiger Information; Eigennutz, Geduld und Nutzenfunktionen sind Annahmen, keine Aussagen ueber echte Verhandler. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
