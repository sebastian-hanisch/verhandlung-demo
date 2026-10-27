"""PDF-Export: erzeugt ein gültiges PDF für jedes Preset und an den Reglergrenzen; Kernschriften verkraften die Zeichen der Texte."""

import pytest

import vhd_constants as C
import vhd_evaluation as ev
import vhd_pdf_export as pdf_export
from vhd_scenario import Settings


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_pdf_builds_for_every_preset(name):
    data = pdf_export.build_pdf(ev.analyse(Settings(**C.PRESETS[name])))
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_pdf_builds_at_the_regler_limits():
    for s in (Settings(d1=0.0, d2=0.0, t=1, first=2), Settings(d1=C.D_MAX, d2=C.D_MAX, t=C.T_MAX, g=10000, c1=1.2, c2=0.3, o=C.O_MAX, variant="C", curv=C.CURV_MIN, scale=10.0)):
        assert pdf_export.build_pdf(ev.analyse(s))[:5] == b"%PDF-"


def test_text_replacement_removes_characters_the_core_fonts_do_not_know():
    text = pdf_export._text("δ₁ → 0,9 – „Anteil“ ≤ 2·10⁻⁵ € Δ")
    assert text.encode("latin-1") and "delta1" in text and "->" in text and "EUR" in text and "Delta" in text and "10-5" in text
