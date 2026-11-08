"""Inhalt der App an festen Fällen (DEMO-PLAYBOOK §12): was im Ergebnis, in den Tabellen und in den Verdikten steht."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(preset=None):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    if preset:
        at.sidebar.button(key=f"preset_{preset}").click().run()
        assert not at.exception, [str(e) for e in at.exception]
    return at


def _text(at):
    return "\n".join(e.value for kind in (at.markdown, at.caption, at.info, at.success, at.warning) for e in kind)


def _metrics(at):
    return {m.label: (m.value, m.delta) for m in at.metric}


def test_default_page_has_the_sections_and_the_default_result():
    at = _run()
    subs = [s.value for s in at.subheader]
    assert [s for s in subs if s.startswith("🔬")] == ["🔬 Experiment: Das Modell", "🔬 Experiment: Wie viel steckt im Beweis?", "🔬 Experiment: Orakel und gelernter Verzweiger"]
    assert "🚧 Wo die Annahmen enden" in subs
    m = _metrics(at)
    assert m["Optimum"] == ("10", "Untergrenze 9, 12 Container") and m["Engste Passung"] == ("58", "Beweis 47, Fund 11") and m["Gelernt"] == ("58", "±0 %")
    assert m["Orakel"] == ("58", "±0 %") and m["Beweisanteil"] == ("81 %", "Engste Passung")


def test_default_page_model_and_tables_quote_the_stored_numbers():
    text = _text(_run())
    assert "| Achsenabschnitt | 1,06 |" in text and "| sicherer Stapel | 2,44 |" in text and "| neue Blocker | -1,39 |" in text
    assert "5 420 Kindern" in text and "Trefferquote beim Training 79 %" in text
    assert "| 4 × 3 | 170 | 99 % | 98 % | 66 % |" in text and "| 4 × 4 | 213 | 92 % | 89 % | 58 % |" in text
    assert "| 4 × 3 (100) | 1 667 | 1 727 | 2 171 | 3 055 | 46 % |" in text and "| 4 × 4 (100) | 21 544 | 22 595 | 26 700 | 37 309 | 72 % |" in text
    assert "| 5 × 4 (40) | 61 598 / 28 069 | 65 019 / 29 797 | 35 152 / 539 | 62 703 / 25 290 | +5,6 % | −43 % |" in text


def test_default_page_verdicts_for_the_default_cell():
    at = _run()
    text = _text(at)
    assert "4 × 3: Von 1 667 Knoten der Engsten Passung stecken 759 im Beweis und 908 im Fund: der Beweis ist 46 %." in text
    assert "4 × 3: Gegenüber der Engsten Passung (1 667 Knoten) braucht der gelernte Verzweiger +3,6 % Knoten und das Orakel −14 %; das Orakel ist ein Richtwert für die Obergrenze." in text
    assert any(w.value.endswith("Der gelernte Verzweiger ist hier schlechter als die Regel.") for w in at.warning)


def test_proof_dominates_preset_shows_the_warning_and_the_oracle_loses_in_total():
    at = _run("Der Beweis dominiert (4 Stapel, 4 Lagen)")
    m = _metrics(at)
    assert m["Engste Passung"] == ("2 608", "Beweis 2 473, Fund 135") and m["Orakel"] == ("2 693", "+3,3 %") and m["Beweisanteil"][0] == "95 %"
    assert any(w.value.startswith("Optimum 13 (Untergrenze 10).") and "braucht 7,2 % mehr Knoten" in w.value for w in at.warning)


def test_one_stage_preset_says_there_is_no_proof():
    at = _run("Nur Fund, kein Beweis (4 Stapel, 3 Lagen)")
    assert _metrics(at)["Beweisanteil"][0] == "0,0 %" and any("keinen Beweis, nur den Fund" in i.value for i in at.info)


@pytest.mark.parametrize("stacks,layers", [(3, 2), (3, 4)])
def test_cells_without_stored_bands_do_not_crash(stacks, layers):
    at = _run()
    at.sidebar.slider(key="stacks_slider").set_value(stacks).run()
    at.sidebar.slider(key="layers_slider").set_value(layers).run()
    assert not at.exception
    text = _text(at)
    assert "wurde der Beweisanteil nicht vorberechnet" in text and "wurde das Orakel nicht vorberechnet" in text
