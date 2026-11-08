"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import gvl_algorithm as A
import gvl_constants as C
import gvl_evaluation as ev


def _res(stages, opt=10, capped=False):
    return A.SearchResult(None if capped else opt, tuple(stages), capped)


def _a(fit, learned, oracle=None, opt=10, lb=9):
    results = {"fit": fit, "learned": learned, "random": _res((47, 14)), "worst": _res((47, 61))}
    if oracle is not None:
        results["oracle"] = oracle
    return ev.Analysis(((1,),), 5, 12, lb, results)


def test_bay_verdict_states_and_numbers():
    kind, text = ev.verdict_bay(_a(_res((47, 11)), _res((47, 11)), _res((47, 11))))
    assert kind == "info" and text == ("Optimum 10 (Untergrenze 9). Engste Passung: 58 Knoten (Beweis 47, Fund 11); gelernt: 58 Knoten (Beweis 47, Fund 11); "
                                       "Orakel: 58 Knoten (Beweis 47, Fund 11). Der gelernte Verzweiger ist hier nicht besser als die Engste Passung (±0 %).")
    kind, text = ev.verdict_bay(_a(_res((47, 11)), _res((47, 25)), _res((47, 11))))
    assert kind == "warning" and text.endswith("Der gelernte Verzweiger braucht 24 % mehr Knoten als die Engste Passung.")
    kind, text = ev.verdict_bay(_a(_res((200,), 11), _res((145,), 11), _res((12,), 11), opt=11, lb=11))
    assert kind == "info" and text.endswith("Das Optimum ist die Untergrenze: Es gibt nur eine Stufe und keinen Beweis, nur den Fund.")
    kind, text = ev.verdict_bay(_a(_res((100, 100)), _res((100, 90))))
    assert kind == "success" and text.endswith("Der gelernte Verzweiger spart 5,0 % der Knoten gegenüber der Engsten Passung.")          # genau 5 % zählt als spürbar
    assert ev.verdict_bay(_a(_res((100, 100)), _res((100, 91))))[0] == "info" and C.LEARN_GAIN == 0.05
    assert ev.verdict_bay(_a(_res((100, 100)), _res((100, 101))))[0] == "warning" and C.LEARN_LOSS == 0.0                                  # schon ein Knoten mehr ist „schlechter“


def test_bay_verdict_without_oracle_and_with_limits():
    kind, text = ev.verdict_bay(_a(_res((47, 11)), _res((47, 11))))
    assert "Orakel" not in text
    kind, text = ev.verdict_bay(_a(_res((10,), capped=True), _res((10,)), opt=10))
    assert kind == "info" and text == "Optimum 10; Engste Passung: am Limit, gelernt: 10 Knoten (Beweis 0, Fund 10): kein Vergleich möglich."
    capped = ev.Analysis(((1,),), 5, 12, 9, {n: _res((10,), capped=True) for n in ev.ALL_NAMES})
    assert ev.verdict_bay(capped) == ("info", f"Das Optimum ist nicht bewiesen (Knotenlimit {ev.num(C.LIVE_CAP)}); die Reihenfolgen finden es nicht innerhalb der Grenze.")


def test_result_rows_by_hand():
    rows = ev.result_rows(_a(_res((47, 11)), _res((47, 25)), _res((47, 11))))
    assert [r["name"] for r in rows] == ["Engste Passung", "gelernt", "Orakel", "zufällig", "schlechteste zuerst"]
    assert rows[1] == {"name": "gelernt", "nodes": "72", "proof": "47", "find": "25", "stages": "47 / 25"} and rows[4]["stages"] == "47 / 61"
    big = ev.result_rows(_a(_res((1500, 20)), _res((1500, 30))))
    assert big[0]["nodes"] == "1 520" and big[0]["stages"] == "1 500 / 20" and big[2]["nodes"] == "nicht gerechnet"


def _stats(fit_nodes=1000, fit_find=300, learned=1000, oracle=700, random=1500, n=100):
    def o(nodes, find):
        return {"nodes_sum": nodes, "nodes_median": float(nodes) / n, "find_sum": find, "find_median": float(find) / n, "proof_sum": nodes - find}
    return {"n": n, "n_ok": n, "orders": {"fit": o(fit_nodes, fit_find), "learned": o(learned, fit_find), "oracle": o(oracle, 10), "random": o(random, 700), "worst": o(2 * fit_nodes, 900)},
            "proof_share": (fit_nodes - fit_find) / fit_nodes}


def test_proof_verdict_two_states_and_numbers():
    kind, text = ev.verdict_proof(_stats(1667, 908), "4 × 3")
    assert kind == "info" and text == ("4 × 3: Von 1 667 Knoten der Engsten Passung stecken 759 im Beweis und 908 im Fund: der Beweis ist 46 %. "
                                       "Der Fund ist der größere Teil: Eine bessere Reihenfolge hat hier mehr Spielraum.")
    kind, text = ev.verdict_proof(_stats(21544, 5964), "4 × 4")
    assert kind == "warning" and text.endswith("Der Beweis dominiert: Eine bessere Reihenfolge kann kaum mehr als den Fund sparen.") and "der Beweis ist 72 %" in text
    assert ev.verdict_proof(_stats(1000, 500), "x")[0] == "warning" and ev.verdict_proof(_stats(1000, 501), "x")[0] == "info" and C.PROOF_DOMINATES == 0.5      # genau die Hälfte zählt


def test_learning_verdict_three_states_and_numbers():
    kind, text = ev.verdict_learning(_stats(1000, 300, learned=1040, oracle=860), "5 × 3")
    assert kind == "warning" and text == ("5 × 3: Gegenüber der Engsten Passung (1 000 Knoten) braucht der gelernte Verzweiger +4,0 % Knoten und das Orakel −14 %; "
                                          "das Orakel ist ein Richtwert für die Obergrenze. Der gelernte Verzweiger ist hier schlechter als die Regel.")
    kind, text = ev.verdict_learning(_stats(learned=940, oracle=500), "x")
    assert kind == "success" and "−6,0 % Knoten und das Orakel −50 %" in text and text.endswith("Der gelernte Verzweiger spart hier spürbar Knoten.")
    kind, text = ev.verdict_learning(_stats(learned=1000, oracle=700), "x")
    assert kind == "info" and "±0 % Knoten" in text and text.endswith("Der gelernte Verzweiger ist hier nicht besser als die Regel.")
    assert ev.verdict_learning(_stats(learned=950), "x")[0] == "success" and ev.verdict_learning(_stats(learned=951), "x")[0] == "info" and ev.verdict_learning(_stats(learned=1001), "x")[0] == "warning"
