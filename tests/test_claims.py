"""Aussagen der Demo und des READMEs gegen die vorberechneten Bänder und gegen eine frische Nachrechnung kleiner Zellen (DEMO-PLAYBOOK §5, §12).

Alle Größen sind deterministisch (feste Seeds, Knotenzahlen, keine Zeiten) und müssen bei der Nachrechnung gleich sein."""

import numpy as np
import pytest

import gvl_constants as C
import gvl_evaluation as ev


def K(s, l):
    return ev.key(s, l)


def _bands():
    return ev.load_bands()


def test_the_stored_cells_cover_exactly_the_cells_of_the_constants():
    b = _bands()
    assert {k: v["n"] for k, v in b["beweis"].items()} == {K(s, l): n for s, l, n in C.PROOF_CELLS}
    assert {k: v["n"] for k, v in b["orakel"].items()} == {K(s, l): n for s, l, n in C.ORACLE_CELLS}
    assert set(b["top1"]) == {K(s, l) for s, l, _ in C.TOP1_CELLS}
    assert all(v["n_ok"] == v["n"] for v in list(b["beweis"].values()) + list(b["orakel"].values()))              # keine Bucht am Limit


def test_the_model_the_readme_quotes():
    m = _bands()["modell"]
    assert [round(w, 2) for w in m["weights"]] == [1.06, 2.44, 0.44, 0.36, -1.35, -1.39, -0.61] and m["n_children"] == 5420 and m["cells"] == [list(c) for c in C.TRAIN_CELLS]
    assert round(m["share_optimal"], 3) == 0.644 and round(m["accuracy"], 3) == 0.791 and m["seed0"] == 1000 == C.TRAIN_SEED0


def test_a_stored_stage_split_is_consistent():
    for cells in (_bands()["beweis"], _bands()["orakel"]):
        for v in cells.values():
            for o in v["orders"].values():
                assert o["nodes_sum"] == o["find_sum"] + o["proof_sum"] and o["nodes_median"] <= o["nodes_sum"]
            assert v["proof_share"] == pytest.approx(v["orders"]["fit"]["proof_sum"] / v["orders"]["fit"]["nodes_sum"])


def test_the_proof_numbers_the_readme_quotes():
    b = _bands()["beweis"]
    assert [b[K(s, l)]["orders"]["fit"]["nodes_sum"] for s, l, _ in C.PROOF_CELLS] == [1667, 2955, 21544, 77285, 1811]
    assert [round(b[K(s, l)]["proof_share"], 2) for s, l, _ in C.PROOF_CELLS] == [0.46, 0.69, 0.72, 0.63, 0.70]
    assert min(b[K(s, l)]["proof_share"] for s, l, _ in C.PROOF_CELLS) > 0.45 and max(b[K(s, l)]["proof_share"] for s, l, _ in C.PROOF_CELLS) < 0.73
    ratios = [b[K(s, l)]["orders"]["worst"]["nodes_sum"] / b[K(s, l)]["orders"]["fit"]["nodes_sum"] for s, l, _ in C.PROOF_CELLS]
    assert round(min(ratios), 1) == 1.5 and round(max(ratios), 1) == 3.4
    rand = [b[K(s, l)]["orders"]["random"]["nodes_sum"] / b[K(s, l)]["orders"]["fit"]["nodes_sum"] for s, l, _ in C.PROOF_CELLS]
    assert round(min(rand), 1) == 1.1 and round(max(rand), 1) == 2.1                                                    # zufällig: 1,1- bis 2,1-fach
    for s, l, _ in C.PROOF_CELLS:                                                                  # Zufall und schlechteste zuerst sind nie besser als die Regel
        o = b[K(s, l)]["orders"]
        assert o["fit"]["nodes_sum"] < o["random"]["nodes_sum"] < o["worst"]["nodes_sum"]


def test_the_oracle_and_the_learned_numbers_the_readme_quotes():
    b = _bands()["orakel"]
    sums = {K(s, l): {n: b[K(s, l)]["orders"][n]["nodes_sum"] for n in ("fit", "learned", "oracle", "random")} for s, l, _ in C.ORACLE_CELLS}
    assert sums["4,3"] == {"fit": 1667, "learned": 1727, "oracle": 1440, "random": 2171} and sums["5,4"] == {"fit": 61598, "learned": 65019, "oracle": 35152, "random": 62703}
    assert [round(100 * (1 - v["oracle"] / v["fit"])) for v in sums.values()] == [14, 4, 24, 43]                      # Obergrenze: das Orakel spart 14, 4, 24, 43 %
    assert [round(100 * (v["learned"] / v["fit"] - 1)) for v in sums.values()] == [4, 5, 3, 6]                       # der gelernte Verzweiger liegt über der Regel
    assert all(v["learned"] > v["fit"] > v["oracle"] for v in sums.values())
    assert b["5,4"]["orders"]["oracle"]["find_sum"] == 539 and b["5,4"]["orders"]["fit"]["find_sum"] == 28069 and b["4,4"]["orders"]["oracle"]["find_sum"] == 668
    proofs = {n: b["5,4"]["orders"][n]["proof_sum"] for n in ("fit", "learned", "oracle", "random")}
    assert proofs == {"fit": 33529, "learned": 35222, "oracle": 34613, "random": 37413}                               # die Beweisknoten ändern sich um höchstens 12 %
    assert max(proofs.values()) / min(proofs.values()) < 1.12


def test_the_top_one_hit_rates_the_readme_quotes():
    t = _bands()["top1"]
    assert {k: (v["fit"], v["learned"], v["random"], v["states"]) for k, v in t.items()} == {"4,3": (169, 166, 112, 170), "4,4": (195, 189, 124, 213), "5,3": (211, 206, 129, 213)}
    assert all(v["fit"] / v["states"] > 0.9 for v in t.values()) and all(v["fit"] >= v["learned"] > v["random"] for v in t.values())


@pytest.mark.parametrize("stacks,layers,n", [(4, 3, 100), (6, 3, 60)])
def test_a_fresh_recomputation_of_proof_cells_reproduces_the_stored_band(stacks, layers, n):
    w = _bands()["modell"]["weights"]
    fresh = ev.order_stats(ev.order_rows(stacks, layers, 2, range(n), ev.NAMES, w), ev.NAMES)
    assert fresh == _bands()["beweis"][K(stacks, layers)]


def test_a_fresh_recomputation_of_an_oracle_cell_and_the_top_one_cell_reproduces_the_stored_bands():
    w = _bands()["modell"]["weights"]
    assert ev.order_stats(ev.order_rows(5, 3, 2, range(100), ev.ALL_NAMES, w), ev.ALL_NAMES) == _bands()["orakel"][K(5, 3)]
    assert ev.top1_stats(4, 3, 2, range(30), w) == _bands()["top1"][K(4, 3)]


def test_the_stored_model_is_the_result_of_the_training_on_a_small_training_set_up_to_scale():
    """Teilprüfung des Trainings: das Training auf 3 × 3 allein (12 Buchten) trifft nicht dieselben Gewichte, aber dieselben Vorzeichen für „sicherer Stapel“ und „neue Blocker“ wie das gespeicherte Modell."""
    from gvl_model import train
    w_small, _, _, _ = train(((3, 3, 12),))
    stored = np.array(_bands()["modell"]["weights"])
    assert np.sign(w_small[1]) == np.sign(stored[1]) == 1 and np.sign(w_small[5]) == np.sign(stored[5]) == -1
