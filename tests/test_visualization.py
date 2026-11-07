"""Abbildungen an synthetischen Daten: Werte, Farben, gesperrte Achsen (DEMO-PLAYBOOK §12)."""

import plotly.graph_objects as go
import pytest

import gvl_algorithm as A
import gvl_constants as C
import gvl_visualization as V


def _locked(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_blocker_cells_and_the_bay_chart(hand):
    assert V.blocker_cells(((1, 5), (2, 3), (4, 6))) == {(0, 1), (1, 1), (2, 1)}
    fig = V.build_bay(*hand[:1], 4, target=1)
    assert _locked(fig) and len(fig.data) == 4 and list(fig.data[0].marker.color) == [C.COLOR_TARGET, C.COLOR_CELL, C.COLOR_CELL]
    assert list(fig.data[1].marker.color) == [C.COLOR_BLOCK] * 3 and [s.y0 for s in fig.layout.shapes] == [4]


def test_stage_chart_stacks_proof_and_find_and_leaves_out_capped_runs():
    results = {"fit": A.SearchResult(10, (47, 11), False), "learned": A.SearchResult(10, (47, 25), False), "oracle": A.SearchResult(None, (60,), True), "random": A.SearchResult(10, (47, 14), False)}
    fig = V.build_stages(results)
    assert _locked(fig) and fig.layout.barmode == "stack" and list(fig.data[0].x) == ["Engste Passung", "gelernt", "zufällig"]
    assert list(fig.data[0].y) == [47, 47, 47] and list(fig.data[1].y) == [11, 25, 14] and [t.name for t in fig.data] == ["Beweis (Stufen vor der letzten)", "Fund (letzte Stufe)"]


def _cells():
    def o(nodes, find):
        return {"nodes_sum": nodes, "find_sum": find, "proof_sum": nodes - find}
    return {"4,4": {"orders": {"fit": o(14420, 4485), "learned": o(14897, 4493), "oracle": o(10913, 668), "random": o(16268, 5428)}},
            "4,3": {"orders": {"fit": o(1667, 908), "learned": o(1727, 968), "oracle": o(1440, 681), "random": o(2171, 1403)}},
            "5,3": {"orders": {"fit": o(2955, 921), "learned": o(3109, 1075), "oracle": o(2836, 794), "random": o(5767, 3705)}}}


def test_proof_chart_is_ordered_by_container_count_and_stacked():
    fig = V.build_proof(_cells())
    assert _locked(fig) and fig.layout.barmode == "stack" and list(fig.data[0].x) == ["4 × 3", "5 × 3", "4 × 4"]
    assert list(fig.data[0].y) == [759, 2034, 9935] and list(fig.data[1].y) == [908, 921, 4485]


def test_relative_chart_shows_percent_of_the_fit_order():
    fig = V.build_relative(_cells())
    assert _locked(fig) and [t.name for t in fig.data] == ["gelernt", "Orakel", "zufällig"] and list(fig.data[0].x) == ["4 × 3", "5 × 3", "4 × 4"]
    assert list(fig.data[1].y) == pytest.approx([100 * 1440 / 1667, 100 * 2836 / 2955, 100 * 10913 / 14420]) and [t.marker.color for t in fig.data] == [C.COLOR_ORDER[n] for n in ("learned", "oracle", "random")]
    assert list(V.build_relative(_cells(), names=("random",)).data[0].y)[0] == pytest.approx(100 * 2171 / 1667)


def test_all_figures_are_plotly_figures(hand):
    assert all(isinstance(f, go.Figure) for f in (V.build_bay(((1,),), 2), V.build_proof(_cells()), V.build_relative(_cells())))
