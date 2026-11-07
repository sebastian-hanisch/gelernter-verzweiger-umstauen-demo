"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12)."""

import pytest

import gvl_algorithm as A
import gvl_constants as C
import gvl_evaluation as ev

W = [1.057, 2.445, 0.436, 0.364, -1.347, -1.388, -0.608]


def _row(seed, opt, **orders):
    return {"seed": seed, "opt": opt, "orders": {n: {"nodes": v[0], "proof": v[1], "find": v[0] - v[1], "capped": len(v) > 2 and v[2]} for n, v in orders.items()}}


def test_order_stats_by_hand():
    rows = [_row(0, 5, fit=(10, 4), learned=(12, 4), random=(20, 5), worst=(30, 6)), _row(1, 6, fit=(30, 20), learned=(30, 20), random=(31, 20), worst=(60, 20)),
            _row(2, 4, fit=(5, 0), learned=(5, 0), random=(9, 0), worst=(11, 0))]
    s = ev.order_stats(rows, ev.NAMES)
    assert (s["n"], s["n_ok"]) == (3, 3)
    assert s["orders"]["fit"] == {"nodes_sum": 45, "nodes_median": 10.0, "find_sum": 21, "find_median": 6.0, "proof_sum": 24}
    assert s["orders"]["learned"]["nodes_sum"] == 47 and s["orders"]["worst"]["proof_sum"] == 26 and s["orders"]["random"]["find_sum"] == 20 - 5 + 31 - 20 + 9 - 0
    assert s["proof_share"] == pytest.approx(24 / 45)


def test_order_stats_skip_bays_where_any_order_hit_the_limit():
    rows = [_row(0, 5, fit=(10, 4), learned=(12, 4)), _row(1, None, fit=(100, 50, True), learned=(100, 50, False)), _row(2, 4, fit=(6, 0), learned=(6, 0))]
    s = ev.order_stats(rows, ("fit", "learned"))
    assert (s["n"], s["n_ok"]) == (3, 2) and s["orders"]["fit"]["nodes_sum"] == 16 and s["orders"]["fit"]["nodes_median"] == 8.0


def test_analysis_without_the_oracle_when_the_bay_is_too_big():
    a = ev.analyse(ev.Settings(4, 4, 2, 25), W)                                                   # 16 Container: das Orakel ist noch dabei
    assert "oracle" in a.results and a.containers == 16 and a.results["oracle"].stages == (12,)
    small = ev.analyse(ev.Settings(4, 3, 2, 0), W)
    assert list(small.results) == list(ev.ALL_NAMES) and small.optimum == 10 and small.lb == 9
    assert small.results["fit"].stages == (47, 11) and small.results["oracle"].nodes == 58


def test_analysis_over_the_oracle_limit_has_no_oracle(monkeypatch):
    monkeypatch.setattr(C, "ORACLE_MAX_CONTAINERS", 12)
    a = ev.analyse(ev.Settings(5, 3, 2, 0), W)
    assert "oracle" not in a.results and a.containers == 15
    rows = ev.result_rows(a)
    assert rows[2] == {"name": "Orakel", "nodes": "nicht gerechnet", "proof": "–", "find": "–", "stages": "–"}


def test_analysis_reports_the_limit_without_a_value(monkeypatch):
    monkeypatch.setattr(C, "LIVE_CAP", 20)
    a = ev.analyse(ev.Settings(4, 4, 2, 23), W)
    assert a.optimum is None and all(r.capped for r in a.results.values())
    assert ev.verdict_bay(a) == ("info", "Das Optimum ist nicht bewiesen (Knotenlimit 20); die Reihenfolgen finden es nicht innerhalb der Grenze.")
    assert ev.result_rows(a)[0]["nodes"] == "am Limit"


def test_orders_may_disagree_only_if_the_code_is_wrong(monkeypatch):
    real = A.search

    def broken(bay, height, order=A.order_fit, cap=1_000_000):
        r = real(bay, height, order, cap)
        return A.SearchResult(r.optimum + 1 if order is A.order_worst else r.optimum, r.stages, r.capped)
    monkeypatch.setattr(A, "search", broken)
    with pytest.raises(AssertionError):
        ev.order_rows(4, 3, 2, range(2), ("fit", "worst"), W)


def test_make_order_rejects_unknown_names():
    with pytest.raises(ValueError):
        ev.make_order("magisch", W)
    assert ev.make_order("fit", W) is A.order_fit and ev.make_order("worst", W) is A.order_worst


def test_number_formats():
    assert ev.num(5) == "5" and ev.num(61598) == "61 598"
    assert ev.num_f(5.0) == "5" and ev.num_f(71.5) == "71,5" and ev.num_f(1234.0) == "1 234"
    assert ev.percent_text(0.0259) == "2,6 %" and ev.percent_text(0.455) == "46 %" and ev.percent_text(0.0625) == "6,3 %" and ev.percent_text(1.0) == "100 %"
    assert ev.signed_percent_text(0.036) == "+3,6 %" and ev.signed_percent_text(-0.136) == "−14 %" and ev.signed_percent_text(0.0) == "±0 %" and ev.signed_percent_text(-0.00001) == "±0 %"
    assert ev.signed_percent_text(0.0004) == "±0 %" and ev.signed_percent_text(0.94) == "+94 %"
    assert ev.key(4, 3) == "4,3"
