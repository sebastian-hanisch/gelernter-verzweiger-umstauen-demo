"""Presets: jede Einstellung hat die Eigenschaft, für die sie steht (DEMO-PLAYBOOK §12), und die Hilfetexte nennen nur gemessene Zahlen."""

import pytest

import gvl_constants as C
import gvl_evaluation as ev
import gvl_presets as P


def _weights():
    return ev.load_bands()["modell"]["weights"]


def _analyse(name):
    p = C.PRESETS[name]
    return ev.analyse(ev.Settings(p["stacks"], p["layers"], p["extra"], p["seed"]), _weights())


def test_every_preset_is_inside_the_slider_bounds_and_has_help():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            lo, hi = P.bounds(state_key)
            assert lo <= p[key] <= hi, (name, key)
    assert set(C.PRESET_HELP) == set(C.PRESETS) and all(C.PRESET_HELP.values())


def test_everyday_bay_is_a_two_stage_search_where_all_orders_agree():
    a = _analyse("Alltag (4 Stapel, 3 Lagen)")
    assert a.optimum == 10 and a.lb == 9 and a.results["fit"].stages == (47, 11)
    assert {n: a.results[n].stages for n in ("learned", "oracle")} == {"learned": (47, 11), "oracle": (47, 11)} and a.results["worst"].stages == (47, 61)


def test_one_stage_bay_has_no_proof():
    a = _analyse("Nur Fund, kein Beweis (4 Stapel, 3 Lagen)")
    assert a.optimum == 7 == a.lb and all(len(r.stages) == 1 for r in a.results.values()) and a.results["fit"].proof == 0
    assert ev.verdict_bay(a)[1].endswith("Es gibt nur eine Stufe und keinen Beweis, nur den Fund.")


def test_proof_dominates_bay_and_the_oracle_does_not_win_in_total():
    a = _analyse("Der Beweis dominiert (4 Stapel, 4 Lagen)")
    fit, oracle, learned = a.results["fit"], a.results["oracle"], a.results["learned"]
    assert (fit.stages, oracle.stages, learned.stages) == ((140, 546, 1787, 135), (140, 574, 1965, 14), (140, 585, 2058, 14))
    assert fit.proof / fit.nodes > 0.9 and oracle.find < fit.find and oracle.nodes > fit.nodes and ev.verdict_bay(a)[0] == "warning"


def test_learned_better_than_the_rule_bay():
    a = _analyse("Gelernt besser als die Regel (4 Stapel, 4 Lagen)")
    nodes = {n: r.nodes for n, r in a.results.items()}
    assert nodes == {"fit": 200, "learned": 145, "oracle": 12, "random": 40, "worst": 28} and a.optimum == 11 == a.lb


def test_learned_worse_than_the_rule_bay():
    a = _analyse("Gelernt schlechter als die Regel (4 Stapel, 3 Lagen)")
    assert (a.results["fit"].nodes, a.results["learned"].nodes, a.results["oracle"].nodes) == (17, 33, 17) and ev.verdict_bay(a)[0] == "warning"


def test_the_default_settings_are_the_first_preset():
    assert {s: P.SETTING_SPECS[s].default for s in P.SETTING_SPECS} == {"stacks_slider": 4, "layers_slider": 3, "extra_slider": 2, "seed_input": 0}
    assert C.PRESETS["Alltag (4 Stapel, 3 Lagen)"] == {"stacks": 4, "layers": 3, "extra": 2, "seed": 0}


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_apply_preset_writes_every_setting(name, monkeypatch):
    state = {}
    monkeypatch.setattr(P.st, "session_state", state)
    P.apply_preset(name)
    assert {k: state[sk] for k, sk in P.PRESET_KEYS.items()} == {k: C.PRESETS[name][k] for k in P.PRESET_KEYS}
    assert P.aktives_preset() == name
