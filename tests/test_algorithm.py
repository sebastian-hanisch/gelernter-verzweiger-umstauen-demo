"""Suche mit austauschbarer Kinderreihenfolge, Merkmale und Reihenfolgen gegen Handrechnung (DEMO-PLAYBOOK §1b und §12)."""

import pytest

import gvl_algorithm as A
from gvl_scenario import make_bay


def test_bounds_and_split_by_hand(hand, counter):
    assert A.lb2(A.autopop(hand[0])[0]) == 4 and A.lb2(A.autopop(counter[0])[0]) == 4 and A.lb2(()) == 0
    st = hand[0]
    i, c, base = A.split_state(st)
    assert (i, c, base) == (0, 5, ((1,), (2, 3), (4, 6)))
    assert A.apply_child(base, c, 1) == ((), (2, 3, 5), (4, 6)) and A.apply_child(base, c, 2) == ((), (2, 3), (4, 6, 5))      # die 1 wird nach dem Hub abgerufen


def test_features_by_hand(hand):
    i, c, base = A.split_state(hand[0])
    assert A.features(base, i, c, 1, 4) == [1.0, 0.0, -0.5, 0.5, 0.0, 1.0, 0.5]                      # Stapel (2, 3): unsicher, Abstand (2 - 5) / 6, halb voll, neuer Blocker, LB2 des Kindes 3 / 6
    assert A.features(base, i, c, 2, 4) == pytest.approx([1.0, 0.0, -1 / 6, 0.5, 0.0, 1.0, 0.5])    # Stapel (4, 6): Abstand (4 - 5) / 6
    safe = A.features(((1,), (2,), (), (4,)), 0, 3, 3, 3)                                            # die 3 auf den Stapel (4): sicher (4 > 3), Abstand (4 - 3) / 4, kein neuer Blocker
    assert safe[1] == 1.0 and safe[2] == pytest.approx(0.25) and safe[5] == 0.0
    empty = A.features(((1,), (2, 3), ()), 0, 5, 2, 3)
    assert empty[4] == 1.0 and empty[3] == 0.0 and empty[2] == 1.0 and empty[1] == 1.0               # leerer Stapel: sicher, Abstand 1, Füllstand 0


def test_orders_by_hand(hand):
    i, c, base = A.split_state(hand[0])
    st = hand[0]
    assert A.order_fit(st, i, c, base, 4) == [2, 1] and A.order_worst(st, i, c, base, 4) == [1, 2]           # Engste Passung: der Stapel mit dem größten Minimum (Schaden am kleinsten) zuerst
    assert A.order_learned_factory([0, 0, -1, 0, 0, 0, 0])(st, i, c, base, 4) == [1, 2]                      # Gewicht -1 auf dem Abstand dreht die Reihenfolge um
    assert A.order_learned_factory([0, 1, 0, 0, 0, 0, 0])(st, i, c, base, 4) == [2, 1]                       # nur „sicher“: beide unsicher, Gleichstand: Engste Passung
    assert A.order_learned_factory([0, 0, 0, 0, 0, 0, 0])(st, i, c, base, 4) == A.order_fit(st, i, c, base, 4)


def test_random_order_is_a_deterministic_permutation(hand):
    i, c, base = A.split_state(hand[0])
    a, b = A.order_random_factory(7), A.order_random_factory(7)
    seq_a = [tuple(a(hand[0], i, c, base, 4)) for _ in range(6)]
    assert seq_a == [tuple(b(hand[0], i, c, base, 4)) for _ in range(6)]
    assert all(sorted(x) == [1, 2] for x in seq_a)


def test_search_by_hand_and_the_stage_properties(hand, counter):
    r = A.search(*hand)
    assert (r.optimum, r.stages, r.nodes, r.proof, r.find, r.capped) == (4, (5,), 5, 0, 5, False)       # eine Stufe: kein Beweis
    c = A.search(*counter)
    assert c.optimum == 4 and len(c.stages) == 1
    bay = make_bay(4, 3, 0)                                                                            # Optimum 10, Untergrenze 9: zwei Stufen
    r = A.search(bay, 5)
    assert (r.optimum, r.stages, r.nodes, r.proof, r.find) == (10, (47, 11), 58, 47, 11)
    assert len(r.stages) == r.optimum - A.lb2(A.autopop(bay)[0]) + 1                                    # die Budgetstufen zählen von der Untergrenze bis zum Optimum


def test_all_orders_find_the_same_optimum_and_the_worst_costs_more(hand):
    bay = make_bay(4, 3, 0)
    res = {n: A.search(bay, 5, o) for n, o in (("fit", A.order_fit), ("worst", A.order_worst), ("random", A.order_random_factory(0)), ("oracle", A.order_oracle))}
    assert {r.optimum for r in res.values()} == {10}
    assert [res[n].stages for n in ("fit", "oracle")] == [(47, 11), (47, 11)] and res["worst"].stages == (47, 61) and res["random"].stages == (47, 14)
    assert A.search(*hand, A.order_worst).optimum == 4


def test_the_proof_stages_do_not_depend_on_the_order_much_and_the_find_does():
    bay = make_bay(4, 4, 23)
    fit, worst, oracle = A.search(bay, 6), A.search(bay, 6, A.order_worst), A.search(bay, 6, A.order_oracle)
    assert fit.stages == (140, 546, 1787, 135) and oracle.stages == (140, 574, 1965, 14) and worst.stages == (140, 607, 2212, 404)
    assert fit.proof == 2473 and oracle.find == 14 < fit.find < worst.find                           # die letzte Stufe hängt an der Reihenfolge, die Beweisstufen kaum
    assert abs(oracle.proof / fit.proof - 1) < 0.1 and abs(worst.proof / fit.proof - 1) < 0.2                # die Beweisstufen ändern sich um höchstens 20 %


def test_node_limit_returns_no_optimum():
    bay = make_bay(4, 4, 23)
    r = A.search(bay, 6, A.order_fit, cap=100)
    assert r.optimum is None and r.capped and r.nodes == 100 and r.stages == (100,)
    r = A.search(bay, 6, A.order_fit, cap=300)
    assert r.optimum is None and r.capped and r.stages == (140, 160) and r.nodes == 300


def test_child_value_and_the_oracle_order(hand):
    i, c, base = A.split_state(hand[0])
    v1, v2 = A.child_value(A.apply_child(base, c, 1), 4), A.child_value(A.apply_child(base, c, 2), 4)
    assert min(v1, v2) == 3 and A.child_value(((), (), ()), 4) == 0                                   # Optimum 4 = ein Hub + Rest 3
    order = A.order_oracle(hand[0], i, c, base, 4)
    assert order[0] == (1 if v1 < v2 else 2 if v2 < v1 else A.order_fit(hand[0], i, c, base, 4)[0]) and sorted(order) == [1, 2]


@pytest.mark.parametrize("sd", range(6))
def test_orders_agree_on_random_bays(sd):
    bay = make_bay(3, 3, sd)
    opts = {A.search(bay, 5, o).optimum for o in (A.order_fit, A.order_worst, A.order_random_factory(sd))}
    assert len(opts) == 1 and None not in opts
