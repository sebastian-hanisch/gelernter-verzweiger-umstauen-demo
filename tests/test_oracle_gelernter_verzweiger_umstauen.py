"""Orakel-Test gelernter-verzweiger-umstauen-demo (ORACLE-PLAYBOOK, DEMO-PLAYBOOK §12): Optimum, Kindwerte und Trefferquote gegen ANDERE Rechenwege.

Eigene, unabhängige Umsetzungen im Test: (a) das Optimum jedes Zustands per vollständiger Rekursion über alle Hübe (ohne Untergrenze, ohne IDA*), (b) die Merkmale eines Kindes aus der
Definition, (c) die Trefferquote „erstes Kind auf einem optimalen Weg“ aus den rekursiven Kindwerten. Vorher an den Handzahlen geprüft: Hand-Bucht Optimum 4, Kindwert 3."""

import functools

import pytest

import gvl_algorithm as A
from gvl_scenario import make_bay
from gvl_model import first_child_optimal

INF = 10 ** 9


def pop_all(state):
    st = [list(s) for s in state]
    while True:
        nums = [c for s in st for c in s]
        if not nums:
            return tuple(tuple(s) for s in st)
        t = min(nums)
        k = next(i for i, s in enumerate(st) if t in s)
        if st[k][-1] != t:
            return tuple(tuple(s) for s in st)
        st[k].pop()


def ref_children(state, height):
    """[(Zielstapel j, Kindzustand nach Hub und Abruf)] des Zustands (nach dem Abruf), unabhängig aus den Regeln."""
    nums = [c for s in state for c in s]
    t = min(nums)
    k = next(i for i, s in enumerate(state) if t in s)
    out = []
    for j in range(len(state)):
        if j != k and len(state[j]) < height:
            new = [list(s) for s in state]
            new[j].append(new[k].pop())
            out.append((j, pop_all(new)))
    return out


@functools.lru_cache(maxsize=None)
def ref_value(state, height):
    """Optimum eines Zustands (nach dem Abruf) durch vollständige Rekursion; kanonisch gemerkt."""
    if not any(state):
        return 0
    vals = [1 + ref_value(tuple(sorted(c)), height) for _, c in ref_children(state, height)]
    return min(vals) if vals else INF


def ref_opt(bay, height):
    return ref_value(tuple(sorted(pop_all(bay))), height)


def ref_features(state, c_index, c, j, height):
    """Merkmale aus der Definition (ohne die Hilfsfunktionen des Kerns)."""
    base = [list(s) for s in state]
    base[c_index].pop()
    n = sum(len(s) for s in base) + 1
    dest = base[j]
    m = min(dest) if dest else 10 ** 9
    after = [list(s) for s in base]
    after[j].append(c)
    child = pop_all(after)
    t = min((x for s in child for x in s), default=None)

    def blockers(stacks):
        total = 0
        for s in stacks:
            for idx, x in enumerate(s):
                if any(y < x for y in s[:idx]):
                    total += 1
        return total

    def lb2(stacks):
        if t is None:
            return 0
        k = next(i for i, s in enumerate(stacks) if t in s)
        above = stacks[k][stacks[k].index(t) + 1:]
        others = [min(s) if s else 10 ** 9 for q, s in enumerate(stacks) if q != k]
        top = max(others) if others else -1
        return blockers(stacks) + sum(1 for x in above if x > top)

    return [1.0, 1.0 if m > c else 0.0, (min(m, n + 1) - c) / n if dest else 1.0, len(dest) / height, 1.0 if not dest else 0.0,
            float(blockers([tuple(dest) + (c,)]) - blockers([tuple(dest)])), lb2(child) / n]


@pytest.mark.parametrize("stacks,layers,seeds", [(3, 2, 40), (3, 3, 30), (4, 3, 20)])
def test_the_search_optimum_equals_the_independent_recursion_for_every_order(stacks, layers, seeds):
    h = layers + 2
    for sd in range(seeds):
        bay = make_bay(stacks, layers, sd)
        ref = ref_opt(bay, h)
        for order in (A.order_fit, A.order_worst, A.order_random_factory(sd), A.order_learned_factory([1.0, 2.4, 0.4, 0.4, -1.3, -1.4, -0.6])):
            assert A.search(bay, h, order).optimum == ref, (stacks, layers, sd)


def test_the_stage_count_is_the_distance_from_the_bound_to_the_optimum():
    for sd in range(40):
        bay = make_bay(4, 3, sd)
        r = A.search(bay, 5)
        assert len(r.stages) == r.optimum - A.lb2(A.autopop(bay)[0]) + 1 and r.nodes == sum(r.stages) and r.proof + r.find == r.nodes and all(x >= 1 for x in r.stages)


def test_the_child_values_of_the_oracle_equal_the_independent_recursion():
    for sd in range(40):
        bay = make_bay(4, 3, sd)
        st = pop_all(bay)
        if not any(st):
            continue
        i, c, base = A.split_state(st)
        for j in A.child_targets(base, i, 5, c):
            child = A.apply_child(base, c, j)
            assert A.child_value(child, 5) == (0 if not any(child) else ref_value(tuple(sorted(child)), 5))
        order = A.order_oracle(st, i, c, base, 5)
        vals = {j: ref_value(tuple(sorted(A.apply_child(base, c, j))), 5) if any(A.apply_child(base, c, j)) else 0 for j in order}
        assert vals[order[0]] == min(vals.values())                                            # das Orakel probiert das beste Kind zuerst


def test_the_features_equal_an_independent_definition_on_many_states():
    n = 0
    for sd in range(30):
        st = pop_all(make_bay(4, 3, sd))
        if not any(st):
            continue
        i, c, base = A.split_state(st)
        for j in A.child_targets(base, i, 5, c):
            assert A.features(base, i, c, j, 5) == pytest.approx(ref_features(st, i, c, j, 5)), (sd, j)
            n += 1
    assert n > 60


def test_the_top_one_hit_count_equals_the_count_from_the_independent_child_values():
    bays = [make_bay(3, 3, sd) for sd in range(25)]
    hit, total = first_child_optimal(bays, 5, A.order_fit)
    ref_hit = ref_total = 0
    for bay in bays:
        st = pop_all(bay)
        while any(st):
            i, c, base = A.split_state(st)
            vals = {j: (0 if not any(A.apply_child(base, c, j)) else ref_value(tuple(sorted(A.apply_child(base, c, j))), 5)) for j in A.child_targets(base, i, 5, c)}
            best = min(vals.values())
            if best >= INF:
                break
            ref_total += 1
            ref_hit += 1 if vals[A.order_fit(st, i, c, base, 5)[0]] == best else 0
            st = A.apply_child(base, c, next(j for j in A.child_targets(base, i, 5, c) if vals[j] == best))
            if ref_total > 4000:
                break
    assert (hit, total) == (ref_hit, ref_total) and total > 80


def test_the_best_order_cannot_save_in_the_proof_stages_beyond_the_table_effect():
    """Die Kernaussage in klein: In 4 x 4 (Bucht 23) steckt fast alles im Beweis, und das perfekte Orakel hat im Fund viel weniger Knoten, im Beweis aber nicht weniger."""
    bay = make_bay(4, 4, 23)
    fit, oracle = A.search(bay, 6), A.search(bay, 6, A.order_oracle)
    assert fit.proof / fit.nodes > 0.9 and oracle.find < fit.find / 5 and oracle.proof >= fit.proof * 0.9
