"""Kern (Stück 1, 5, 6, 9 und 10, verschlankt): restricted Relocation, IDA* mit LB2 und Zustandsmerken mit austauschbarer Kinderreihenfolge und Knotenzählung je Budgetstufe.

Eine Bucht besteht aus Stapeln (Tupel von unten nach oben). Die Abrufnummer 1 kommt zuerst dran; liegt die nächste Nummer nicht oben, wird der oberste Container ihres Stapels
(ein Blocker über dem Ziel) auf einen anderen Stapel mit Platz gesetzt, jeder Hub kostet 1; die nächste Nummer wird kostenlos entnommen, sobald sie oben liegt.

  lb2        Untergrenze: Blocker plus Blocker über dem Ziel ohne sicheren Stapel (Stück 9).
  search     IDA* mit lb2 und Zustandsmerken; `order` bestimmt die Reihenfolge der Kinder. Gibt das Optimum und die Zahl der Knoten JE BUDGETSTUFE zurück: Die Stufen vor der letzten
             sind der BEWEIS (der ganze Baum bis zum Budget wird abgesucht), die letzte Stufe der FUND (bis zum ersten Blatt).
  order_*    Reihenfolgen: Engste Passung (Stück 1), zufällig, schlechteste Passung zuerst, gelernt (Merkmale und Gewichte), Orakel (Wert jedes Kindes per eigener Suche).
  features   Merkmale des Kindes „Blocker c auf Stapel j“.
"""

import functools
import random
from dataclasses import dataclass

INF = 10 ** 9
FEATURE_NAMES = ("Achsenabschnitt", "sicherer Stapel", "Abstand des frühesten Abrufs", "Füllstand", "leerer Stapel", "neue Blocker", "LB2 des Kindes")


# ---------------------------------------------------------------- Bausteine

def autopop(stacks):
    """Ruft ab, was oben liegt und gerade dran ist. Gibt (neue Stapel, abgerufene Nummern in Reihenfolge) zurück."""
    st = [list(s) for s in stacks]
    popped = []
    while True:
        present = [c for s in st for c in s]
        if not present:
            return tuple(tuple(s) for s in st), popped
        t = min(present)
        for s in st:
            if s and s[-1] == t:
                s.pop()
                popped.append(t)
                break
        else:
            return tuple(tuple(s) for s in st), popped


def stack_blockers(stack):
    """Container in einem Stapel (unten nach oben), die über einem früher abzurufenden liegen."""
    total, lowest = 0, INF
    for c in stack:
        if c > lowest:
            total += 1
        else:
            lowest = c
    return total


def blocker_count(stacks):
    return sum(stack_blockers(s) for s in stacks)


def target_stack(stacks):
    """(nächste Abrufnummer, Index ihres Stapels). Die Bucht darf nicht leer sein."""
    t = min(c for s in stacks for c in s)
    for i, s in enumerate(stacks):
        if t in s:
            return t, i


def canon(stacks):
    return tuple(sorted(stacks))


def stack_min(stack):
    return min(stack) if stack else INF


def lb2(stacks):
    """Untergrenze (Stück 9, LB2): Blocker plus Blocker über dem Ziel, die größer sind als die früheste Nummer jedes anderen Stapels."""
    if not any(stacks):
        return 0
    t, i = target_stack(stacks)
    s = stacks[i]
    above = s[s.index(t) + 1:]
    others = [stack_min(x) for k, x in enumerate(stacks) if k != i]
    top = max(others) if others else -1
    return blocker_count(stacks) + sum(1 for c in above if c > top)


def _fit_key(stacks, j, c):
    m = stack_min(stacks[j])
    return (0, m) if m > c else (1, -m)


def child_targets(stacks, src, height, c):
    """Zielstapel für den Blocker c (Stapel `src`, Bucht ohne c) in der Reihenfolge der engsten Passung (Stück 1); nur Stapel mit Platz."""
    dests = [j for j in range(len(stacks)) if j != src and len(stacks[j]) < height]
    dests.sort(key=lambda j: (_fit_key(stacks, j, c), j))
    return dests


def split_state(st):
    """Zustand nach dem Abruf -> (Index des Stapels mit dem Ziel, oberster Container c, Bucht ohne c)."""
    _, i = target_stack(st)
    lst = [list(s) for s in st]
    c = lst[i].pop()
    return i, c, tuple(tuple(s) for s in lst)


def apply_child(base, c, j):
    """Kindzustand nach dem Hub von c auf Stapel j (und dem Abruf)."""
    nxt = [list(s) for s in base]
    nxt[j].append(c)
    return autopop(tuple(tuple(s) for s in nxt))[0]


# ---------------------------------------------------------------- Merkmale und Reihenfolgen

def features(base, i, c, j, height):
    """Merkmale des Kindes „Blocker c auf Stapel j“ (`base` = Bucht ohne c): Achsenabschnitt; sicher (früheste Nummer des Zielstapels größer als c); Abstand (früheste Nummer minus c,
    geteilt durch die Containerzahl; ein leerer Stapel zählt 1); Füllstand (Container / Höchsthöhe); leer; neue Blocker (Zuwachs der Blockerzahl des Zielstapels); LB2 des Kindzustands
    geteilt durch die Containerzahl."""
    n = sum(len(s) for s in base) + 1
    dest = base[j]
    m = stack_min(dest)
    safe = 1.0 if m > c else 0.0
    gap = (min(m, n + 1) - c) / n if m != INF else 1.0
    child = apply_child(base, c, j)
    return [1.0, safe, gap, len(dest) / height, 1.0 if not dest else 0.0, float(stack_blockers(dest + (c,)) - stack_blockers(dest)), lb2(child) / n]


def order_fit(st, i, c, base, height):
    """Engste Passung (Stück 1)."""
    return child_targets(base, i, height, c)


def order_worst(st, i, c, base, height):
    """Gegenprobe: die schlechteste Passung zuerst."""
    return list(reversed(child_targets(base, i, height, c)))


def order_random_factory(seed):
    """Zufällige Reihenfolge, deterministisch aus dem Seed (ein Zufallsgenerator je Suche)."""
    rng = random.Random(seed)

    def order(st, i, c, base, height):
        d = child_targets(base, i, height, c)
        rng.shuffle(d)
        return d
    return order


def order_learned_factory(weights):
    """Gelernte Reihenfolge: Kinder nach dem Wert Merkmale · Gewichte absteigend, bei Gleichstand nach der Engsten Passung."""
    def order(st, i, c, base, height):
        d = child_targets(base, i, height, c)
        score = {j: sum(w * f for w, f in zip(weights, features(base, i, c, j, height))) for j in d}
        return sorted(d, key=lambda j: (-score[j], d.index(j)))
    return order


@functools.lru_cache(maxsize=None)
def _opt_state(state, height, cap):
    r = search(state, height, order_fit, cap)
    return INF if r.optimum is None else r.optimum


def child_value(child, height, cap=300_000):
    """Optimum eines Kindzustands (nach dem Abruf), per Suche mit der Engsten Passung; INF am Limit."""
    return 0 if not any(child) else _opt_state(canon(child), height, cap)


def order_oracle(st, i, c, base, height):
    """Orakel: Kinder nach dem wahren Optimum ihres Zustands aufsteigend (jedes Kind per eigener Suche), bei Gleichstand nach der Engsten Passung."""
    d = child_targets(base, i, height, c)
    val = {j: child_value(apply_child(base, c, j), height) for j in d}
    return sorted(d, key=lambda j: (val[j], d.index(j)))


# ---------------------------------------------------------------- Suche

@dataclass(frozen=True)
class SearchResult:
    optimum: int | None     # None am Knotenlimit
    stages: tuple           # Knoten je Budgetstufe (bei Limit: die Stufen bis dahin, die letzte mit dem Rest)
    capped: bool

    @property
    def nodes(self) -> int:
        return sum(self.stages)

    @property
    def proof(self) -> int:
        """Knoten aller Stufen vor der letzten (der Beweis, dass kein kleineres Budget reicht)."""
        return sum(self.stages[:-1])

    @property
    def find(self) -> int:
        """Knoten der letzten Stufe (der Fund bis zum ersten Blatt)."""
        return self.stages[-1]


class _CapReached(Exception):
    pass


class _Found(Exception):
    pass


def search(bay, height, order=order_fit, cap=1_000_000) -> SearchResult:
    """IDA* mit lb2 und Zustandsmerken (kanonisch); `order(st, i, c, base, height)` liefert die Zielstapel der Kinder in der Probierreihenfolge."""
    start = tuple(tuple(s) for s in bay)
    budget = lb2(autopop(start)[0])
    stages = []
    total = 0
    while True:
        b = budget
        nodes = [0]
        table = {}
        found = []

        def rec(state, g):
            nodes[0] += 1
            if total + nodes[0] > cap:
                raise _CapReached
            st, _ = autopop(state)
            if not any(st):
                found.append(g)
                raise _Found
            if g + lb2(st) > b:
                return
            key = canon(st)
            if table.get(key, INF) <= g:
                return
            table[key] = g
            i, c, base = split_state(st)
            for j in order(st, i, c, base, height):
                nxt = [list(s) for s in base]
                nxt[j].append(c)
                rec(tuple(tuple(s) for s in nxt), g + 1)

        try:
            rec(start, 0)
        except _Found:
            pass
        except _CapReached:
            stages.append(cap - total)                              # die Summe aller Stufen ist genau das Limit
            return SearchResult(None, tuple(stages), True)
        stages.append(nodes[0])
        total += nodes[0]
        if found:
            return SearchResult(found[0], tuple(stages), False)
        budget += 1
