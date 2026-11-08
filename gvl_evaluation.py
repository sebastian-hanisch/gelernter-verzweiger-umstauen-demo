"""Kennzahlen, Verdikt-Logik, vorberechnete Tabellen. Reine Funktionen: die App ruft st.<Art>(Text), die Tests prüfen jede Zahl und jede Schwelle."""

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import median

import gvl_algorithm as A
import gvl_constants as C
from gvl_scenario import make_bay, max_height

BANDS_FILE = Path(__file__).resolve().parent / "gvl_bands.json"
NAMES = ("fit", "learned", "random", "worst")             # Reihenfolgen ohne Orakel
ALL_NAMES = ("fit", "learned", "oracle", "random", "worst")


@dataclass(frozen=True)
class Settings:
    stacks: int = C.DEFAULT_STACKS
    layers: int = C.DEFAULT_LAYERS
    extra: int = C.DEFAULT_EXTRA
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    bay: tuple
    height: int
    containers: int
    lb: int                 # LB2 an der Wurzel
    results: dict           # {Name: SearchResult}; „oracle“ fehlt, wenn die Bucht zu groß ist

    @property
    def optimum(self):
        """Optimum aus einem Lauf, der nicht am Limit war; None, wenn alle am Limit waren."""
        for name in ALL_NAMES:
            r = self.results.get(name)
            if r is not None and r.optimum is not None:
                return r.optimum
        return None


def make_order(name: str, weights, seed: int = 0):
    if name == "fit":
        return A.order_fit
    if name == "learned":
        return A.order_learned_factory(weights)
    if name == "random":
        return A.order_random_factory(seed)
    if name == "worst":
        return A.order_worst
    if name == "oracle":
        return A.order_oracle
    raise ValueError(name)


def analyse(settings: Settings, weights) -> Analysis:
    height = max_height(settings.layers, settings.extra)
    bay = make_bay(settings.stacks, settings.layers, settings.seed)
    n = settings.stacks * settings.layers
    names = ALL_NAMES if n <= C.ORACLE_MAX_CONTAINERS else NAMES
    results = {name: A.search(bay, height, make_order(name, weights, settings.seed), C.LIVE_CAP) for name in names}
    opts = {r.optimum for r in results.values() if r.optimum is not None}
    if len(opts) > 1:
        raise AssertionError(f"Reihenfolgen finden verschiedene Optima: {opts}")
    return Analysis(bay, height, n, A.lb2(A.autopop(bay)[0]), results)


# ---------------------------------------------------------------- Formate

def num(n: int) -> str:
    """Ganze Zahl mit Leerzeichen als Tausendertrenner."""
    return f"{n:,}".replace(",", " ")


def _half_up(x: float, digits: int) -> str:
    """Kaufmännisch gerundet (…5 rundet auf) mit `digits` Nachkommastellen (vermeidet die Binärdarstellung: 1,635 als 1,63)."""
    d = Decimal(repr(round(float(x), 9))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return f"{d:.{digits}f}"


def num_f(x: float) -> str:
    """Zahl mit Leerzeichen als Tausendertrenner; hat sie einen Nachkommaanteil (Median einer geraden Zahl von Läufen), mit einer Nachkommastelle und Komma."""
    x = round(float(x), 6)
    if x.is_integer():
        return num(int(x))
    whole, frac = _half_up(x, 1).split(".")
    return f"{num(int(whole))},{frac}"


def percent_text(x: float) -> str:
    """Anteil als Prozent: unter 10 % mit einer Nachkommastelle, sonst ganzzahlig (Komma)."""
    p = x * 100
    return (_half_up(p, 1) if abs(p) < 10 else _half_up(p, 0)).replace(".", ",") + " %"


def signed_percent_text(x: float) -> str:
    """Relative Änderung mit Vorzeichen („+4 %“, „−14 %“, „±0 %“)."""
    t = percent_text(abs(x))
    if t in ("0,0 %", "0 %"):
        return "±0 %"
    return ("+" if x > 0 else "−") + t


def _le(x: float, limit: float) -> bool:
    """x <= limit mit Toleranz für Gleitkommarauschen."""
    return round(x, 9) <= limit


def _ge(x: float, limit: float) -> bool:
    return round(x, 9) >= limit


# ---------------------------------------------------------------- Verdikt einer Bucht

def _line(name: str, r: A.SearchResult) -> str:
    if r.capped:
        return f"{C.ORDER_NAMES[name]}: am Limit"
    return f"{C.ORDER_NAMES[name]}: {num(r.nodes)} Knoten (Beweis {num(r.proof)}, Fund {num(r.find)})"


def verdict_bay(a: Analysis) -> tuple[str, str]:
    """Verdikt der Kernfrage: was spart ein gelernter Verzweiger an dieser Bucht?"""
    fit, learned = a.results["fit"], a.results["learned"]
    if a.optimum is None:
        return "info", f"Das Optimum ist nicht bewiesen (Knotenlimit {num(C.LIVE_CAP)}); die Reihenfolgen finden es nicht innerhalb der Grenze."
    if fit.capped or learned.capped:
        return "info", f"Optimum {a.optimum}; {_line('fit', fit)}, {_line('learned', learned)}: kein Vergleich möglich."
    head = f"Optimum {a.optimum} (Untergrenze {a.lb}). {_line('fit', fit)}; {_line('learned', learned)}"
    oracle = a.results.get("oracle")
    if oracle is not None and not oracle.capped:
        head += f"; {_line('oracle', oracle)}"
    head += "."
    change = learned.nodes / fit.nodes - 1
    if len(fit.stages) == 1:
        return "info", head + " Das Optimum ist die Untergrenze: Es gibt nur eine Stufe und keinen Beweis, nur den Fund."
    if _le(change, -C.LEARN_GAIN):
        return "success", head + f" Der gelernte Verzweiger spart {percent_text(-change)} der Knoten gegenüber der Engsten Passung."
    if change > C.LEARN_LOSS:
        return "warning", head + f" Der gelernte Verzweiger braucht {percent_text(change)} mehr Knoten als die Engste Passung."
    return "info", head + f" Der gelernte Verzweiger ist hier nicht besser als die Engste Passung ({signed_percent_text(change)})."


def result_rows(a: Analysis) -> list[dict]:
    """Zeilen der Ergebnistabelle: je Reihenfolge Knoten, Beweis und Fund."""
    rows = []
    for name in ALL_NAMES:
        r = a.results.get(name)
        if r is None:
            rows.append({"name": C.ORDER_NAMES[name], "nodes": "nicht gerechnet", "proof": "–", "find": "–", "stages": "–"})
        elif r.capped:
            rows.append({"name": C.ORDER_NAMES[name], "nodes": "am Limit", "proof": "–", "find": "–", "stages": "–"})
        else:
            rows.append({"name": C.ORDER_NAMES[name], "nodes": num(r.nodes), "proof": num(r.proof), "find": num(r.find), "stages": " / ".join(num(x) for x in r.stages)})
    return rows


# ---------------------------------------------------------------- Stichproben (vorberechnet)

def key(stacks: int, layers: int) -> str:
    return f"{stacks},{layers}"


def order_rows(stacks: int, layers: int, extra: int, seeds, names, weights, cap: int = C.BAND_CAP) -> list[dict]:
    """Je Bucht und Reihenfolge: Knoten gesamt, im Beweis und im Fund, am Limit?; alle Läufe, die fertig werden, müssen dasselbe Optimum finden."""
    height = max_height(layers, extra)
    out = []
    for sd in seeds:
        bay = make_bay(stacks, layers, sd)
        row = {"seed": sd, "opt": None, "orders": {}}
        opts = set()
        for name in names:
            r = A.search(bay, height, make_order(name, weights, sd), cap)
            row["orders"][name] = {"nodes": r.nodes, "proof": r.proof, "find": r.find, "capped": r.capped}
            if r.optimum is not None:
                opts.add(r.optimum)
        if len(opts) > 1:
            raise AssertionError(f"Bucht {sd}: Reihenfolgen finden verschiedene Optima: {opts}")
        row["opt"] = next(iter(opts), None)
        out.append(row)
    return out


def order_stats(rows: list[dict], names) -> dict:
    """Je Zelle und Reihenfolge: Knoten gesamt (Summe, Median), im Fund (Summe, Median), im Beweis (Summe), Läufe am Limit; nur Buchten, in denen alle Reihenfolgen fertig wurden."""
    ok = [r for r in rows if not any(r["orders"][n]["capped"] for n in names)]
    out = {"n": len(rows), "n_ok": len(ok), "orders": {}}
    for name in names:
        x = [r["orders"][name] for r in ok]
        out["orders"][name] = {"nodes_sum": sum(v["nodes"] for v in x), "nodes_median": float(median(v["nodes"] for v in x)), "find_sum": sum(v["find"] for v in x),
                               "find_median": float(median(v["find"] for v in x)), "proof_sum": sum(v["proof"] for v in x)}
    out["proof_share"] = out["orders"]["fit"]["proof_sum"] / out["orders"]["fit"]["nodes_sum"]
    return out


def top1_stats(stacks: int, layers: int, extra: int, seeds, weights) -> dict:
    """Entlang optimaler Wege: Zahl der Zustände und die Treffer des ersten Kindes für Engste Passung, gelernt und zufällig (Test-Buchten, nicht die Trainingsbuchten)."""
    from gvl_model import first_child_optimal
    height = max_height(layers, extra)
    bays = [make_bay(stacks, layers, sd) for sd in seeds]
    out = {}
    total = 0
    for name in ("fit", "learned", "random"):
        hit, total = first_child_optimal(bays, height, make_order(name, weights, 0))
        out[name] = hit
    out["states"] = total
    return out


def load_bands() -> dict:
    """Vorberechnete Kennzahlen, erzeugt von generate_precomputed.py: {"modell": {...}, "beweis": {"s,l": order_stats}, "orakel": {"s,l": order_stats}, "top1": {"s,l": top1_stats}}."""
    return json.loads(BANDS_FILE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- Verdikte der Tabellen

def verdict_proof(stats: dict, name: str) -> tuple[str, str]:
    """Verdikt „wie viel steckt im Beweis?“ für eine Zelle (Engste Passung)."""
    o = stats["orders"]["fit"]
    share = stats["proof_share"]
    text = (f"{name}: Von {num(o['nodes_sum'])} Knoten der Engsten Passung stecken {num(o['proof_sum'])} im Beweis und {num(o['find_sum'])} im Fund: der Beweis ist {percent_text(share)}.")
    if _ge(share, C.PROOF_DOMINATES):
        return "warning", text + " Der Beweis dominiert: Eine bessere Reihenfolge kann kaum mehr als den Fund sparen."
    return "info", text + " Der Fund ist der größere Teil: Eine bessere Reihenfolge hat hier mehr Spielraum."


def verdict_learning(stats: dict, name: str) -> tuple[str, str]:
    """Verdikt „was bringt Lernen?“ für eine Zelle mit Orakel: Knotenänderung des gelernten Verzweigers und des Orakels gegenüber der Engsten Passung."""
    o = stats["orders"]
    fit = o["fit"]["nodes_sum"]
    learned = o["learned"]["nodes_sum"] / fit - 1
    oracle = o["oracle"]["nodes_sum"] / fit - 1
    text = (f"{name}: Gegenüber der Engsten Passung ({num(fit)} Knoten) braucht der gelernte Verzweiger {signed_percent_text(learned)} Knoten und das Orakel {signed_percent_text(oracle)}; "
            f"das Orakel ist ein Richtwert für die Obergrenze.")
    if _le(learned, -C.LEARN_GAIN):
        return "success", text + " Der gelernte Verzweiger spart hier spürbar Knoten."
    if learned > C.LEARN_LOSS:
        return "warning", text + " Der gelernte Verzweiger ist hier schlechter als die Regel."
    return "info", text + " Der gelernte Verzweiger ist hier nicht besser als die Regel."
