"""Erzeugt gvl_bands.json: das gelernte Modell und die Kennzahlen der festen Buchten (Platz 2).

    python generate_precomputed.py

"modell": Gewichte der logistischen Regression, Zahl der Trainingskinder, Anteil optimaler Kinder, Trefferquote (Training auf TRAIN_CELLS, Buchten ab TRAIN_SEED0).
"beweis": je Zelle aus PROOF_CELLS Knoten je Reihenfolge (Engste Passung, gelernt, zufällig, schlechteste zuerst) gesamt, im Beweis und im Fund.
"orakel": je Zelle aus ORACLE_CELLS dasselbe mit dem Orakel (Wert jedes Kindes per eigener Suche).
"top1":   je Zelle aus TOP1_CELLS die Zahl der Zustände entlang optimaler Wege und die Treffer des ersten Kindes (Engste Passung, gelernt, zufällig).

Rechenzeit einige Minuten (das Training und das Orakel brauchen je Kind eine eigene Suche; die Zellen laufen in getrennten Prozessen). Dieselben Funktionen wie die Tests
(`gvl_model.collect`, `fit_logistic`, `gvl_evaluation.order_rows`, `order_stats`, `top1_stats`); die Datei speichert nur Kennzahlen, keine Zeilen."""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import gvl_constants as C
import gvl_evaluation as ev
import gvl_model as M
from gvl_scenario import make_bay, max_height

sys.stdout.reconfigure(encoding="utf-8")


def collect_cell(stacks, layers, n):
    X, y = M.collect([make_bay(stacks, layers, C.TRAIN_SEED0 + sd) for sd in range(n)], max_height(layers, 2))
    return stacks, layers, X, y


def stats_job(kind, stacks, layers, n, weights):
    if kind == "beweis":
        return kind, ev.key(stacks, layers), ev.order_stats(ev.order_rows(stacks, layers, 2, range(n), ev.NAMES, weights), ev.NAMES)
    if kind == "orakel":
        return kind, ev.key(stacks, layers), ev.order_stats(ev.order_rows(stacks, layers, 2, range(n), ev.ALL_NAMES, weights), ev.ALL_NAMES)
    return kind, ev.key(stacks, layers), ev.top1_stats(stacks, layers, 2, range(n), weights)


def main():
    workers = max(2, (os.cpu_count() or 4) - 2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = {}
        for fut in as_completed([pool.submit(collect_cell, s, l, n) for s, l, n in C.TRAIN_CELLS]):
            s, l, X, y = fut.result()
            parts[(s, l)] = (X, y)
            print("Training", s, l, "fertig", flush=True)
        X = np.vstack([parts[(s, l)][0] for s, l, _ in C.TRAIN_CELLS])
        y = np.concatenate([parts[(s, l)][1] for s, l, _ in C.TRAIN_CELLS])
        w = M.fit_logistic(X, y)
        weights = [float(v) for v in w]
        out = {"modell": {"weights": weights, "n_children": int(len(y)), "share_optimal": float(y.mean()), "accuracy": M.accuracy(X, y, w), "cells": [list(c) for c in C.TRAIN_CELLS],
                          "seed0": C.TRAIN_SEED0}, "beweis": {}, "orakel": {}, "top1": {}}
        print("Gewichte", [round(v, 3) for v in weights], flush=True)
        tasks = [("orakel", s, l, n) for s, l, n in reversed(C.ORACLE_CELLS)] + [("top1", s, l, n) for s, l, n in C.TOP1_CELLS] + [("beweis", s, l, n) for s, l, n in reversed(C.PROOF_CELLS)]
        futures = {pool.submit(stats_job, kind, s, l, n, weights): (kind, s, l) for kind, s, l, n in tasks}
        for fut in as_completed(futures):
            kind, k, stats = fut.result()
            out[kind][k] = stats
            print(kind, k, "fertig", flush=True)
    with open("gvl_bands.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)
    print("geschrieben: gvl_bands.json", flush=True)


if __name__ == "__main__":
    main()
