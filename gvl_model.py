"""Das gelernte Modell: Trainingsdaten entlang optimaler Wege, logistische Regression (reines numpy, Newton-Verfahren, deterministisch).

Je Zustand auf einem optimalen Weg gibt es mehrere Kinder (Blocker c auf Stapel j); das Etikett ist 1, wenn das Kind einen optimalen Weg hat (sein Optimum gleich dem kleinsten Kindoptimum),
sonst 0. Die Merkmale stehen in gvl_algorithm.features. Der Weg wird dem ersten Kind mit dem kleinsten Wert gefolgt."""

import numpy as np

import gvl_algorithm as A
from gvl_scenario import make_bay, max_height


def collect(bays, height, limit_states=40):
    """Trainingsdaten (Merkmale, Etiketten) aus Buchten: je Zustand entlang eines optimalen Weges alle Kinder."""
    X, y = [], []
    for bay in bays:
        st, _ = A.autopop(bay)
        for _ in range(limit_states):
            if not any(st):
                break
            i, c, base = A.split_state(st)
            rows = []
            for j in A.child_targets(base, i, height, c):
                rows.append((j, A.features(base, i, c, j, height), A.child_value(A.apply_child(base, c, j), height)))
            best = min(r[2] for r in rows)
            if best >= A.INF:
                break
            for j, f, v in rows:
                X.append(f)
                y.append(1.0 if v == best else 0.0)
            j_best = next(r[0] for r in rows if r[2] == best)
            st = A.apply_child(base, c, j_best)
    return np.array(X, dtype=float), np.array(y, dtype=float)


def fit_logistic(X, y, l2=1e-2, iters=50):
    """Logistische Regression mit L2-Strafe durch das Newton-Verfahren; Startgewichte 0, also deterministisch. Gibt die Gewichte zurück."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        weight = p * (1 - p) + 1e-6
        grad = X.T @ (p - y) + l2 * w
        hess = (X.T * weight) @ X + l2 * np.eye(X.shape[1])
        w = w - np.linalg.solve(hess, grad)
    return w


def accuracy(X, y, w) -> float:
    """Anteil der Kinder, deren Wahrscheinlichkeit auf der richtigen Seite von 0,5 liegt."""
    p = 1.0 / (1.0 + np.exp(-X @ w))
    return float(((p > 0.5) == (y == 1)).mean())


def train(cells, seed0=1000):
    """Trainiert auf den Zellen (Stapel, Lagen, Zahl der Buchten) mit den Buchten ab Seed seed0 und Platz 2; gibt (Gewichte, Zahl der Kinder, Anteil optimaler Kinder, Trefferquote)."""
    Xs, ys = [], []
    for s, layers, n in cells:
        X, y = collect([make_bay(s, layers, seed0 + sd) for sd in range(n)], max_height(layers, 2))
        Xs.append(X)
        ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    w = fit_logistic(X, y)
    return w, len(y), float(y.mean()), accuracy(X, y, w)


def first_child_optimal(bays, height, order, limit_states=40) -> tuple[int, int]:
    """Entlang eines optimalen Weges: in wie vielen Zuständen steht das erste Kind der Reihenfolge `order` auf einem optimalen Weg? Gibt (Treffer, Zustände) zurück."""
    hit = total = 0
    for bay in bays:
        st, _ = A.autopop(bay)
        for _ in range(limit_states):
            if not any(st):
                break
            i, c, base = A.split_state(st)
            vals = {j: A.child_value(A.apply_child(base, c, j), height) for j in A.child_targets(base, i, height, c)}
            best = min(vals.values())
            if best >= A.INF:
                break
            first = order(st, i, c, base, height)[0]
            total += 1
            hit += 1 if vals[first] == best else 0
            st = A.apply_child(base, c, next(j for j in A.child_targets(base, i, height, c) if vals[j] == best))
    return hit, total
