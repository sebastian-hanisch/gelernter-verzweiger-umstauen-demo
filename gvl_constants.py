"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

STACKS_MIN, STACKS_MAX, DEFAULT_STACKS = 3, 5, 4          # Zahl der Stapel
LAYERS_MIN, LAYERS_MAX, DEFAULT_LAYERS = 2, 4, 3          # Anfangsfüllung je Stapel (Lagen)
EXTRA_MIN, EXTRA_MAX, DEFAULT_EXTRA = 1, 3, 2             # Platz: Höchsthöhe = Lagen + Zuschlag (Standard 2 wie in Stück 1 bis 13)
DEFAULT_SEED = 0
SEED_MAX = 999999

LIVE_CAP = 300_000                                        # Knotenlimit der Suche je Reihenfolge (live)
ORACLE_MAX_CONTAINERS = 20                                # das Orakel (eine Suche je Kind) wird live nur bis zu dieser Containerzahl gerechnet
BAND_CAP = 1_000_000                                      # Knotenlimit in der Vorberechnung

# Zellen der vorberechneten Tabellen: (Stapel, Lagen, Zahl der festen Buchten, Platz 2); Test-Buchten ab Seed 0, Trainings-Buchten ab TRAIN_SEED0.
PROOF_CELLS = ((4, 3, 100), (5, 3, 100), (4, 4, 100), (5, 4, 60), (6, 3, 60))      # A: Beweis und Reihenfolgen ohne Orakel
ORACLE_CELLS = ((4, 3, 100), (5, 3, 100), (4, 4, 60), (5, 4, 40))                  # B: Orakel und gelernter Verzweiger
TOP1_CELLS = ((4, 3, 30), (5, 3, 30), (4, 4, 20))                                  # Trefferquote des ersten Kindes entlang optimaler Wege
TRAIN_CELLS = ((3, 3, 100), (4, 3, 100), (5, 3, 60), (4, 4, 40))
TRAIN_SEED0 = 1000

# Schwellen der Verdikte
LEARN_GAIN = 0.05                                         # „gelernt besser“: Knoten höchstens so viel (relativ) unter der Engsten Passung
LEARN_LOSS = 0.0                                          # „gelernt schlechter“: Knoten über der Engsten Passung
PROOF_DOMINATES = 0.5                                     # „der Beweis dominiert“: Anteil der Knoten vor der letzten Stufe mindestens

COLOR_CELL = "#7a8fa6"
COLOR_BLOCK = "#b5545a"        # blockierend (liegt über einem früher abzurufenden)
COLOR_TARGET = "#2F6B65"       # nächster Abruf
COLOR_ORDER = {"fit": "#2F6B65", "learned": "#4c78a8", "oracle": "#1c4742", "random": "#d9822b", "worst": "#b5545a"}
ORDER_NAMES = {"fit": "Engste Passung", "learned": "gelernt", "oracle": "Orakel", "random": "zufällig", "worst": "schlechteste zuerst"}

# Presets (Eigenschaften in tests/test_presets.py)
PRESETS = {
    "Alltag (4 Stapel, 3 Lagen)": {"stacks": 4, "layers": 3, "extra": DEFAULT_EXTRA, "seed": 0},
    "Nur Fund, kein Beweis (4 Stapel, 3 Lagen)": {"stacks": 4, "layers": 3, "extra": DEFAULT_EXTRA, "seed": 1},
    "Der Beweis dominiert (4 Stapel, 4 Lagen)": {"stacks": 4, "layers": 4, "extra": DEFAULT_EXTRA, "seed": 23},
    "Gelernt besser als die Regel (4 Stapel, 4 Lagen)": {"stacks": 4, "layers": 4, "extra": DEFAULT_EXTRA, "seed": 25},
    "Gelernt schlechter als die Regel (4 Stapel, 3 Lagen)": {"stacks": 4, "layers": 3, "extra": DEFAULT_EXTRA, "seed": 10},
}
PRESET_HELP = {
    "Alltag (4 Stapel, 3 Lagen)": "12 Container: 58 Knoten, 47 davon im Beweis; Engste Passung, gelernt und Orakel machen dasselbe.",
    "Nur Fund, kein Beweis (4 Stapel, 3 Lagen)": "12 Container: Das Optimum ist die Untergrenze, es gibt nur eine Stufe (8 Knoten): kein Beweis, nur der Fund.",
    "Der Beweis dominiert (4 Stapel, 4 Lagen)": "16 Container: 2 473 von 2 608 Knoten stecken im Beweis. Das Orakel spart im Fund (135 auf 14 Knoten), verliert aber im Beweis: insgesamt mehr Knoten.",
    "Gelernt besser als die Regel (4 Stapel, 4 Lagen)": "16 Container, nur eine Stufe: Engste Passung 200 Knoten, gelernt 145, Orakel 12. Hier ist die Regel sogar schlechter als zufällig (40).",
    "Gelernt schlechter als die Regel (4 Stapel, 3 Lagen)": "12 Container: gelernt 33 Knoten gegen 17 der Engsten Passung; das Orakel braucht ebenfalls 17.",
}
