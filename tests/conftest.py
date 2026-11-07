"""Gemeinsame Fixtures: die Buchten, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """Die Bucht von Hand: (1, 5), (2, 3), (4, 6), Höchsthöhe 4. Blocker 5, 3 und 6, die 5 hat keinen sicheren Stapel (Minima 2 und 4): LB2 = 4 = Optimum, eine einzige Stufe.
    Kinder des ersten Hubs (die 5 auf Stapel 2 oder 3): Merkmale von Hand in tests/test_algorithm.py."""
    return ((1, 5), (2, 3), (4, 6)), 4


@pytest.fixture
def counter():
    """Das Gegenbeispiel aus Stück 1: (3, 6), (2, 4), (1, 5), Höchsthöhe 4. LB2 = 4 = Optimum."""
    return ((3, 6), (2, 4), (1, 5)), 4
