# Gelernter Verzweiger: lernt die Suche, wohin zuerst? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gelernter-verzweiger-umstauen-demo.streamlit.app/)**

Vierzehntes Stück (gelb) der Konzepte-Reihe **„Umstauen: Untergrenze und Suche“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Suche aus Stück 5 und 9 probiert die **Kinder** (wohin kommt der Blocker?) in der Reihenfolge der **„engsten Passung“** aus Stück 1. Kann ein **gelernter Verzweiger** besser ordnen? Und was spart **selbst ein perfektes Orakel** höchstens? Die Demo **rechnet das live** an einer Bucht und misst vorberechnet: Die Suche besteht aus dem **Beweis** (alle Budgetstufen vor der letzten: der ganze Baum bis zum Budget wird abgesucht, egal in welcher Reihenfolge) und dem **Fund** (die letzte Stufe, bis zum ersten Blatt). Eine gute Reihenfolge verkürzt nur den **Fund**.

## Warum dieses Problem

Gelernte Verzweiger sind ein aktueller Strang bei Relocation und Pre-Marshalling (Literaturtreffer 2025; Hottung, Tanaka, Tierney 2020; nicht gelesen). Hier die Frage, was sie **gegen die Untergrenze** bringen: Die Messung stellt den gelernten Verzweiger zwischen zwei Grenzen. Die **Regel „engste Passung“** ist schon fast so gut wie das **Orakel**, das den Wert jedes Kindes kennt: Sie probiert in 92 bis 99 % der Zustände auf optimalen Wegen ein optimales Kind zuerst. Ein **gelernter** Verzweiger (logistische Regression auf sieben Merkmalen, 79 % Trefferquote beim Training) ist **nicht besser** als die Regel (3 bis 6 % mehr Knoten). Und selbst das **perfekte** Orakel spart nur in der **letzten** Stufe: Die Stufen davor (der **Beweis**, dass kein Weg mit kleinerem Budget existiert) durchsucht die Suche vollständig. **Lernen verkürzt den Fund, nicht den Beweis.**

## Modell

- **Suche:** IDA\* des restricted Relocation mit der Untergrenze LB2 (Stück 9) und Zustandsmerken; **Knoten** = Aufrufe der Suche, je Budgetstufe gezählt. Die Stufen vor der letzten sind der **Beweis**, die letzte der **Fund**.
- **Reihenfolgen der Kinder:** *Engste Passung* (Stück 1); *gelernt* (Kinder absteigend nach Merkmale · Gewichte); *Orakel* (nach dem wahren Optimum des Kindzustands, je Kind eine eigene Suche); *zufällig* und *schlechteste zuerst* als Gegenproben.
- **Merkmale eines Kindes** („Blocker *c* auf Stapel *j*“): Achsenabschnitt; sicherer Stapel; Abstand des frühesten Abrufs; Füllstand; leerer Stapel; neue Blocker; LB2 des Kindzustands (geteilt durch die Containerzahl).
- **Training:** logistische Regression (reines numpy, Newton-Verfahren, Startgewichte 0, L2-Strafe 10<sup>−2</sup>, deterministisch); Etikett „das Kind liegt auf einem optimalen Weg“; Trainingsdaten aus 300 Buchten ab Seed 1000 (3 × 3, 4 × 3, 5 × 3 und 4 × 4, 5 420 Kinder entlang optimaler Wege); Test-Buchten ab Seed 0.
- **Alle Reihenfolgen finden in jeder Bucht dasselbe Optimum** (geprüft, bei Abweichung bricht die Auswertung ab).

## Methodik

- **Live:** an der gewählten Bucht die Suche mit fünf Reihenfolgen (Knotenlimit 300 000 je Reihenfolge); das Orakel bis 20 Container.
- **Vorberechnet** (`gvl_bands.json`, erzeugt von `generate_precomputed.py`, einige Minuten in mehreren Prozessen): das Modell, der Beweisanteil in fünf Zellen, das Orakel in vier Zellen, die Trefferquote des ersten Kindes in drei Zellen.
- Es gibt **nur Knotenzahlen, keine Zeiten**.
- **Abschnitte der App:** Ergebnis (live); das Modell; wie viel steckt im Beweis; Orakel und gelernter Verzweiger; Grenzen; mathematische Formulierung.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen sind Messungen an festen Buchten (Platz 2), nachgerechnet in `tests/test_claims.py`, `tests/test_presets.py` und `tests/test_oracle_gelernter_verzweiger_umstauen.py`.

**A: Wie viel steckt im Beweis?** (Summe der Knoten über die Buchten; Beweisanteil = Knoten vor der letzten Stufe / alle, Engste Passung)

| Zelle (Buchten) | Engste Passung | gelernt | zufällig | schlechteste zuerst | Beweisanteil |
|---|---|---|---|---|---|
| 4 × 3 (100) | 1 667 | 1 727 | 2 171 | 3 055 | 46 % |
| 5 × 3 (100) | 2 955 | 3 109 | 5 767 | 7 422 | 69 % |
| 4 × 4 (100) | 21 544 | 22 595 | 26 700 | 37 309 | 72 % |
| 5 × 4 (60) | 77 285 | 84 578 | 86 676 | 119 658 | 63 % |
| 6 × 3 (60) | 1 811 | 1 981 | 3 811 | 6 099 | 70 % |

Die Reihenfolge der Kinder **zählt** (schlechteste zuerst: 1,5- bis 3,4-fach so viele Knoten wie die Engste Passung; zufällig 1,3- bis 2,1-fach), aber nur, bis die Regel gut ist. Der **Beweis** macht 46 bis 72 % der Knoten aus.

**B: Orakel und gelernter Verzweiger** (Summe der Knoten über alle Stufen / über die letzte Stufe, der Fund)

| Zelle (Buchten) | Engste Passung | gelernt | Orakel | zufällig | gelernt gegenüber Engster Passung | Orakel gegenüber Engster Passung |
|---|---|---|---|---|---|---|
| 4 × 3 (100) | 1 667 / 908 | 1 727 / 968 | 1 440 / 681 | 2 171 / 1 403 | +3,6 % | −14 % |
| 5 × 3 (100) | 2 955 / 921 | 3 109 / 1 075 | 2 836 / 794 | 5 767 / 3 705 | +5,2 % | −4,0 % |
| 4 × 4 (60) | 14 420 / 4 485 | 14 897 / 4 493 | 10 913 / 668 | 16 268 / 5 428 | +3,3 % | −24 % |
| 5 × 4 (40) | 61 598 / 28 069 | 65 019 / 29 797 | 35 152 / 539 | 62 703 / 25 290 | +5,6 % | −43 % |

Das Orakel kennt den optimalen Wert jedes Kindes (mit einer eigenen Suche je Kind, also viel teurer als das Sparen): Seine Ersparnis gegenüber der Engsten Passung ist die **Obergrenze jedes Verzweigers** (14, 4, 24 und 43 %). Sie liegt **fast nur im Fund** (5 × 4: 28 069 auf 539 Knoten), die **Beweisknoten** ändern sich dagegen kaum (5 × 4: 33 529 bei der Engsten Passung, 35 222 gelernt, 34 613 beim Orakel, 37 413 zufällig; höchstens 12 % Unterschied, durch die Zustandsmerken). Der gelernte Verzweiger liegt in jeder Zelle **über** der Regel.

**C: Das Modell** (Gewichte der logistischen Regression)

| Merkmal | Gewicht |
|---|---|
| Achsenabschnitt | 1,06 |
| sicherer Stapel | 2,44 |
| Abstand des frühesten Abrufs | 0,44 |
| Füllstand | 0,36 |
| leerer Stapel | −1,35 |
| neue Blocker | −1,39 |
| LB2 des Kindes | −0,61 |

Trainingsgenauigkeit 79 % (64 % der Kinder liegen auf einem optimalen Weg, Raten „immer ja“ trifft also 64 %). **Trefferquote des ersten Kindes** entlang optimaler Wege (Test-Buchten): Engste Passung 99 % (4 × 3: 169 von 170), 92 % (4 × 4: 195 von 213), 99 % (5 × 3: 211 von 213); gelernt 98, 89 und 97 %; zufällig 66, 58 und 61 %.

| Frage | Befund |
|---|---|
| **Ordnet ein gelernter Verzweiger besser als die Regel?** | Nein: Er ist in allen vier Zellen etwas schlechter (+3 bis +6 % Knoten) und trifft das optimale Kind seltener als die Regel. |
| **Was spart ein perfektes Orakel?** | 4 bis 43 % der Knoten (Obergrenze jedes Verzweigers), fast nur in der letzten Stufe. |
| **Wo steckt der Rest?** | Im Beweis: 46 bis 72 % der Knoten; seine Größe hängt kaum von der Reihenfolge ab (höchstens 20 % Unterschied zwischen den Reihenfolgen in der Bucht 4 × 4, Nr. 23). |
| **Kann das Orakel verlieren?** | Ja, an einer Bucht: 4 × 4, Nr. 23 spart das Orakel im Fund (135 auf 14 Knoten), braucht aber im Beweis mehr (2 679 gegen 2 473): insgesamt 2 693 gegen 2 608. |
| **Ist die Regel immer die beste Reihenfolge?** | Nein: In der Bucht 4 × 4, Nr. 25 braucht die Regel 200 Knoten, zufällig 40, schlechteste zuerst 28, gelernt 145, das Orakel 12. |

**Presets** (Eigenschaften in `tests/test_presets.py`): „Alltag“ (4 × 3, Bucht 0: 58 Knoten, 47 im Beweis), „Nur Fund, kein Beweis“ (4 × 3, Bucht 1: eine Stufe, 8 Knoten), „Der Beweis dominiert“ (4 × 4, Bucht 23), „Gelernt besser als die Regel“ (4 × 4, Bucht 25: 145 gegen 200), „Gelernt schlechter als die Regel“ (4 × 3, Bucht 10: 33 gegen 17).

**Korrektur gegenüber dem Plan:** Die Zahlen stimmen mit der Vormessung überein (deterministisch). Zwei Zahlen waren im Plan ungenau und sind hier korrigiert: Gewicht „sicherer Stapel“ 2,44 (Plan: 2,45, Rundung) und die Spanne der Beweisknoten bei 5 × 4 (höchstens 12 % zwischen Engster Passung, gelernt, Orakel und zufällig, 16 % mit „schlechteste zuerst“).

## Ehrliche Grenzen

- **Ein einfaches Modell.** Logistische Regression auf sieben Merkmalen; ein stärkeres Modell (zum Beispiel ein neuronales Netz mit Merkmalen der ganzen Bucht) wurde nicht gebaut. Das Orakel zeigt, wie viel höchstens zu holen wäre (bis 43 % in 5 × 4), nicht, ob es erreichbar ist.
- **Nur Knoten, keine Zeiten:** Der gelernte Verzweiger kostet je Kind eine Auswertung, das Orakel je Kind eine Suche.
- **Restricted Relocation mit LB2.** Mit schwächerer Untergrenze (Stück 5: LB1) ändert sich der Beweisanteil; für das Pre-Marshalling (Stück 11) gilt die Messung nicht.
- **Volle Buchten, Platz 2;** Training auf 3 × 3 bis 5 × 3 und 4 × 4; größere Buchten und anderer Platz sind nicht gemessen.
- **Literatur nicht nachgebaut:** gelernte Verzweiger für das Relocation (Reinforcement Learning 2025) und das Pre-Marshalling (Hottung, Tanaka, Tierney 2020) sind aus Literaturtreffern bekannt, nicht gelesen und nicht reproduziert.

## Verifikation

- **Kern:** Suche von Hand (eine Hand-Bucht in einer Stufe, eine Bucht in zwei Stufen mit 47 und 11 Knoten), Merkmale eines Kindes von Hand (Hand-Bucht, sicherer Stapel, leerer Stapel), jede Reihenfolge von Hand, Zufallsreihenfolge deterministisch, Stufeneigenschaften (Zahl der Stufen = Optimum − Untergrenze + 1), Knotenlimit, Kindwerte.
- **Orakel:** Optimum für **jede Reihenfolge** gegen eine **eigene vollständige Rekursion** (ohne Untergrenze, ohne IDA\*) in über 300 Buchten; die Kindwerte des Orakels gegen die Rekursion; die **Merkmale** gegen eine zweite Umsetzung aus der Definition; die **Trefferquote** des ersten Kindes gegen eine unabhängige Zählung aus den Kindwerten.
- **Modell:** die logistische Regression findet auf synthetischen Daten das bekannte Modell wieder, ist deterministisch, die L2-Strafe schrumpft die Gewichte, das Training auf einer kleinen Zelle ist wiederholbar.
- **Zahlen im README:** jede Zahl der Befunde in `tests/test_claims.py` (gespeicherte Bänder); zwei Beweis-Zellen, eine Orakel-Zelle und die Trefferquote werden frisch nachgerechnet (Knotenzahlen, keine Zeiten).
- **Verdikte:** reine Funktionen, jede Schwelle (5 % Ersparnis, Beweisanteil 50 %, genau-an-der-Grenze-Fälle) und jede genannte Zahl an von Hand gesetzten Werten (`tests/test_verdicts.py`).
- **Presets und Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum), Inhalt (Kennzahlen je Preset, Tabellenzeilen), Layout, Footer, Laufzeit-Importe (`numpy`), Abbildungen.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Aufbau nach DEMO-PLAYBOOK §2) |
| `generate_precomputed.py` | trainiert das Modell und erzeugt `gvl_bands.json` |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, plotly, numpy |
| `gvl_algorithm.py` | Relocation, IDA\* mit Stufenzählung und austauschbarer Kinderreihenfolge, Merkmale, Orakel |
| `gvl_bands.json` | das Modell und die vorberechneten Kennzahlen |
| `gvl_constants.py` | Regler-Grenzen, Limits, Zellen, Schwellen, Farben, Presets |
| `gvl_evaluation.py` | Analyse einer Bucht, Stichproben, Verdikte (reine Funktionen) |
| `gvl_layout.py` | Fester Kopf (Stylesheet) |
| `gvl_model.py` | Trainingsdaten, logistische Regression, Trefferquote |
| `gvl_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `gvl_scenario.py` | Buchten (volle Buchten) |
| `gvl_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `tests/` | Handrechnung, Orakel, Modell, Randfälle, Behauptungen, Verdikte, Presets, Abbildungen, AppTest, Layout, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Stärkere Lernverfahren** (neuronale Netze, verstärkendes Lernen) und gelernte Untergrenzen.
- **Zeitmessung:** nur Knoten.
- **PDF-Export:** nicht vorgesehen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

95 Tests, rund eine Minute.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
