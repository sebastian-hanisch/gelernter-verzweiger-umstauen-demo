"""Gelernter Verzweiger: lernt die Suche, welches Kind sie zuerst probieren soll? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 14 der Reihe „Umstauen: Untergrenze und Suche“ (gelb). Die Suche aus Stück 5 und 9 probiert die Kinder in der Reihenfolge der „engsten Passung“. Hier die Frage, ob ein gelernter
Verzweiger besser ordnet, und was selbst ein perfektes Orakel höchstens sparen kann: Es spart praktisch nur im Fund (letzte Budgetstufe), nicht im Beweis (alle Stufen davor).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gvl_algorithm as A
import gvl_constants as C
from gvl_evaluation import (
    ALL_NAMES,
    Settings,
    analyse,
    key,
    load_bands,
    num,
    num_f,
    percent_text,
    result_rows,
    signed_percent_text,
    verdict_bay,
    verdict_learning,
    verdict_proof,
)
from gvl_layout import dunkles_design, kopf_css
from gvl_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from gvl_visualization import build_bay, build_proof, build_relative, build_stages

st.set_page_config(page_title="Gelernter Verzweiger – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _bands():
    return load_bands()


@st.cache_data(show_spinner=False)
def _analysis(settings, weights):
    return analyse(settings, list(weights))


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gvl_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧭 Gelernter Verzweiger: lernt die Suche, wohin zuerst?")
st.markdown(
    """
Die Suche aus Stück 5 und 9 probiert die **Kinder** (wohin kommt der Blocker?) in der Reihenfolge der **„engsten Passung“**. Kann ein **gelernter Verzweiger** besser ordnen? Und was spart **selbst ein perfektes Orakel** höchstens? Die Demo **rechnet das live** an einer Bucht und misst vorberechnet: Die Suche besteht aus dem **Beweis** (alle Budgetstufen vor der letzten: der Baum bis zum Budget wird bis auf den Einfluss der Zustandsmerken, hier bis etwa 16 % in einer Zelle und 20 % in einer Bucht, unabhängig von der Reihenfolge abgesucht) und dem **Fund** (die letzte Stufe, bis zum ersten Blatt). Eine gute Reihenfolge kann im Wesentlichen nur den **Fund** verkürzen.

Weiter unten: die Abschnitte **🔬 Experiment** (das Modell, wie viel im Beweis steckt, Orakel und gelernter Verzweiger), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Dies ist das vierzehnte Stück der Reihe „Umstauen: Untergrenze und Suche“ (gelb).
"""
)

with st.expander("So wird gemessen", expanded=True):
    st.markdown(
        """
1. **Suche:** IDA\\* des restricted Relocation mit der Untergrenze LB2 (Stück 9) und Zustandsmerken; **Knoten** = Aufrufe der Suche, je Budgetstufe gezählt.
2. **Beweis und Fund:** Die letzte Stufe findet die Lösung (bis zum ersten Blatt). Alle Stufen davor beweisen, dass es mit kleinerem Budget nicht geht; dort wird der ganze Baum abgesucht.
3. **Reihenfolgen:** *Engste Passung* (Stück 1), *gelernt* (logistische Regression auf sieben Merkmalen eines Kindes), *Orakel* (der wahre Wert jedes Kindes, per eigener Suche je Kind), *zufällig* und *schlechteste zuerst* als Gegenproben.
4. **Training:** auf Buchten ab Seed 1000 (andere als die Testbuchten): Je Zustand auf einem optimalen Weg das Etikett „Kind liegt auf einem optimalen Weg“.
5. Alle Reihenfolgen finden in jeder Bucht **dasselbe Optimum** (geprüft); es zählen nur Knoten, keine Zeiten.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Bucht**")
    stacks = st.slider("Stapel", *bounds("stacks_slider"), key="stacks_slider", help="Wie viele Stapel die Bucht hat.")
    layers = st.slider("Lagen je Stapel", *bounds("layers_slider"), key="layers_slider", help="So viele Container stehen anfangs auf jedem Stapel.")
    extra = st.slider("Platz (Lagen über der Anfangshöhe)", *bounds("extra_slider"), key="extra_slider",
                      help="Höchsthöhe = Lagen + Platz. Standard 2. Die vorberechneten Tabellen gelten für Platz 2.")
    seed = st.number_input("Bucht-Nummer (Seed)", *bounds("seed_input"), key="seed_input", step=1, help="Die Nummer bestimmt, wie die Abrufnummern auf die Plätze verteilt sind.")
    st.button("🎲 Neue Bucht würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"stacks_slider": int(stacks), "layers_slider": int(layers), "extra_slider": int(extra), "seed_input": int(seed)})

bands = _bands()
model = bands["modell"]
weights = tuple(model["weights"])
settings = Settings(int(stacks), int(layers), int(extra), int(seed))
with st.spinner("Suche mit fünf Reihenfolgen (das Orakel braucht je Kind eine eigene Suche) …"):
    a = _analysis(settings, weights)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
fit, learned, oracle = a.results["fit"], a.results["learned"], a.results.get("oracle")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Optimum", "nicht bewiesen" if a.optimum is None else a.optimum, delta=f"Untergrenze {a.lb}, {a.containers} Container", delta_color="off")
m2.metric("Engste Passung", "am Limit" if fit.capped else num(fit.nodes), delta=None if fit.capped else f"Beweis {num(fit.proof)}, Fund {num(fit.find)}", delta_color="off", help="Knoten der Suche in der Reihenfolge der engsten Passung.")
m3.metric("Gelernt", "am Limit" if learned.capped else num(learned.nodes), delta=None if learned.capped or fit.capped else signed_percent_text(learned.nodes / fit.nodes - 1), delta_color="off",
          help="Knoten mit der gelernten Reihenfolge; darunter die Änderung gegenüber der Engsten Passung.")
if oracle is None:
    m4.metric("Orakel", "nicht gerechnet", delta=f"über {C.ORACLE_MAX_CONTAINERS} Container", delta_color="off")
else:
    m4.metric("Orakel", "am Limit" if oracle.capped else num(oracle.nodes), delta=None if oracle.capped or fit.capped else signed_percent_text(oracle.nodes / fit.nodes - 1), delta_color="off",
              help="Knoten mit der Reihenfolge nach dem wahren Wert jedes Kindes: ein Richtwert für die Obergrenze.")
m5.metric("Beweisanteil", "–" if fit.capped else percent_text(fit.proof / fit.nodes), delta="Engste Passung", delta_color="off", help="Anteil der Knoten vor der letzten Stufe.")
st.plotly_chart(build_bay(a.bay, a.height, target=min(c for s in a.bay for c in s)), width="stretch", key="chart_bay")
st.caption("Rot: blockierende Container (liegen über einem früher abzurufenden), grün: der nächste Abruf, grau-blau: alle übrigen.")
table = ["| Reihenfolge | Knoten | Beweis | Fund | Knoten je Stufe |", "|---|---|---|---|---|"]
for r in result_rows(a):
    table.append(f"| {r['name']} | {r['nodes']} | {r['proof']} | {r['find']} | {r['stages']} |")
st.markdown("\n".join(table))
st.plotly_chart(build_stages(a.results), width="stretch", key="chart_stages")
st.caption("Knoten = Aufrufe der Suche. Die Suche mit dem Orakel zählt nicht die Knoten der Wertberechnung je Kind.")

st.markdown("## ❓ Kernfrage: Was spart ein gelernter Verzweiger?")
kind, text = verdict_bay(a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment: das Modell ----------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Das Modell")
st.caption(f"Logistische Regression (Newton-Verfahren, Startgewichte 0), trainiert auf {num(model['n_children'])} Kindern aus Buchten ab Seed {model['seed0']} "
           f"({', '.join(f'{s} × {l}' for s, l, _ in model['cells'])}); Etikett „Kind liegt auf einem optimalen Weg“ (Anteil {percent_text(model['share_optimal'])}); "
           f"Trefferquote beim Training {percent_text(model['accuracy'])}.")
tm = ["| Merkmal | Gewicht |", "|---|---|"]
for name, w in zip(A.FEATURE_NAMES, model["weights"]):
    tm.append(f"| {name} | {num_f(round(w, 2)) if float(w).is_integer() else str(round(w, 2)).replace('.', ',')} |")
st.markdown("\n".join(tm))
top1 = bands["top1"]
tt = ["| Zelle | Zustände auf optimalen Wegen | Engste Passung trifft | gelernt trifft | zufällig trifft |", "|---|---|---|---|---|"]
for s_, l_, _ in C.TOP1_CELLS:
    t = top1[key(s_, l_)]
    tt.append(f"| {s_} × {l_} | {t['states']} | {percent_text(t['fit'] / t['states'])} | {percent_text(t['learned'] / t['states'])} | {percent_text(t['random'] / t['states'])} |")
st.markdown("\n".join(tt))
st.caption("„Trifft“: Das erste Kind der Reihenfolge liegt auf einem optimalen Weg (Testbuchten ab Seed 0, nicht die Trainingsbuchten).")

st.markdown("---")

# --- Experiment: wie viel steckt im Beweis -------------------------------------------------------------------------------------------------------

beweis, orakel = bands["beweis"], bands["orakel"]
st.subheader("🔬 Experiment: Wie viel steckt im Beweis?")
st.caption("Je Zelle feste Buchten (Platz 2), vorberechnet (`generate_precomputed.py`); Summe der Knoten über die Buchten. Beweisanteil = Knoten vor der letzten Stufe / alle (Engste Passung).")
ta = ["| Zelle (Buchten) | Engste Passung | gelernt | zufällig | schlechteste zuerst | Beweisanteil (Engste Passung) |", "|---|---|---|---|---|---|"]
for s_, l_, n_ in C.PROOF_CELLS:
    c = beweis[key(s_, l_)]
    ta.append(f"| {s_} × {l_} ({c['n_ok']}) | " + " | ".join(num(c["orders"][k]["nodes_sum"]) for k in ("fit", "learned", "random", "worst")) + f" | {percent_text(c['proof_share'])} |")
st.markdown("\n".join(ta))
st.plotly_chart(build_proof(beweis), width="stretch", key="chart_proof")
here = beweis.get(key(settings.stacks, settings.layers))
if here is None:
    st.info(f"Für {settings.stacks} × {settings.layers} wurde der Beweisanteil nicht vorberechnet; die Tabelle zeigt die Zellen, die gerechnet wurden.")
else:
    st.markdown(f"**Zelle {settings.stacks} × {settings.layers} (Ihre Einstellung):**")
    kind, text = verdict_proof(here, f"{settings.stacks} × {settings.layers}")
    getattr(st, kind)(text)

st.markdown("---")

# --- Experiment: Orakel und gelernter Verzweiger ------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment: Orakel und gelernter Verzweiger")
st.caption("Summe der Knoten über alle Stufen / über die letzte Stufe (den Fund). Das Orakel kennt den wahren Wert jedes Kindes: sein Abstand zur Engsten Passung ist ein Richtwert für die Obergrenze, keine strenge Schranke.")
tb = ["| Zelle (Buchten) | Engste Passung | gelernt | Orakel | zufällig | gelernt gegenüber Engster Passung | Orakel gegenüber Engster Passung |", "|---|---|---|---|---|---|---|"]
for s_, l_, n_ in C.ORACLE_CELLS:
    c = orakel[key(s_, l_)]
    o = c["orders"]
    cells = " | ".join(f"{num(o[k]['nodes_sum'])} / {num(o[k]['find_sum'])}" for k in ("fit", "learned", "oracle", "random"))
    tb.append(f"| {s_} × {l_} ({c['n_ok']}) | {cells} | {signed_percent_text(o['learned']['nodes_sum'] / o['fit']['nodes_sum'] - 1)} | {signed_percent_text(o['oracle']['nodes_sum'] / o['fit']['nodes_sum'] - 1)} |")
st.markdown("\n".join(tb))
st.plotly_chart(build_relative(orakel), width="stretch", key="chart_relative")
here_o = orakel.get(key(settings.stacks, settings.layers))
if here_o is None:
    st.info(f"Für {settings.stacks} × {settings.layers} wurde das Orakel nicht vorberechnet (Zellen: 4 × 3, 5 × 3, 4 × 4, 5 × 4); die Tabelle zeigt sie.")
else:
    st.markdown(f"**Zelle {settings.stacks} × {settings.layers} (Ihre Einstellung):**")
    kind, text = verdict_learning(here_o, f"{settings.stacks} × {settings.layers}")
    getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein einfaches Modell** | Eine logistische Regression auf sieben Merkmalen; ein stärkeres Modell (zum Beispiel ein neuronales Netz mit Merkmalen der ganzen Bucht) wurde nicht gebaut. Das Orakel gibt einen Richtwert, wie viel zu holen wäre (bis 43 % der Knoten in 5 × 4; keine strenge Schranke), nicht, ob es erreichbar ist. | – |
| **Nur Knoten, keine Zeiten** | Der gelernte Verzweiger kostet je Kind eine Auswertung, das Orakel je Kind eine Suche; die Knotenzahl der Suche ist nicht die ganze Rechenzeit. | – |
| **Restricted Relocation mit LB2** | Mit schwächerer Untergrenze (Stück 5: LB1) ändert sich der Beweisanteil; beim Pre-Marshalling (Stück 11) gilt die Messung nicht. | Stück 5, 11 |
| **Volle Buchten, Platz 2** | Training auf 3 × 3 bis 5 × 3 und 4 × 4; größere Buchten und anderer Platz sind nicht gemessen. | – |
| **Literatur nicht nachgebaut** | Gelernte Verzweiger für das Relocation (Reinforcement Learning, 2025) und für das Pre-Marshalling (Hottung, Tanaka und Tierney 2020) sind aus Literaturtreffern bekannt, nicht gelesen und nicht reproduziert. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Suche.** IDA\* mit zulässiger Untergrenze $h$ (LB2, Stück 9): Für $b = h(s_0), h(s_0) + 1, \dots$ eine Tiefensuche, die einen Knoten bei $g + h > b$ abschneidet. Die erste Stufe $b^\*$, in der ein Blatt (leere Bucht) erreicht wird, ist das Optimum: $b^\* = \text{OPT}$.

**Beweis und Fund.** Jede Stufe $b < \text{OPT}$ findet kein Blatt und durchsucht deshalb den ganzen Baum $\{ s : g(s) + h(s) \le b \}$ (bis auf die Zustandsmerken). Ihre Knotenzahl hängt **kaum** von der Reihenfolge der Kinder ab (bis auf Unterschiede der Zustandsmerken, die die Beweisgröße je nach Reihenfolge um bis zu etwa 16 % in einer Zelle und 20 % in einer Bucht ändern). Nur die letzte Stufe $b = \text{OPT}$ bricht beim ersten Blatt ab; hier entscheidet die Reihenfolge, wie früh das Blatt kommt.

**Richtwert für die Obergrenze.** Ein Orakel, das in jedem Zustand die Kinder nach ihrem wahren Wert $V(c)$ ordnet, findet in der letzten Stufe das Blatt auf geradem Weg. Im Fund kann kein Verzweiger weniger Knoten brauchen als das Orakel, und in den Stufen davor spart eine Reihenfolge bis auf die Zustandsmerken nichts: $\text{Knoten}_{\text{Verzweiger}} \ge \text{Beweis} + \text{Fund}_{\text{Orakel}}$, wobei der Beweis der des jeweiligen Verzweigers ist (bis auf die Zustandsmerken). Das Orakel ist deshalb ein Richtwert, keine strenge Schranke: In der Bucht 4 × 4, Nr. 23 braucht es insgesamt mehr Knoten als die Regel.

**Gelernter Verzweiger.** Je Zustand $s$ auf einem optimalen Weg und je Kind $c$ ein Merkmalsvektor $x_c \in \mathbb{R}^7$ und das Etikett $y_c = 1$, wenn $V(c) = \min_{c'} V(c')$. Logistische Regression: $P(y = 1 \mid x) = 1 / (1 + e^{-w^\top x})$, $w$ durch das Newton-Verfahren (L2-Strafe $10^{-2}$, Startwert 0). Die Reihenfolge der Kinder: absteigend nach $w^\top x_c$.

**Literatur.** Gelernte Verzweiger für das Relocation und das Pre-Marshalling: aus Literaturtreffern bekannt, nicht gelesen. Die Zahlen hier sind eigene Messungen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
