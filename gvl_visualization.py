"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import plotly.graph_objects as go

import gvl_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def blocker_cells(stacks) -> set:
    """Container, die über einem früher abzurufenden im selben Stapel liegen (Position (Stapel, Lage))."""
    out = set()
    for i, s in enumerate(stacks):
        lowest = 10 ** 9
        for j, c in enumerate(s):
            if c > lowest:
                out.add((i, j))
            else:
                lowest = c
    return out


def build_bay(stacks, height, target=None):
    """Die Bucht als Säulen: je Lage ein Balken mit der Abrufnummer. Rot: blockiert einen früher abzurufenden, grün: nächster Abruf
    (`target`, eine Abrufnummer oder None), grau-blau: übrige."""
    blockers = blocker_cells(stacks)
    fig = go.Figure()
    xs = list(range(len(stacks)))
    for j in range(height):
        ys, colors, texts = [], [], []
        for i, s in enumerate(stacks):
            if j < len(s):
                c = s[j]
                ys.append(1)
                texts.append(str(c))
                colors.append(C.COLOR_TARGET if c == target else C.COLOR_BLOCK if (i, j) in blockers else C.COLOR_CELL)
            else:
                ys.append(0)
                texts.append("")
                colors.append(C.COLOR_CELL)
        fig.add_trace(go.Bar(x=xs, y=ys, base=j, marker_color=colors, text=texts, textposition="inside", insidetextanchor="middle",
                             textfont=dict(color="#ffffff", size=16), width=0.78, showlegend=False,
                             hovertemplate="Abruf %{text}<extra></extra>", marker_line=dict(width=0)))
    fig.update_layout(barmode="overlay")
    fig.add_hline(y=height, line_dash="dot", line_color="rgba(128,128,128,0.7)", annotation_text="Höchsthöhe", annotation_position="top left")
    fig.update_xaxes(tickvals=xs, ticktext=[f"Stapel {i + 1}" for i in xs], showgrid=False)
    fig.update_yaxes(range=[0, height + 0.35], showticklabels=False, showgrid=False, zeroline=False)
    return _base(fig, height=48 * height + 90)


def _cell_label(k):
    s, l = k.split(",")
    return f"{s} × {l}"


def _cells_in_order(cells):
    return sorted(cells, key=lambda k: (int(k.split(",")[0]) * int(k.split(",")[1]), k))


def build_stages(results):
    """Knoten je Reihenfolge, gestapelt in Beweis (Stufen vor der letzten) und Fund (letzte Stufe); Läufe am Limit fehlen. `results`: {Name: SearchResult}."""
    names = [n for n in C.ORDER_NAMES if n in results and not results[n].capped]
    x = [C.ORDER_NAMES[n] for n in names]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[results[n].proof for n in names], name="Beweis (Stufen vor der letzten)", marker_color=C.COLOR_CELL, hovertemplate="%{x}: %{y:,.0f}<extra>%{fullData.name}</extra>"))
    fig.add_trace(go.Bar(x=x, y=[results[n].find for n in names], name="Fund (letzte Stufe)", marker_color=C.COLOR_TARGET, hovertemplate="%{x}: %{y:,.0f}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Knoten", rangemode="tozero")
    return _base(fig, height=320)


def build_proof(cells):
    """Knoten der Engsten Passung je Zelle (Summe über die Buchten), gestapelt in Beweis und Fund. `cells`: {"s,l": order_stats}."""
    keys = _cells_in_order(cells)
    x = [_cell_label(k) for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[cells[k]["orders"]["fit"]["proof_sum"] for k in keys], name="Beweis (Stufen vor der letzten)", marker_color=C.COLOR_CELL,
                         hovertemplate="%{x}: %{y:,.0f}<extra>%{fullData.name}</extra>"))
    fig.add_trace(go.Bar(x=x, y=[cells[k]["orders"]["fit"]["find_sum"] for k in keys], name="Fund (letzte Stufe)", marker_color=C.COLOR_TARGET,
                         hovertemplate="%{x}: %{y:,.0f}<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category", title_text="Zelle (Stapel × Lagen)")
    fig.update_yaxes(title_text="Knoten der Engsten Passung (Summe)", rangemode="tozero")
    return _base(fig, height=340)


def build_relative(cells, names=("learned", "oracle", "random")):
    """Knoten je Reihenfolge relativ zur Engsten Passung (Engste Passung = 100 %), je Zelle. `cells`: {"s,l": order_stats mit den genannten Reihenfolgen}."""
    keys = _cells_in_order(cells)
    x = [_cell_label(k) for k in keys]
    fig = go.Figure()
    for name in names:
        fig.add_trace(go.Bar(x=x, y=[100.0 * cells[k]["orders"][name]["nodes_sum"] / cells[k]["orders"]["fit"]["nodes_sum"] for k in keys], name=C.ORDER_NAMES[name],
                             marker_color=C.COLOR_ORDER[name], hovertemplate="%{x}: %{y:.0f} %<extra>%{fullData.name}</extra>"))
    fig.update_layout(barmode="group")
    fig.update_xaxes(type="category", title_text="Zelle (Stapel × Lagen)")
    fig.update_yaxes(title_text="Knoten gegenüber der Engsten Passung (100 %)", rangemode="tozero")
    return _base(fig, height=340)
