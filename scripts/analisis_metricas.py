#!/usr/bin/env python3
"""
Análisis de métricas de gem5 (Cortex A76) contra un baseline.

Uso desde terminal:
    python analisis_metricas.py --base 1_metricas.txt --dir resultados
    python analisis_metricas.py --base 1_metricas.txt --dir resultados --mcpat mcpat.csv

Convención de nombres (para sensibilidad y heatmaps):
    Cada archivo se llama  <param><valor>_<param><valor>_..._metricas.txt
    Ejemplo: rob256_iq32_metricas.txt  ->  rob=256, iq=32
    (Los tokens que no tengan forma letras+número se ignoran.)

Archivo McPAT opcional (CSV con columnas):  config,power_w
    donde 'config' es el nombre del archivo sin "_metricas.txt" (p. ej. rob256_iq32).

También se puede usar desde un notebook:
    from analisis_metricas import *
"""

import argparse
import glob
import itertools
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except ImportError:  # seaborn es opcional
    sns = None

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------

# Parámetros del baseline (opcional). Solo si quieres que el baseline aparezca
# en las gráficas de sensibilidad/heatmap. Usa los mismos nombres que en tus archivos.
# Ejemplo con los defaults de CortexA76.py:
# BASELINE_PARAMS = {"rob": 128, "iq": 16, "lq": 68, "sq": 72}
BASELINE_PARAMS = {}

# Para energía: E = P * delay. La guía del assignment dice "P * CPI", pero
# físicamente el retardo es el tiempo de ejecución. Cambia a "cpi" si tu profesor
# confirma que debe usarse CPI.
DELAY_METRIC = "simSeconds"  # "simSeconds" o "cpi"

# Nombre corto -> clave en el archivo de métricas
RAW = {
    "simSeconds": "simSeconds",
    "hostSeconds": "hostSeconds",
    "cpi": "system.cpu.cpi",
    "ipc": "system.cpu.ipc",
    "issueRate": "system.cpu.issueRate",
    "fuBusyRate": "system.cpu.fuBusyRate",
    "statFuBusy_MemWrite": "system.cpu.statFuBusy::MemWrite",
    "dcache_miss": "system.cpu.dcache.overallMissRate::total",
    "icache_miss": "system.cpu.icache.overallMissRate::total",
    "l2_miss": "system.cpu.l2cache.overallMissRate::total",
    "l3_miss": "system.cpu.l3cache.overallMissRate::total",
    "bp_committed": "system.cpu.branchPred.committed_0::total",
    "bp_mispredicted": "system.cpu.branchPred.mispredicted_0::total",
    "n_int": "system.cpu.commitStats0.numIntInsts",
    "n_vec": "system.cpu.commitStats0.numVecInsts",
    "n_load": "system.cpu.commitStats0.numLoadInsts",
    "n_store": "system.cpu.commitStats0.numStoreInsts",
    "n_memref": "system.cpu.commitStats0.numMemRefs",
    "n_control": "system.cpu.commitStats0.committedControl::IsControl",
    "t_IntAlu": "system.cpu.commitStats0.committedInstType::IntAlu",
    "t_SimdFloatAdd": "system.cpu.commitStats0.committedInstType::SimdFloatAdd",
    "t_SimdFloatMultAcc": "system.cpu.commitStats0.committedInstType::SimdFloatMultAcc",
    "t_SimdAlu": "system.cpu.commitStats0.committedInstType::SimdAlu",
}

# Para cada métrica: "low" = menor es mejor, "high" = mayor es mejor,
# "neutral" = solo informativa (no se marca como mejora/empeoramiento).
DIRECTION = {
    "simSeconds": "low",
    "cpi": "low",
    "ipc": "high",
    "issueRate": "high",
    "dcache_miss": "low",
    "icache_miss": "low",
    "l2_miss": "low",
    "l3_miss": "low",
    "branch_mispred_rate": "low",
    "hostSeconds": "neutral",
    "fuBusyRate": "neutral",
    "statFuBusy_MemWrite": "low",
}

# Métricas que se muestran en la tabla resumen (en este orden)
SUMMARY_METRICS = [
    "simSeconds", "cpi", "ipc", "issueRate", "speedup",
    "icache_miss", "dcache_miss", "l2_miss", "l3_miss",
    "branch_mispred_rate", "fuBusyRate", "statFuBusy_MemWrite",
    "pct_IntAlu", "pct_Simd", "pct_load", "pct_store", "pct_control",
    "hostSeconds",
]

OUT_DIR = "analisis_out"


# ----------------------------------------------------------------------------
# 1. PARSER / LOADER
# ----------------------------------------------------------------------------

def parse_metrics_file(path):
    """Lee un archivo 'clave: valor' y devuelve {clave: float | NaN}."""
    data = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, _, val = line.strip().rpartition(": ")
            if not key:  # línea sin espacio tras ':'
                key, _, val = line.strip().partition(":")
            try:
                data[key.strip()] = float(val)
            except ValueError:  # "None", vacío, etc.
                data[key.strip()] = np.nan
    return data


def config_name_from_path(path):
    return re.sub(r"_?metricas\.txt$", "", os.path.basename(path))


def params_from_name(name):
    """'rob256_iq32' -> {'rob': 256, 'iq': 32}"""
    params = {}
    for tok in name.split("_"):
        m = re.fullmatch(r"([A-Za-z]+)(\d+(?:\.\d+)?)", tok)
        if m:
            params[m.group(1)] = float(m.group(2))
    return params


def derive_metrics(row):
    """Calcula métricas derivadas a partir de las crudas."""
    r = dict(row)

    def div(a, b):
        a, b = r.get(a, np.nan), r.get(b, np.nan)
        return a / b if b and not np.isnan(b) else np.nan

    r["branch_mispred_rate"] = div("bp_mispredicted", "bp_committed")

    # Total de instrucciones comprometidas a partir de CPI/IPC no está disponible
    # directamente; usamos int + vec como aproximación del total.
    total = np.nansum([r.get("n_int", np.nan), r.get("n_vec", np.nan)])
    r["total_insts_approx"] = total if total > 0 else np.nan
    t = r["total_insts_approx"]

    def pct(x):
        v = r.get(x, np.nan)
        return 100 * v / t if t and not np.isnan(v) else np.nan

    r["pct_IntAlu"] = pct("t_IntAlu")
    r["pct_Simd"] = 100 * np.nansum(
        [r.get("t_SimdFloatAdd", np.nan), r.get("t_SimdFloatMultAcc", np.nan),
         r.get("t_SimdAlu", np.nan)]) / t if t and not np.isnan(t) else np.nan
    r["pct_load"] = pct("n_load")
    r["pct_store"] = pct("n_store")
    r["pct_control"] = pct("n_control")
    return r


def load_metrics(path, label=None):
    raw = parse_metrics_file(path)
    short = {k: raw.get(v, np.nan) for k, v in RAW.items()}
    row = derive_metrics(short)
    row["config"] = label or config_name_from_path(path)
    return row


def load_all(base_path, dir_path, pattern="*_metricas.txt"):
    """DataFrame con una fila por simulación (baseline primero, config='BASE')."""
    rows = [load_metrics(base_path, label="BASE")]
    base_abs = os.path.abspath(base_path)
    for p in sorted(glob.glob(os.path.join(dir_path, pattern))):
        if os.path.abspath(p) == base_abs:
            continue
        rows.append(load_metrics(p))

    df = pd.DataFrame(rows)
    # Parámetros parseados del nombre
    params = [params_from_name(c) if c != "BASE" else dict(BASELINE_PARAMS)
              for c in df["config"]]
    pdf = pd.DataFrame(params, index=df.index)
    df = pd.concat([df, pdf], axis=1)

    # Speedup respecto al baseline
    base_cpi = df.loc[df["config"] == "BASE", "cpi"].iloc[0]
    base_t = df.loc[df["config"] == "BASE", "simSeconds"].iloc[0]
    df["speedup"] = base_t / df["simSeconds"]
    df["speedup_cpi"] = base_cpi / df["cpi"]
    return df, list(pdf.columns)


# ----------------------------------------------------------------------------
# 2. COMPARADOR VS BASELINE
# ----------------------------------------------------------------------------

def compare_to_baseline(df, config, metrics=SUMMARY_METRICS):
    """Tabla: Métrica | Baseline | Nuevo | Δ% | Cambio | Veredicto."""
    base = df[df["config"] == "BASE"].iloc[0]
    new = df[df["config"] == config].iloc[0]
    rows = []
    for m in metrics:
        b, n = base.get(m, np.nan), new.get(m, np.nan)
        if m == "speedup":  # el baseline es 1.0 por definición
            b = 1.0
        delta = (n - b) / abs(b) * 100 if pd.notna(b) and b != 0 and pd.notna(n) else np.nan
        arrow = "–" if pd.isna(delta) or abs(delta) < 1e-9 else ("↑" if delta > 0 else "↓")
        d = "high" if m == "speedup" else DIRECTION.get(m, "neutral")
        if pd.isna(delta) or abs(delta) < 1e-9 or d == "neutral":
            verdict = "·"
        elif (delta < 0 and d == "low") or (delta > 0 and d == "high"):
            verdict = "✔ mejora"
        else:
            verdict = "✘ empeora"
        rows.append({"Métrica": m, "Baseline": b, "Nuevo": n,
                     "Δ%": delta, "Cambio": arrow, "Veredicto": verdict})
    return pd.DataFrame(rows)


def print_comparison(df, config):
    t = compare_to_baseline(df, config)
    print(f"\n=== {config}  vs  BASE ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:,.6g}"))
    return t


def style_comparison(table):
    """Versión coloreada para notebooks (requiere jinja2)."""
    def color(v):
        if "mejora" in str(v):
            return "color: green; font-weight: bold"
        if "empeora" in str(v):
            return "color: red; font-weight: bold"
        return ""
    return table.style.map(color, subset=["Veredicto"]).format(
        {"Baseline": "{:.6g}", "Nuevo": "{:.6g}", "Δ%": "{:+.2f}"})


def delta_matrix(df, metrics=("cpi", "ipc", "dcache_miss", "icache_miss",
                              "l2_miss", "branch_mispred_rate")):
    """Matriz configs x métricas con Δ% vs baseline."""
    out = {}
    for c in df["config"]:
        if c == "BASE":
            continue
        t = compare_to_baseline(df, c, list(metrics)).set_index("Métrica")
        out[c] = t["Δ%"]
    return pd.DataFrame(out).T


# ----------------------------------------------------------------------------
# 3. GRÁFICAS
# ----------------------------------------------------------------------------

def _save(fig, name, show):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  gráfica guardada: {path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_speedup_bars(df, show=False):
    """Barras de speedup por configuración (baseline = 1.0)."""
    d = df.sort_values("speedup")
    colors = ["tab:gray" if c == "BASE" else ("tab:green" if s > 1 else "tab:red")
              for c, s in zip(d["config"], d["speedup"])]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(d))))
    ax.barh(d["config"], d["speedup"], color=colors)
    ax.axvline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Speedup vs baseline")
    ax.set_title("Speedup por configuración")
    _save(fig, "barras_speedup.png", show)


def plot_delta_bars(df, config, show=False):
    """Barras de Δ% de las métricas clave para una configuración."""
    t = compare_to_baseline(df, config).dropna(subset=["Δ%"])
    t = t[~t["Métrica"].isin(["hostSeconds"])]
    colors = ["tab:green" if "mejora" in v else "tab:red" if "empeora" in v else "tab:gray"
              for v in t["Veredicto"]]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(t))))
    ax.barh(t["Métrica"], t["Δ%"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Δ% vs baseline")
    ax.set_title(f"Cambio porcentual: {config}")
    ax.invert_yaxis()
    _save(fig, f"delta_{config}.png", show)


def plot_sensitivity(df, params, metric="cpi", show=False):
    """CPI (o métrica) vs valor de cada parámetro: puntos + media por valor."""
    for p in params:
        sub = df.dropna(subset=[p])
        if sub[p].nunique() < 2:
            continue
        g = sub.groupby(p)[metric].agg(["mean", "min", "max"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(sub[p], sub[metric], alpha=0.4, label="simulaciones")
        ax.plot(g.index, g["mean"], "o-", color="tab:red", label="media")
        ax.fill_between(g.index, g["min"], g["max"], alpha=0.15, color="tab:red")
        ax.set_xlabel(p)
        ax.set_ylabel(metric)
        ax.set_title(f"Sensibilidad de {metric} a '{p}'")
        ax.legend()
        _save(fig, f"sensibilidad_{p}_{metric}.png", show)


def plot_heatmaps(df, params, metric="cpi", show=False):
    """Heatmap de la métrica (media) para cada par de parámetros."""
    useful = [p for p in params if df[p].nunique() > 1]
    for a, b in itertools.combinations(useful, 2):
        piv = df.pivot_table(index=a, columns=b, values=metric, aggfunc="mean")
        if piv.size < 4:
            continue
        fig, ax = plt.subplots(figsize=(6, 4.5))
        if sns is not None:
            sns.heatmap(piv, annot=True, fmt=".3f", cmap="viridis_r", ax=ax)
        else:
            im = ax.imshow(piv.values, cmap="viridis_r", aspect="auto")
            ax.set_xticks(range(len(piv.columns)), piv.columns)
            ax.set_yticks(range(len(piv.index)), piv.index)
            fig.colorbar(im, ax=ax)
        ax.set_title(f"{metric}: {a} × {b}")
        _save(fig, f"heatmap_{a}_{b}_{metric}.png", show)


# ----------------------------------------------------------------------------
# 4. ENERGÍA, EDP Y PARETO
# ----------------------------------------------------------------------------

def add_energy(df, mcpat_csv):
    """Une potencia (McPAT) y calcula energía y EDP."""
    pw = pd.read_csv(mcpat_csv)  # columnas: config, power_w
    df = df.merge(pw, on="config", how="left")
    delay = df[DELAY_METRIC] if DELAY_METRIC in df else df["simSeconds"]
    df["energy"] = df["power_w"] * delay
    df["edp"] = df["energy"] * delay
    return df


def pareto_front(x, y):
    """Índices no dominados (minimizar x e y)."""
    pts = sorted(zip(x, y, range(len(x))))
    front, best_y = [], np.inf
    for xi, yi, i in pts:
        if yi < best_y:
            front.append(i)
            best_y = yi
    return front


def plot_pareto(df, show=False):
    d = df.dropna(subset=["energy"]).reset_index(drop=True)
    if d.empty:
        print("  (sin datos de energía: se omite Pareto)")
        return
    delay_col = DELAY_METRIC if DELAY_METRIC in d else "simSeconds"
    f = pareto_front(d[delay_col].values, d["energy"].values)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d[delay_col], d["energy"], alpha=0.5, label="simulaciones")
    ax.plot(d.loc[f, delay_col], d.loc[f, "energy"], "r.-", label="frente de Pareto")
    b = d[d["config"] == "BASE"]
    if not b.empty:
        ax.scatter(b[delay_col], b["energy"], c="k", marker="*", s=200, label="baseline")
    ax.set_xlabel(f"Delay ({delay_col})")
    ax.set_ylabel("Energía")
    ax.set_title("Rendimiento vs. energía")
    ax.legend()
    _save(fig, "pareto_energia_rendimiento.png", show)


def best_configs(df):
    d = df.dropna(subset=["energy"])
    if d.empty:
        return None
    best = {
        "Mejor rendimiento": d.loc[d["simSeconds"].idxmin()],
        "Menor energía": d.loc[d["energy"].idxmin()],
        "Mejor EDP": d.loc[d["edp"].idxmin()],
    }
    out = pd.DataFrame(best).T[["config", "simSeconds", "cpi", "power_w", "energy", "edp"]]
    print("\n=== Mejores configuraciones ===")
    print(out.to_string())
    return out


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="archivo de métricas del baseline")
    ap.add_argument("--dir", default=".", help="carpeta con los *_metricas.txt")
    ap.add_argument("--pattern", default="*_metricas.txt")
    ap.add_argument("--mcpat", default=None, help="CSV opcional: config,power_w")
    ap.add_argument("--show", action="store_true", help="mostrar gráficas en pantalla")
    args = ap.parse_args()

    df, params = load_all(args.base, args.dir, args.pattern)
    print(f"Simulaciones cargadas: {len(df)} (incluye baseline)")
    print(f"Parámetros detectados en los nombres: {params or 'ninguno'}")

    # --- Tablas Δ% vs baseline ---
    for c in df["config"]:
        if c != "BASE":
            print_comparison(df, c)

    if len(df) > 1:
        print("\n=== Matriz de Δ% (configs × métricas) ===")
        print(delta_matrix(df).round(2).to_string())

    # --- Gráficas ---
    print("\nGenerando gráficas...")
    if len(df) > 1:
        plot_speedup_bars(df, args.show)
        for c in df["config"]:
            if c != "BASE":
                plot_delta_bars(df, c, args.show)
    if params:
        plot_sensitivity(df, params, "cpi", args.show)
        plot_heatmaps(df, params, "cpi", args.show)

    # --- Energía / EDP / Pareto ---
    if args.mcpat:
        df = add_energy(df, args.mcpat)
        plot_pareto(df, args.show)
        best_configs(df)

    # --- Exportar CSV (OPCIONAL: quita los '#' para activarlo) ---
    # os.makedirs(OUT_DIR, exist_ok=True)
    # df.to_csv(os.path.join(OUT_DIR, "resultados_completos.csv"), index=False)
    # delta_matrix(df).to_csv(os.path.join(OUT_DIR, "delta_vs_baseline.csv"))
    # for c in df["config"]:
    #     if c != "BASE":
    #         compare_to_baseline(df, c).to_csv(
    #             os.path.join(OUT_DIR, f"comparacion_{c}.csv"), index=False)
    # print(f"CSVs exportados en {OUT_DIR}/")


if __name__ == "__main__":
    main()
