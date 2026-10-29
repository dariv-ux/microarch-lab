#!/usr/bin/env python3
"""
Experimento 3: exploracion por FUERZA BRUTA del DSE de Cortex A76.

Recorre las 1024 configuraciones (4^5) definidas en dse_exp2.py, lee los
resultados de cada carpeta sims/<id>/ y determina la mejor configuracion en
rendimiento (menor CPI). No lanza simulaciones nuevas.

Ejecutar desde la raiz de gem5 (donde esta la carpeta sims/):
    python3 brute_force_exp3.py
    python3 brute_force_exp3.py --sims-dir sims --out exp3_bruteforce --workers 5
"""

import argparse
import csv
import itertools
import json
import os
import statistics
import sys

# Debe coincidir exactamente con SPACE / DEFAULT de dse_exp2.py
SPACE = {
    "l1d_size": ["16kB", "32kB", "64kB", "128kB"],
    "l2_size": ["128kB", "256kB", "512kB", "1MB"],
    "num_fu_intALU": [1, 2, 3, 4],
    "iq_entries": [16, 32, 48, 64],
    "rob_entries": [64, 128, 192, 256],
}
PARAMS = list(SPACE)
DEFAULT = {"l1d_size": "64kB", "l2_size": "256kB", "num_fu_intALU": 2,
           "iq_entries": 16, "rob_entries": 128}

# Metricas que se leen de cada result.json
METRICAS = ["cpi", "ipc", "simSeconds", "hostSeconds", "wall_s",
            "dcache_miss", "icache_miss", "l2_miss", "l3_miss"]


def cfg_id(cfg):
    return "_".join(f"{k}-{cfg[k]}" for k in PARAMS)


def leer_simInsts(stats_path):
    """Lee simInsts de stats.txt (para verificar que todas ejecutan lo mismo)."""
    try:
        with open(stats_path) as f:
            for linea in f:
                p = linea.split()
                if len(p) >= 2 and p[0] == "simInsts":
                    return float(p[1])
    except OSError:
        pass
    return None


def cargar(sims_dir):
    """Recorre las 1024 carpetas. Devuelve (filas, faltantes, fallidas)."""
    filas, faltantes, fallidas = [], [], []
    for valores in itertools.product(*SPACE.values()):
        cfg = dict(zip(PARAMS, valores))
        sid = cfg_id(cfg)
        carpeta = os.path.join(sims_dir, sid)
        rpath = os.path.join(carpeta, "result.json")
        if not os.path.exists(rpath):
            faltantes.append(sid)
            continue
        with open(rpath) as f:
            r = json.load(f)
        if not r.get("ok") or r.get("cpi") is None:
            fallidas.append(sid)
            continue
        fila = {"id": sid, **cfg}
        for m in METRICAS:
            fila[m] = r.get(m)
        fila["simInsts"] = leer_simInsts(os.path.join(carpeta, "stats.txt"))
        filas.append(fila)
    return filas, faltantes, fallidas


def guardar_csv(path, filas, campos):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)


def efecto_por_parametro(filas):
    """CPI medio/min/max al fijar cada valor de cada parametro."""
    out = []
    for p in PARAMS:
        for v in SPACE[p]:
            cpis = [r["cpi"] for r in filas if r[p] == v]
            if cpis:
                out.append({"parametro": p, "valor": v, "n": len(cpis),
                            "cpi_medio": statistics.mean(cpis),
                            "cpi_min": min(cpis), "cpi_max": max(cpis)})
    return out


def graficar(efectos, ruta):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[AVISO] matplotlib no esta instalado; se omite la grafica.")
        return False
    fig, axs = plt.subplots(1, len(PARAMS), figsize=(4 * len(PARAMS), 3.6), sharey=True)
    for ax, p in zip(axs, PARAMS):
        e = [x for x in efectos if x["parametro"] == p]
        xs = [str(x["valor"]) for x in e]
        ax.plot(xs, [x["cpi_medio"] for x in e], "o-", label="medio")
        ax.fill_between(xs, [x["cpi_min"] for x in e], [x["cpi_max"] for x in e],
                        alpha=0.2, label="min-max")
        ax.set_title(p)
        ax.set_xlabel("valor")
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("CPI")
    axs[0].legend()
    fig.suptitle("Rendimiento (CPI) vs recursos - fuerza bruta")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Experimento 3: fuerza bruta sobre las simulaciones.")
    ap.add_argument("--sims-dir", default="sims")
    ap.add_argument("--out", default="exp3_bruteforce")
    ap.add_argument("--workers", type=int, default=5,
                    help="simulaciones en paralelo usadas en dse_exp2.py (para estimar tiempo total)")
    args = ap.parse_args()

    if not os.path.isdir(args.sims_dir):
        sys.exit(f"No existe '{args.sims_dir}'. Ejecuta desde la carpeta de gem5.")
    os.makedirs(args.out, exist_ok=True)

    total_esperado = len(list(itertools.product(*SPACE.values())))
    filas, faltantes, fallidas = cargar(args.sims_dir)
    print(f"Configuraciones esperadas: {total_esperado}")
    print(f"Validas: {len(filas)} | faltantes: {len(faltantes)} | fallidas: {len(fallidas)}")
    if faltantes or fallidas:
        print("[AVISO] El espacio NO esta completo; el optimo es solo el mejor de las validas.")
        with open(os.path.join(args.out, "incompletas.txt"), "w") as f:
            f.write("FALTANTES:\n" + "\n".join(faltantes) + "\nFALLIDAS:\n" + "\n".join(fallidas) + "\n")
    if not filas:
        sys.exit("No hay resultados validos.")

    # Verificacion: todas deben simular el mismo numero de instrucciones
    insts = {r["simInsts"] for r in filas if r["simInsts"] is not None}
    if len(insts) > 1:
        print(f"[AVISO] simInsts distintos entre simulaciones ({len(insts)} valores): "
              "comparar CPI podria no ser equivalente a comparar tiempo.")
    elif len(insts) == 1:
        print(f"Verificacion OK: todas simulan {int(next(iter(insts))):,} instrucciones.")

    # Busqueda exhaustiva: orden por CPI (menor = mejor)
    filas.sort(key=lambda r: r["cpi"])
    for i, r in enumerate(filas, start=1):
        r["ranking"] = i
    mejor, peor = filas[0], filas[-1]
    ref = next((r for r in filas if r["id"] == cfg_id(DEFAULT)), None)

    campos = ["ranking", "id"] + PARAMS + METRICAS + ["simInsts"]
    guardar_csv(os.path.join(args.out, "todas_ordenadas_por_cpi.csv"), filas, campos)
    guardar_csv(os.path.join(args.out, "top10.csv"), filas[:10], campos)

    efectos = efecto_por_parametro(filas)
    guardar_csv(os.path.join(args.out, "efecto_por_parametro.csv"), efectos,
                ["parametro", "valor", "n", "cpi_medio", "cpi_min", "cpi_max"])
    graficar(efectos, os.path.join(args.out, "cpi_vs_recursos.png"))

    # Velocidad de simulacion
    host = [r["hostSeconds"] for r in filas if r["hostSeconds"] is not None]
    wall = [r["wall_s"] for r in filas if r["wall_s"] is not None]
    velocidad = {
        "n_simulaciones": len(filas),
        "gem5_hostSeconds_medio": statistics.mean(host) if host else None,
        "gem5_hostSeconds_min": min(host) if host else None,
        "gem5_hostSeconds_max": max(host) if host else None,
        "gem5_hostSeconds_suma": sum(host) if host else None,
        "wall_s_suma": sum(wall) if wall else None,
        "workers": args.workers,
        "tiempo_total_estimado_paralelo_h": (sum(wall) / args.workers / 3600) if wall else None,
        "tiempo_total_secuencial_h": (sum(wall) / 3600) if wall else None,
    }
    with open(os.path.join(args.out, "velocidad_simulacion.json"), "w") as f:
        json.dump(velocidad, f, indent=2)

    resumen = {
        "mejor_cpi": {k: mejor[k] for k in ["id"] + PARAMS + ["cpi", "ipc", "simSeconds"]},
        "peor_cpi": {k: peor[k] for k in ["id"] + PARAMS + ["cpi"]},
        "default": ({k: ref[k] for k in ["id", "cpi"]} if ref else None),
        "speedup_mejor_vs_default": (ref["cpi"] / mejor["cpi"]) if ref else None,
    }
    with open(os.path.join(args.out, "mejor_configuracion.json"), "w") as f:
        json.dump(resumen, f, indent=2)

    print("\n=== MEJOR CONFIGURACION (menor CPI) ===")
    for p in PARAMS:
        print(f"  {p:15s}: {mejor[p]}")
    print(f"  CPI = {mejor['cpi']:.4f}  (IPC = {mejor['ipc']:.4f})")
    if ref:
        print(f"  CPI por defecto = {ref['cpi']:.4f} -> speedup = {resumen['speedup_mejor_vs_default']:.3f}x")
    print(f"  Peor CPI = {peor['cpi']:.4f}")
    print("\nTop 5:")
    for r in filas[:5]:
        print(f"  #{r['ranking']}: CPI={r['cpi']:.4f}  " + " ".join(f"{p}={r[p]}" for p in PARAMS))
    if host:
        print(f"\ngem5: hostSeconds medio = {velocidad['gem5_hostSeconds_medio']:.1f} s por simulacion")
    if wall:
        print(f"Tiempo total (estimado con {args.workers} en paralelo) = "
              f"{velocidad['tiempo_total_estimado_paralelo_h']:.2f} h")
    print(f"\nResultados guardados en: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
