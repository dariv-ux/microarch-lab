#!/usr/bin/env python3
"""
Experimento 4: McPAT + Energy + EDP sobre las simulaciones del DSE (exp. 2).

Para cada carpeta sims/<id>/ con result.json valido:
  1. Lee el CPI.
  2. Convierte stats.txt + config.json -> config.xml (gem5toMcPAT_cortexA76.py).
  3. Ejecuta McPAT y extrae Total Leakage y Runtime Dynamic del bloque Processor.
  4. Power  = Leakage + Runtime Dynamic
     Energy = Power  * CPI
     EDP    = Energy * CPI          (formulas de la guia)

Todo se guarda en exp4_EDP/ (al lado de build/, scripts/, sims/...):
  exp4_EDP/mcpat/<id>/{config.xml, mcpat_out.txt, mcpat_result.json}
  exp4_EDP/results_exp4.csv              (incremental)
  exp4_EDP/results_exp4_sorted_EDP.csv   (final, mejores primero)
  exp4_EDP/summary.txt                   (top 3 por CPI, energia y EDP + tiempos)

Reanudable: si existe mcpat_result.json para una simulacion, se salta.

Ejecutar desde la raiz de gem5:
    python3 edp_exp4.py --workers 4
    python3 edp_exp4.py --limit 2        # prueba rapida
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.getcwd()
SIMS_DIR = os.path.join(ROOT, "sims")
MCPAT_BIN = os.path.join(ROOT, "mcpat/mcpat")
CONVERTER = os.path.join(ROOT, "scripts/McPAT/gem5toMcPAT_cortexA76.py")
TEMPLATE = os.path.join(ROOT, "scripts/McPAT/ARM_A76_2.1GHz.xml")

OUT_DIR = os.path.join(ROOT, "exp4_EDP")
MCPAT_DIR = os.path.join(OUT_DIR, "mcpat")
CSV_PATH = os.path.join(OUT_DIR, "results_exp4.csv")
CSV_SORTED = os.path.join(OUT_DIR, "results_exp4_sorted_EDP.csv")
SUMMARY = os.path.join(OUT_DIR, "summary.txt")

PARAMS = ["l1d_size", "l2_size", "num_fu_intALU", "iq_entries", "rob_entries"]

RE_LEAK = re.compile(r"^\s*Total Leakage\s*=\s*([-+0-9.eE]+)\s*W")
RE_DYN = re.compile(r"^\s*Runtime Dynamic\s*=\s*([-+0-9.eE]+)\s*W")


def parse_mcpat(path):
    """Toma la PRIMERA aparicion de Total Leakage y Runtime Dynamic
    despues de la linea 'Processor:' (valores del procesador completo)."""
    leak = dyn = None
    in_proc = False
    with open(path) as f:
        for line in f:
            if line.strip() == "Processor:":
                in_proc = True
                continue
            if not in_proc:
                continue
            if leak is None:
                m = RE_LEAK.match(line)
                if m:
                    leak = float(m.group(1))
            if dyn is None:
                m = RE_DYN.match(line)
                if m:
                    dyn = float(m.group(1))
            if leak is not None and dyn is not None:
                break
    return leak, dyn


def process(sid):
    """Corre conversor + McPAT para una simulacion. Devuelve un dict."""
    simdir = os.path.join(SIMS_DIR, sid)
    outdir = os.path.join(MCPAT_DIR, sid)
    res_path = os.path.join(outdir, "mcpat_result.json")

    with open(os.path.join(simdir, "result.json")) as f:
        sim = json.load(f)

    # Reanudable
    if os.path.exists(res_path):
        with open(res_path) as f:
            return json.load(f)

    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)

    stats = os.path.join(simdir, "stats.txt")
    cfgjson = os.path.join(simdir, "config.json")
    if not (os.path.exists(stats) and os.path.exists(cfgjson)):
        print(f"[ERROR] {sid}: faltan stats.txt o config.json", flush=True)
        return {"id": sid, "ok": False}

    # El conversor escribe config.xml en el directorio actual -> cwd=outdir
    t0 = time.time()
    conv = subprocess.run(
        [sys.executable, CONVERTER, stats, cfgjson, TEMPLATE],
        cwd=outdir, capture_output=True, text=True)
    t_conv = time.time() - t0
    xml = os.path.join(outdir, "config.xml")
    if conv.returncode != 0 or not os.path.exists(xml):
        print(f"[ERROR] {sid}: conversor fallo\n{conv.stderr[-500:]}", flush=True)
        return {"id": sid, "ok": False}

    t0 = time.time()
    out_txt = os.path.join(outdir, "mcpat_out.txt")
    with open(out_txt, "w") as out:
        rc = subprocess.run([MCPAT_BIN, "-infile", xml, "-print_level", "1"],
                            cwd=outdir, stdout=out, stderr=subprocess.STDOUT).returncode
    t_mcpat = time.time() - t0

    leak, dyn = parse_mcpat(out_txt)
    if rc != 0 or leak is None or dyn is None:
        print(f"[ERROR] {sid}: McPAT rc={rc}, leak={leak}, dyn={dyn}", flush=True)
        return {"id": sid, "ok": False}

    cpi = sim["cpi"]
    power = leak + dyn
    energy = power * cpi          # formula de la guia
    edp = energy * cpi            # formula de la guia
    res = {
        "id": sid,
        **{k: sim[k] for k in PARAMS},
        "ok": True,
        "cpi": cpi,
        "simSeconds": sim["simSeconds"],
        "hostSeconds": sim["hostSeconds"],
        "leakage_W": leak,
        "dynamic_W": dyn,
        "power_W": power,
        "energy": energy,
        "edp": edp,
        # referencia: energia real en Joules (mismo ranking que con CPI)
        "energy_J": power * sim["simSeconds"],
        "converter_seconds": t_conv,
        "mcpat_seconds": t_mcpat,
    }
    with open(res_path, "w") as f:
        json.dump(res, f)
    print(f"[OK] {sid}  CPI={cpi:.4f}  P={power:.3f}W  EDP={edp:.4f}  "
          f"(mcpat {t_mcpat:.1f}s)", flush=True)
    return res


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(r):
    cfg = ", ".join(f"{k}={r[k]}" for k in PARAMS)
    return (f"{cfg}\n      CPI={r['cpi']:.4f}  Power={r['power_W']:.3f} W  "
            f"Energy={r['energy']:.4f}  EDP={r['edp']:.4f}")


def write_summary(rows, wall_total):
    lines = [f"Simulaciones procesadas: {len(rows)}", ""]
    for title, key in [("Mejor RENDIMIENTO (CPI minimo)", "cpi"),
                       ("Mejor ENERGIA (Energy minima)", "energy"),
                       ("Mejor EDP (EDP minimo)", "edp")]:
        lines.append(f"== Top 3 - {title} ==")
        for i, r in enumerate(sorted(rows, key=lambda r: r[key])[:3], 1):
            lines.append(f"  {i}. {fmt(r)}")
        lines.append("")

    pw = [r["power_W"] for r in rows]
    lines += [f"Rango de potencia: {min(pw):.3f} W - {max(pw):.3f} W", ""]

    t_m = [r["mcpat_seconds"] for r in rows]
    t_c = [r["converter_seconds"] for r in rows]
    t_g = [r["hostSeconds"] for r in rows]
    lines += [
        "== Velocidades ==",
        f"gem5  : {sum(t_g)/len(t_g):.1f} s por simulacion | total {sum(t_g)/3600:.2f} h (suma de hostSeconds)",
        f"McPAT : {sum(t_m)/len(t_m):.2f} s por simulacion | total {sum(t_m)/60:.2f} min (suma)",
        f"Conversor gem5->McPAT: {sum(t_c)/len(t_c):.2f} s por simulacion | total {sum(t_c)/60:.2f} min (suma)",
        f"Tiempo real de esta ejecucion del experimento 4: {wall_total/60:.2f} min",
    ]
    text = "\n".join(lines)
    with open(SUMMARY, "w") as f:
        f.write(text + "\n")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="maximo de simulaciones a procesar (pruebas)")
    args = ap.parse_args()

    for p in (SIMS_DIR, MCPAT_BIN, CONVERTER, TEMPLATE):
        if not os.path.exists(p):
            sys.exit(f"No se encontro '{p}'. Ejecuta desde la carpeta de gem5.")
    os.makedirs(MCPAT_DIR, exist_ok=True)

    # Solo simulaciones validas (result.json con ok=True)
    ids = []
    for sid in sorted(os.listdir(SIMS_DIR)):
        rp = os.path.join(SIMS_DIR, sid, "result.json")
        if os.path.exists(rp):
            with open(rp) as f:
                if json.load(f).get("ok"):
                    ids.append(sid)
    if args.limit:
        ids = ids[: args.limit]
    print(f"Simulaciones validas a procesar: {len(ids)} | workers: {args.workers}")

    rows = []
    failed = []
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, sid): sid for sid in ids}
        for fut in as_completed(futs):
            r = fut.result()
            if r.get("ok"):
                rows.append(r)
                write_csv(CSV_PATH, rows)   # incremental
            else:
                failed.append(futs[fut])
    wall = time.time() - t_start

    if not rows:
        sys.exit("No se proceso ninguna simulacion con exito.")

    rows.sort(key=lambda r: r["edp"])       # mejores primero
    write_csv(CSV_SORTED, rows)
    write_summary(rows, wall)

    if failed:
        print(f"\n[AVISO] {len(failed)} simulaciones fallaron: {failed[:5]}...")
    print(f"\nCSV ordenado: {CSV_SORTED}")


if __name__ == "__main__":
    main()
