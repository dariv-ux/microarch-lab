#!/usr/bin/env python3
"""
Experimento 2: DSE de Cortex A76 en gem5 (sin McPAT).

Espacio: 4^5 = 1024 combinaciones. Se recorren en orden aleatorio (semilla fija),
5 simulaciones en paralelo, hasta agotar el presupuesto de tiempo.
Reanudable: las simulaciones ya terminadas (sims/<id>/result.json) se saltan.

Ejecutar desde la raiz de gem5:
    python3 dse_exp2.py --workers 5 --budget-hours 8.5
"""

import argparse
import csv
import itertools
import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.getcwd()
GEM5_BIN = os.path.join(ROOT, "build/ARM/gem5.fast")
GEM5_CFG = os.path.join(ROOT, "scripts/CortexA76_scripts_gem5/CortexA76.py")
WORKLOAD = os.path.join(ROOT, "workloads/mp3_dec/mp3_dec")
WORKLOAD_IN = os.path.join(ROOT, "workloads/mp3_dec/mp3dec_testfile.mp3")
WORKLOAD_ARGS = "-w mp3dec_outfile.wav " + WORKLOAD_IN  # el wav sale en el dir de cada sim

SIMS_DIR = os.path.join(ROOT, "sims")
CSV_PATH = os.path.join(ROOT, "results_exp2.csv")

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

STATS = ["simSeconds", "hostSeconds", "system.cpu.cpi", "system.cpu.ipc",
         "system.cpu.dcache.overallMissRate::total",
         "system.cpu.icache.overallMissRate::total",
         "system.cpu.l2cache.overallMissRate::total",
         "system.cpu.l3cache.overallMissRate::total"]


def cfg_id(cfg):
    return "_".join(f"{k}-{cfg[k]}" for k in PARAMS)


def read_stats(path):
    out = {n: None for n in STATS}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and p[0] in out and out[p[0]] is None:
                try:
                    out[p[0]] = float(p[1])
                except ValueError:
                    pass
    return out


def simulate(cfg):
    sid = cfg_id(cfg)
    simdir = os.path.join(SIMS_DIR, sid)
    result_path = os.path.join(simdir, "result.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            return json.load(f)

    if os.path.isdir(simdir):
        shutil.rmtree(simdir)
    os.makedirs(simdir)

    cmd = [GEM5_BIN, f"--outdir={simdir}", GEM5_CFG, "-d", simdir,
           "-c", WORKLOAD, "-o", WORKLOAD_ARGS] + [f"--{k}={cfg[k]}" for k in PARAMS]
    t0 = time.time()
    with open(os.path.join(simdir, "simout.txt"), "w") as out:
        out.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        out.flush()
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0

    stats = os.path.join(simdir, "stats.txt")
    if rc != 0 or not os.path.exists(stats):
        print(f"[ERROR] {sid} (rc={rc})", flush=True)
        return {"id": sid, **cfg, "ok": False}

    s = read_stats(stats)
    res = {"id": sid, **cfg, "ok": True,
           "simSeconds": s["simSeconds"], "hostSeconds": s["hostSeconds"],
           "cpi": s["system.cpu.cpi"], "ipc": s["system.cpu.ipc"],
           "dcache_miss": s["system.cpu.dcache.overallMissRate::total"],
           "icache_miss": s["system.cpu.icache.overallMissRate::total"],
           "l2_miss": s["system.cpu.l2cache.overallMissRate::total"],
           "l3_miss": s["system.cpu.l3cache.overallMissRate::total"],
           "wall_s": wall}
    with open(result_path, "w") as f:
        json.dump(res, f)
    for fn in os.listdir(simdir):  # liberar disco
        if fn.endswith(".wav"):
            os.remove(os.path.join(simdir, fn))
    print(f"[OK] {sid}  CPI={res['cpi']}  ({wall/60:.1f} min)", flush=True)
    return res


def save_csv(done):
    if not done:
        return
    rows = list(done.values())
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--budget-hours", type=float, default=8.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="maximo de simulaciones (util para pruebas, ej. --limit 1)")
    args = ap.parse_args()

    for p in (GEM5_BIN, GEM5_CFG, WORKLOAD, WORKLOAD_IN):
        if not os.path.exists(p):
            sys.exit(f"No se encontro '{p}'. Ejecuta desde la carpeta de gem5.")
    os.makedirs(SIMS_DIR, exist_ok=True)

    # Todo el espacio, en orden aleatorio reproducible; la config por defecto va primero
    allc = [dict(zip(PARAMS, v)) for v in itertools.product(*SPACE.values())]
    random.Random(args.seed).shuffle(allc)
    allc = [DEFAULT] + [c for c in allc if cfg_id(c) != cfg_id(DEFAULT)]
    if args.limit:
        allc = allc[: args.limit]

    # Cargar lo ya hecho
    done = {}
    for c in allc:
        p = os.path.join(SIMS_DIR, cfg_id(c), "result.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            if r.get("ok"):
                done[r["id"]] = r
    pending_cfgs = [c for c in allc if cfg_id(c) not in done]
    print(f"Espacio: {len(allc)} configs | ya hechas: {len(done)} | por hacer: {len(pending_cfgs)}")

    deadline = time.time() + args.budget_hours * 3600
    it = iter(pending_cfgs)
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        running = set()

        def submit_next():
            if time.time() >= deadline:
                return False
            try:
                running.add(ex.submit(simulate, next(it)))
                return True
            except StopIteration:
                return False

        for _ in range(args.workers):
            submit_next()
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in finished:
                running.discard(fut)
                r = fut.result()
                if r.get("ok"):
                    done[r["id"]] = r
                    save_csv(done)
                submit_next()

    print(f"\nTerminado: {len(done)} simulaciones en {(time.time()-t_start)/3600:.2f} h")
    print(f"CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()
