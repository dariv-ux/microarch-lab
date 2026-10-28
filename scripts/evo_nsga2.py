#!/usr/bin/env python3
"""
Experimento 3: exploracion evolutiva (NSGA-II) del DSE de Cortex A76.

Objetivos (ambos a minimizar):
    1) CPI
    2) Energy = Power * CPI      (Power = Leakage + Runtime Dynamic de McPAT)
   EDP = Energy * CPI se calcula por individuo; su optimo siempre cae sobre el
   frente de Pareto (CPI, Energy), asi que un solo experimento da los tres
   mejores pedidos (rendimiento, energia y EDP).

Cada evaluacion = gem5 + conversor gem5->McPAT + McPAT, todo dentro del mismo
worker (misma tuberia que dse_exp2.py + edp_exp4.py). El XML plantilla NO se
modifica.

Genoma (14 genes). Parametros que McPAT no mapea (num_fu_*, predictor de
ramas, ...) se dejan en su valor por defecto. Anchos agrupados:
    front_width  -> fetch = decode = rename
    issue_width  -> issue = dispatch
    commit_width -> commit = wb
Latencias de cache ligadas al tamano (ver tablas LAT_*).

Reanudable: cada evaluacion se guarda en evo/sims/<id>/result.json y el estado
de las generaciones en evo/generations.json. Se puede ejecutar por tandas con
distinto --budget-hours; la cache evita repetir simulaciones.

Ejecutar desde la raiz de gem5:
    python3 evo_nsga2.py --smoke                      # prueba: solo config por defecto
    python3 evo_nsga2.py --workers 6 --budget-hours 8.5
    python3 evo_nsga2.py --workers 6 --budget-hours 6  # continuar otra tanda
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

ROOT = os.getcwd()
GEM5_BIN = os.path.join(ROOT, "build/ARM/gem5.fast")
GEM5_CFG = os.path.join(ROOT, "scripts/CortexA76_scripts_gem5/CortexA76.py")
WORKLOAD = os.path.join(ROOT, "workloads/mp3_dec/mp3_dec")
WORKLOAD_IN = os.path.join(ROOT, "workloads/mp3_dec/mp3dec_testfile.mp3")
WORKLOAD_ARGS = "-w mp3dec_outfile.wav " + WORKLOAD_IN

MCPAT_BIN = os.path.join(ROOT, "mcpat/mcpat")
CONVERTER = os.path.join(ROOT, "scripts/McPAT/gem5toMcPAT_cortexA76.py")
TEMPLATE = os.path.join(ROOT, "scripts/McPAT/ARM_A76_2.1GHz.xml")

OUT_DIR = os.path.join(ROOT, "evo")
SIMS_DIR = os.path.join(OUT_DIR, "sims")
GEN_PATH = os.path.join(OUT_DIR, "generations.json")
EVALS_CSV = os.path.join(OUT_DIR, "evals_all.csv")
PARETO_CSV = os.path.join(OUT_DIR, "pareto.csv")
HISTORY_CSV = os.path.join(OUT_DIR, "history.csv")
SUMMARY = os.path.join(OUT_DIR, "summary.txt")

# ----------------------------------------------------------------------
# Espacio de busqueda (14 genes) -- todos ordinales
# ----------------------------------------------------------------------
SPACE = OrderedDict([
    ("l1i_size", ["16kB", "32kB", "64kB", "128kB"]),
    ("l1d_size", ["16kB", "32kB", "64kB", "128kB"]),
    ("l2_size", ["128kB", "256kB", "512kB", "1MB"]),
    ("l1i_assoc", [2, 4, 8]),
    ("l1d_assoc", [2, 4, 8]),
    ("l2_assoc", [4, 8, 16]),
    ("front_width", [2, 4, 6, 8]),
    ("issue_width", [2, 4, 6, 8]),
    ("commit_width", [2, 4, 6, 8]),
    ("iq_entries", [16, 32, 48, 64]),
    ("rob_entries", [64, 128, 192, 256]),
    ("lq_entries", [32, 48, 68, 96]),
    ("sq_entries", [32, 48, 72, 96]),
    ("btb_entries", [1024, 2048, 4096, 8192]),
])
GENES = list(SPACE)

# Configuracion por defecto del Cortex A76 del laboratorio
DEFAULT = {
    "l1i_size": "64kB", "l1d_size": "64kB", "l2_size": "256kB",
    "l1i_assoc": 4, "l1d_assoc": 4, "l2_assoc": 8,
    "front_width": 4, "issue_width": 8, "commit_width": 4,
    "iq_entries": 16, "rob_entries": 128, "lq_entries": 68,
    "sq_entries": 72, "btb_entries": 4096,
}

# Latencia (ciclos) ligada al tamano: caches mas grandes -> mas lentas
LAT_L1I = {"16kB": 2, "32kB": 2, "64kB": 2, "128kB": 3}
LAT_L1D = {"16kB": 3, "32kB": 3, "64kB": 4, "128kB": 5}
LAT_L2 = {"128kB": 8, "256kB": 9, "512kB": 10, "1MB": 12}

RE_LEAK = re.compile(r"^\s*Total Leakage\s*=\s*([-+0-9.eE]+)\s*W")
RE_DYN = re.compile(r"^\s*Runtime Dynamic\s*=\s*([-+0-9.eE]+)\s*W")

STATS = ["simSeconds", "hostSeconds", "simInsts", "system.cpu.cpi", "system.cpu.ipc",
         "system.cpu.dcache.overallMissRate::total",
         "system.cpu.icache.overallMissRate::total",
         "system.cpu.l2cache.overallMissRate::total"]


# ----------------------------------------------------------------------
# Utilidades de genoma
# ----------------------------------------------------------------------
def genome_id(g):
    key = "|".join(f"{k}={g[k]}" for k in GENES)
    return "g_" + hashlib.md5(key.encode()).hexdigest()[:12]


def gem5_options(g):
    o = [
        f"--l1i_size={g['l1i_size']}", f"--l1i_assoc={g['l1i_assoc']}", f"--l1i_lat={LAT_L1I[g['l1i_size']]}",
        f"--l1d_size={g['l1d_size']}", f"--l1d_assoc={g['l1d_assoc']}", f"--l1d_lat={LAT_L1D[g['l1d_size']]}",
        f"--l2_size={g['l2_size']}", f"--l2_assoc={g['l2_assoc']}", f"--l2_lat={LAT_L2[g['l2_size']]}",
        f"--fetch_width={g['front_width']}", f"--decode_width={g['front_width']}",
        f"--rename_width={g['front_width']}",
        f"--issue_width={g['issue_width']}", f"--dispatch_width={g['issue_width']}",
        f"--commit_width={g['commit_width']}", f"--wb_width={g['commit_width']}",
        f"--iq_entries={g['iq_entries']}", f"--rob_entries={g['rob_entries']}",
        f"--lq_entries={g['lq_entries']}", f"--sq_entries={g['sq_entries']}",
        f"--btb_entries={g['btb_entries']}",
    ]
    return o


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Evaluacion de un individuo (se ejecuta en un worker)
# ----------------------------------------------------------------------
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


def parse_mcpat(path):
    """Primera aparicion de Total Leakage / Runtime Dynamic tras 'Processor:'."""
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


def simulate(genome, gen):
    gid = genome_id(genome)
    simdir = os.path.join(SIMS_DIR, gid)
    rpath = os.path.join(simdir, "result.json")
    if os.path.exists(rpath):
        with open(rpath) as f:
            return json.load(f)
    if os.path.isdir(simdir):
        shutil.rmtree(simdir)
    os.makedirs(simdir)

    base = {"id": gid, **{k: genome[k] for k in GENES}, "gen": gen}
    t_all = time.time()

    def fail(msg):
        print(f"[ERROR] {gid}: {msg}", flush=True)
        r = {**base, "ok": False, "error": msg}
        write_json(rpath, r)
        return r

    # 1) gem5
    cmd = [GEM5_BIN, f"--outdir={simdir}", GEM5_CFG, "-d", simdir,
           "-c", WORKLOAD, "-o", WORKLOAD_ARGS] + gem5_options(genome)
    t0 = time.time()
    with open(os.path.join(simdir, "simout.txt"), "w") as out:
        out.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        out.flush()
        rc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT).returncode
    t_gem5 = time.time() - t0
    stats = os.path.join(simdir, "stats.txt")
    cfgjson = os.path.join(simdir, "config.json")
    if rc != 0 or not os.path.exists(stats) or not os.path.exists(cfgjson):
        return fail(f"gem5 rc={rc}")
    s = read_stats(stats)
    cpi = s["system.cpu.cpi"]
    if cpi is None:
        return fail("sin CPI en stats.txt")
    for fn in os.listdir(simdir):          # liberar disco
        if fn.endswith(".wav"):
            os.remove(os.path.join(simdir, fn))

    # 2) conversor gem5 -> McPAT (escribe config.xml en cwd)
    t0 = time.time()
    conv = subprocess.run([sys.executable, CONVERTER, stats, cfgjson, TEMPLATE],
                          cwd=simdir, capture_output=True, text=True)
    t_conv = time.time() - t0
    xml = os.path.join(simdir, "config.xml")
    if conv.returncode != 0 or not os.path.exists(xml):
        return fail("conversor: " + conv.stderr[-300:])

    # 3) McPAT
    t0 = time.time()
    out_txt = os.path.join(simdir, "mcpat_out.txt")
    with open(out_txt, "w") as out:
        rc = subprocess.run([MCPAT_BIN, "-infile", xml, "-print_level", "1"],
                            cwd=simdir, stdout=out, stderr=subprocess.STDOUT).returncode
    t_mcpat = time.time() - t0
    leak, dyn = parse_mcpat(out_txt)
    if rc != 0 or leak is None or dyn is None:
        return fail(f"McPAT rc={rc} leak={leak} dyn={dyn}")

    power = leak + dyn
    energy = power * cpi          # formula de la guia
    edp = energy * cpi            # formula de la guia
    res = {
        **base, "ok": True,
        "cpi": cpi, "ipc": s["system.cpu.ipc"],
        "simSeconds": s["simSeconds"], "hostSeconds": s["hostSeconds"],
        "simInsts": s["simInsts"],
        "dcache_miss": s["system.cpu.dcache.overallMissRate::total"],
        "icache_miss": s["system.cpu.icache.overallMissRate::total"],
        "l2_miss": s["system.cpu.l2cache.overallMissRate::total"],
        "leakage_W": leak, "dynamic_W": dyn, "power_W": power,
        "energy": energy, "edp": edp,
        "energy_J": power * s["simSeconds"],
        "gem5_wall_s": t_gem5, "converter_seconds": t_conv,
        "mcpat_seconds": t_mcpat, "wall_total_s": time.time() - t_all,
        "finished_at": time.time(),
    }
    write_json(rpath, res)
    print(f"[OK] gen{gen} {gid} CPI={cpi:.4f} P={power:.3f}W EDP={edp:.4f} "
          f"({res['wall_total_s']/60:.1f} min)", flush=True)
    return res


# ----------------------------------------------------------------------
# NSGA-II
# ----------------------------------------------------------------------
def objective(r):
    return (r["cpi"], r["energy"])


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_nds(objs):
    n = len(objs)
    S = [[] for _ in range(n)]
    cnt = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].append(q)
            elif dominates(objs[q], objs[p]):
                cnt[p] += 1
        if cnt[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                cnt[q] -= 1
                if cnt[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(front, objs):
    d = {i: 0.0 for i in front}
    for m in range(len(objs[0])):
        s = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[s[0]][m], objs[s[-1]][m]
        d[s[0]] = d[s[-1]] = math.inf
        if hi > lo:
            for k in range(1, len(s) - 1):
                d[s[k]] += (objs[s[k + 1]][m] - objs[s[k - 1]][m]) / (hi - lo)
    return [d[i] for i in front]


def rank_and_crowd(objs):
    fronts = fast_nds(objs)
    rank = [0] * len(objs)
    crowd = [0.0] * len(objs)
    for r, f in enumerate(fronts):
        cd = crowding(f, objs)
        for i, c in zip(f, cd):
            rank[i] = r
            crowd[i] = c
    return rank, crowd


def nsga_select(ids, evals, mu):
    """Supervivientes (mu+lambda): por frente y, en el ultimo, por crowding."""
    objs = [objective(evals[i]) for i in ids]
    out = []
    for f in fast_nds(objs):
        if len(out) + len(f) <= mu:
            out += [ids[i] for i in f]
        else:
            cd = crowding(f, objs)
            order = sorted(range(len(f)), key=lambda k: -cd[k])
            out += [ids[f[k]] for k in order[: mu - len(out)]]
            break
    return out


def crossover(a, b, rng, pc=0.9):
    if rng.random() > pc:
        return dict(a)
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in GENES}


def mutate(g, rng, pm):
    g = dict(g)
    for k in GENES:
        if rng.random() < pm:
            vals = SPACE[k]
            i = vals.index(g[k])
            if rng.random() < 0.6:       # paso a un valor vecino
                j = min(len(vals) - 1, max(0, i + rng.choice([-1, 1])))
            else:                        # salto aleatorio
                j = rng.randrange(len(vals))
            g[k] = vals[j]
    return g


def random_genome(rng):
    return {k: rng.choice(SPACE[k]) for k in GENES}


def make_offspring(pop_ids, evals, gen, lam, seen_ids, args):
    """Determinista para (seed, gen, poblacion): al reanudar regenera los mismos hijos."""
    rng = random.Random(f"{args.seed}-{gen}")
    objs = [objective(evals[i]) for i in pop_ids]
    rank, crowd = rank_and_crowd(objs)
    n = len(pop_ids)

    def tournament():
        if n == 1:
            return 0
        a, b = rng.sample(range(n), 2)
        if rank[a] != rank[b]:
            return a if rank[a] < rank[b] else b
        return a if crowd[a] >= crowd[b] else b

    genome_of = {i: {k: evals[i][k] for k in GENES} for i in pop_ids}
    children, batch_ids = [], set()
    for _ in range(lam):
        child = cid = None
        for _try in range(40):
            p1, p2 = pop_ids[tournament()], pop_ids[tournament()]
            child = mutate(crossover(genome_of[p1], genome_of[p2], rng), rng, args.mut_prob)
            cid = genome_id(child)
            if cid not in seen_ids and cid not in batch_ids:
                break
        if cid not in batch_ids:
            children.append(child)
            batch_ids.add(cid)
    return children


# ----------------------------------------------------------------------
# Poblacion inicial, E/S y reportes
# ----------------------------------------------------------------------
def brute_force_seeds(path):
    seeds = []
    if not path or not os.path.exists(path):
        return seeds
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for key in ("cpi", "energy", "edp"):
        r = min(rows, key=lambda r: float(r[key]))
        g = dict(DEFAULT)
        for k in ("l1d_size", "l2_size", "iq_entries", "rob_entries"):
            v = int(r[k]) if k.endswith("entries") else r[k]
            if v in SPACE[k]:
                g[k] = v
        seeds.append(g)
    return seeds


def initial_population(mu, seeds, seed):
    rng = random.Random(f"{seed}-init")
    pop, ids = [], set()
    for g in [dict(DEFAULT)] + seeds:
        if genome_id(g) not in ids:
            pop.append(g)
            ids.add(genome_id(g))
    while len(pop) < mu:
        g = random_genome(rng)
        if genome_id(g) not in ids:
            pop.append(g)
            ids.add(genome_id(g))
    return pop[:mu] if len(pop) > mu else pop


def load_evals(retry_failed=False):
    evals = {}
    if not os.path.isdir(SIMS_DIR):
        return evals
    for d in os.listdir(SIMS_DIR):
        p = os.path.join(SIMS_DIR, d, "result.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            if not r.get("ok") and retry_failed:
                shutil.rmtree(os.path.join(SIMS_DIR, d))
                continue
            evals[r["id"]] = r
    return evals


def ok_rows(evals):
    return sorted([r for r in evals.values() if r.get("ok")], key=lambda r: r["finished_at"])


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def best_so_far(rows):
    if not rows:
        return {}
    return {"best_cpi": min(r["cpi"] for r in rows),
            "best_energy": min(r["energy"] for r in rows),
            "best_edp": min(r["edp"] for r in rows)}


def save_reports(evals, generations):
    rows = ok_rows(evals)
    if not rows:
        return
    write_csv(EVALS_CSV, rows)
    objs = [objective(r) for r in rows]
    write_csv(PARETO_CSV, sorted([rows[i] for i in fast_nds(objs)[0]], key=lambda r: r["cpi"]))
    hist = [{"gen": g["gen"], "n_evals": g["n_evals"], "pop_size": len(g["pop"]),
             **{k: g[k] for k in ("best_cpi", "best_energy", "best_edp")}} for g in generations]
    write_csv(HISTORY_CSV, hist)


def fmt(r):
    cfg = ", ".join(f"{k}={r[k]}" for k in GENES)
    return (f"{cfg}\n      CPI={r['cpi']:.4f}  Power={r['power_W']:.3f} W  "
            f"Energy={r['energy']:.4f}  EDP={r['edp']:.4f}")


def write_summary(evals, generations, bf_csv, wall_total):
    rows = ok_rows(evals)
    if not rows:
        return
    L = [f"Evaluaciones validas: {len(rows)} | generaciones completadas: {len(generations)}", ""]
    for title, key in [("Mejor RENDIMIENTO (CPI minimo)", "cpi"),
                       ("Mejor ENERGIA (Energy minima)", "energy"),
                       ("Mejor EDP (EDP minimo)", "edp")]:
        L.append(f"== Top 3 - {title} ==")
        for i, r in enumerate(sorted(rows, key=lambda r: r[key])[:3], 1):
            L.append(f"  {i}. {fmt(r)}")
        L.append("")
    d = evals.get(genome_id(DEFAULT))
    if d and d.get("ok"):
        L += ["Config por defecto A76: " + f"CPI={d['cpi']:.4f} Energy={d['energy']:.4f} EDP={d['edp']:.4f}", ""]
    if bf_csv and os.path.exists(bf_csv):
        with open(bf_csv, newline="") as f:
            bf = list(csv.DictReader(f))
        L.append("Mejor de la fuerza bruta (5 parametros, 1024 sims): "
                 f"CPI={min(float(r['cpi']) for r in bf):.4f} "
                 f"Energy={min(float(r['energy']) for r in bf):.4f} "
                 f"EDP={min(float(r['edp']) for r in bf):.4f}")
        L.append("")
    g5 = [r["hostSeconds"] for r in rows]
    gw = [r["gem5_wall_s"] for r in rows]
    tm = [r["mcpat_seconds"] for r in rows]
    tc = [r["converter_seconds"] for r in rows]
    L += ["== Velocidades ==",
          f"gem5 : {sum(g5)/len(g5):.1f} s/sim (hostSeconds) | pared {sum(gw)/len(gw):.1f} s | total {sum(g5)/3600:.2f} h",
          f"McPAT: {sum(tm)/len(tm):.2f} s/sim | total {sum(tm)/60:.2f} min",
          f"Conversor: {sum(tc)/len(tc):.2f} s/sim | total {sum(tc)/60:.2f} min",
          f"Tiempo real de esta tanda: {wall_total/3600:.2f} h"]
    text = "\n".join(L)
    with open(SUMMARY, "w") as f:
        f.write(text + "\n")
    print("\n" + text)


# ----------------------------------------------------------------------
# Bucle principal
# ----------------------------------------------------------------------
def evaluate(genomes, gen, ex, evals, deadline, args):
    """Evalua los genomas que faltan. Devuelve True si todos quedaron evaluados."""
    queue, seen = [], set()
    for g in genomes:
        gid = genome_id(g)
        if gid not in evals and gid not in seen:
            queue.append(g)
            seen.add(gid)
    running = {}

    def est_seconds():
        w = [r["wall_total_s"] for r in evals.values() if r.get("ok") and "wall_total_s" in r]
        return statistics.median(w[-30:]) if len(w) >= 3 else args.est_eval_min * 60

    while queue or running:
        while queue and len(running) < args.workers and time.time() + est_seconds() <= deadline:
            g = queue.pop(0)
            running[ex.submit(simulate, g, gen)] = g
        if not running:
            break                      # sin tiempo para lanzar mas
        done, _ = wait(running, return_when=FIRST_COMPLETED)
        for fut in done:
            running.pop(fut)
            try:
                r = fut.result()
            except Exception as e:     # no se guarda -> se reintenta en la proxima tanda
                print(f"[EXC] {e}", flush=True)
                continue
            evals[r["id"]] = r
    return not queue


def run_evolution(args):
    os.makedirs(SIMS_DIR, exist_ok=True)
    evals = load_evals(args.retry_failed)
    generations = json.load(open(GEN_PATH)) if os.path.exists(GEN_PATH) else []
    t_start = time.time()
    deadline = t_start + args.budget_hours * 3600
    seeds = brute_force_seeds(args.seed_csv)
    print(f"Evaluaciones previas: {len(evals)} | generaciones guardadas: {len(generations)} | "
          f"workers: {args.workers} | presupuesto: {args.budget_hours} h")

    def ok_ids(ids):
        return [i for i in ids if i in evals and evals[i].get("ok")]

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # ---- generacion 0 ----
        if not generations:
            init = initial_population(args.mu, seeds, args.seed)
            print(f"Generacion 0: {len(init)} individuos iniciales (defecto + semillas fuerza bruta + aleatorios)")
            if not evaluate(init, 0, ex, evals, deadline, args):
                print("Presupuesto agotado antes de terminar la generacion 0; reanuda para continuar.")
                save_reports(evals, generations)
                write_summary(evals, generations, args.seed_csv, time.time() - t_start)
                return
            pop = nsga_select(ok_ids([genome_id(g) for g in init]), evals, args.mu)
            generations.append({"gen": 0, "pop": pop, "n_evals": len(ok_rows(evals)),
                                **best_so_far(ok_rows(evals))})
            write_json(GEN_PATH, generations)
            save_reports(evals, generations)

        # ---- generaciones siguientes ----
        while True:
            last = generations[-1]
            g = last["gen"] + 1
            if args.max_gens is not None and g > args.max_gens:
                print(f"Alcanzado --max-gens={args.max_gens}")
                break
            pop = ok_ids(last["pop"])
            seen = {r["id"] for r in evals.values() if r.get("gen", 0) < g}
            children = make_offspring(pop, evals, g, args.lam, seen, args)
            new = sum(1 for c in children if genome_id(c) not in evals)
            print(f"\n=== Generacion {g}: {len(children)} hijos ({new} por simular) ===")
            if not evaluate(children, g, ex, evals, deadline, args):
                print("Presupuesto agotado a mitad de generacion; lo evaluado queda en cache.")
                break
            comb = list(OrderedDict.fromkeys(pop + ok_ids([genome_id(c) for c in children])))
            newpop = nsga_select(comb, evals, args.mu)
            generations.append({"gen": g, "pop": newpop, "n_evals": len(ok_rows(evals)),
                                **best_so_far(ok_rows(evals))})
            write_json(GEN_PATH, generations)
            save_reports(evals, generations)
            b = generations[-1]
            print(f"Gen {g} | evals={b['n_evals']} | mejor CPI={b['best_cpi']:.4f} "
                  f"energy={b['best_energy']:.4f} EDP={b['best_edp']:.4f}")
            if time.time() >= deadline:
                break

    save_reports(evals, generations)
    write_summary(evals, generations, args.seed_csv, time.time() - t_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--budget-hours", type=float, default=8.5)
    ap.add_argument("--mu", type=int, default=24, help="tamano de poblacion")
    ap.add_argument("--lam", type=int, default=24, help="hijos por generacion")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mut-prob", type=float, default=0.15, help="prob. de mutacion por gen")
    ap.add_argument("--max-gens", type=int, default=None)
    ap.add_argument("--est-eval-min", type=float, default=14.0,
                    help="estimacion inicial (min) de una evaluacion; evita lanzar sims que no caben en el presupuesto")
    ap.add_argument("--seed-csv", default="results_exp4_sorted_EDP.csv",
                    help="CSV de la fuerza bruta para sembrar la poblacion inicial ('' para no usar)")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="prueba: evalua solo la config por defecto con 1 worker y sale")
    args = ap.parse_args()

    for p in (GEM5_BIN, GEM5_CFG, WORKLOAD, WORKLOAD_IN, MCPAT_BIN, CONVERTER, TEMPLATE):
        if not os.path.exists(p):
            sys.exit(f"No se encontro '{p}'. Ejecuta desde la carpeta de gem5.")

    if args.smoke:
        os.makedirs(SIMS_DIR, exist_ok=True)
        print("Smoke test: config por defecto (gen 0)")
        r = simulate(dict(DEFAULT), 0)
        print(json.dumps(r, indent=2))
        return

    run_evolution(args)


if __name__ == "__main__":
    main()
