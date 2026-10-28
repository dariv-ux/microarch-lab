#!/usr/bin/env python3
"""
Ejecuta 3 simulaciones de gem5 (Cortex A76) con distintos parametros
microarquitectonicos y guarda los resultados numerados:

    results/1_config.ini  results/1_stats.txt  results/1_simout.txt
    results/1_metricas.txt  results/1_config.json
    (igual para 2 y 3)

Debe ejecutarse desde la carpeta raiz de gem5 (donde estan build/,
scripts/ y workloads/).

Uso 1 (interactivo, el script te pregunta los cambios de cada simulacion):
    python3 run_sims.py

Uso 2 (todo por linea de comandos, una opcion --sim por simulacion):
    python3 run_sims.py \
        --sim "--l1i_size=128kB" \
        --sim "--l1i_size=64kB --l1d_assoc=8" \
        --sim ""

Una cadena vacia significa "configuracion por defecto".
"""

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

# ----------------------------------------------------------------------
# Rutas por defecto (segun el quick start del laboratorio)
# ----------------------------------------------------------------------
GEM5_BIN = "./build/ARM/gem5.fast"
GEM5_CFG = "scripts/CortexA76_scripts_gem5/CortexA76.py"
WORKLOAD = "workloads/mp3_dec/mp3_dec"
WORKLOAD_ARGS = "-w mp3dec_outfile.wav workloads/mp3_dec/mp3dec_testfile.mp3"
M5OUT = "m5out"          # carpeta temporal donde gem5 escribe sus salidas
RESULTS = "results/exp1_pipeline"      # carpeta donde se guardan los resultados numerados
N_SIMS = 3


def pedir_parametros_interactivo(n_sims):
    """Pregunta por terminal los cambios del procesador de cada simulacion."""
    print("Escribe los parametros de gem5 para cada simulacion.")
    print("Ejemplo: --l1i_size=128kB --l1d_assoc=8")
    print("(Enter vacio = configuracion por defecto)\n")
    sims = []
    for i in range(1, n_sims + 1):
        sims.append(input(f"Simulacion {i} -> parametros: ").strip())
    return sims


def leer_metricas(stats_path, salida_path=None):
    """Extrae simSeconds, hostSeconds y cpi de stats.txt (si existen).

    Si se indica `salida_path`, guarda ademas las metricas en ese archivo
    de texto con el formato 'metrica: valor'.
    """
    buscadas = {
        # Tempora simulationis
        "simSeconds": None,
        "hostSeconds": None,

        # Efficientia et Pipeline
        "system.cpu.cpi": None,
        "system.cpu.ipc": None,
        "system.cpu.issueRate": None,
        "system.cpu.fetch.nisnDist::max_value": None,
        "system.cpu.commit.numCommittedDist::max_value": None,
    
        # Adumbratio per genus praecepti (Commit)
        "system.cpu.commitStats0.numIntInsts": None,
        "system.cpu.commitStats0.committedInstType::IntAlu": None,
        "system.cpu.commitStats0.numFpInsts": None,
        "system.cpu.commitStats0.numVecInsts": None,
        "system.cpu.commitStats0.committedInstType::SimdFloatAdd": None,
        "system.cpu.commitStats0.committedInstType::SimdFloatMultAcc": None,
        "system.cpu.commitStats0.committedInstType::SimdAlu": None,
        "system.cpu.commitStats0.numLoadInsts": None,
        "system.cpu.commitStats0.numStoreInsts": None,
        "system.cpu.commitStats0.numMemRefs": None,
        "system.cpu.commitStats0.committedControl::IsControl": None,

        # Unitates functionalium (FU)
        "system.cpu.statFuBusy::MemWrite": None,
        "system.cpu.fuBusyRate": None,

        # Proportiones delectorum latibuli (Cache Miss Rates)
        "system.cpu.dcache.overallMissRate::total": None,
        "system.cpu.icache.overallMissRate::total": None,
        "system.cpu.l2cache.overallMissRate::total": None,
        "system.cpu.l3cache.overallMissRate::total": None,

        # Praedictio saltuum
        "system.cpu.branchPred.committed_0::total": None,
        "system.cpu.branchPred.mispredicted_0::total": None,
    }
    with open(stats_path) as f:
        for linea in f:
            partes = linea.split()
            if len(partes) >= 2 and partes[0] in buscadas and buscadas[partes[0]] is None:
                buscadas[partes[0]] = partes[1]

    if salida_path is not None:
        with open(salida_path, "w") as out:
            for nombre, valor in buscadas.items():
                out.write(f"{nombre}: {valor}\n")

    return buscadas


def correr_simulacion(num, params):
    """Corre una simulacion y guarda los archivos con el numero `num`."""
    # Limpiar m5out para no copiar resultados viejos por error
    if os.path.isdir(M5OUT):
        shutil.rmtree(M5OUT)

    cmd = [
        GEM5_BIN,
        f"--outdir={M5OUT}",
        GEM5_CFG,
        "-c", WORKLOAD,
        "-o", WORKLOAD_ARGS,
    ] + shlex.split(params)

    print(f"\n=== Simulacion {num} ===")
    print("Comando:", " ".join(shlex.quote(c) for c in cmd))

    t0 = time.time()
    simout_path = os.path.join(RESULTS, f"{num}_simout.txt")
    with open(simout_path, "w") as out:
        out.write("CMD: " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        out.flush()
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    if proc.returncode != 0:
        print(f"[ERROR] La simulacion {num} fallo (codigo {proc.returncode}). "
              f"Revisa {simout_path}")
        return False

    # Guardar config.ini y stats.txt con el numero de simulacion
    for nombre in ("config.ini", "stats.txt"):
        origen = os.path.join(M5OUT, nombre)
        if os.path.exists(origen):
            shutil.copy(origen, os.path.join(RESULTS, f"{num}_{nombre}"))
        else:
            print(f"[AVISO] No se encontro {origen}")

    # config.json es opcional (lo usa McPAT): se guarda como i_config.json
    origen_json = os.path.join(M5OUT, "config.json")
    if os.path.exists(origen_json):
        shutil.copy(origen_json, os.path.join(RESULTS, f"{num}_config.json"))

    # Leer metricas y guardarlas en i_metricas.txt
    metricas_path = os.path.join(RESULTS, f"{num}_metricas.txt")
    m = leer_metricas(os.path.join(RESULTS, f"{num}_stats.txt"), metricas_path)
    print(f"Terminada en {dt:.1f} s (tiempo real)")
    print(f"  simSeconds : {m['simSeconds']}")
    print(f"  hostSeconds: {m['hostSeconds']}")
    print(f"  CPI        : {m['system.cpu.cpi']}")
    print(f"Guardados en {RESULTS}/: {num}_config.ini, {num}_stats.txt, "
          f"{num}_metricas.txt, {num}_simout.txt")
    return True


def main():
    ap = argparse.ArgumentParser(description="Ejecuta 3 simulaciones gem5 A76.")
    ap.add_argument("--sim", action="append", default=None,
                    help="Parametros de una simulacion (repetir 3 veces). "
                         'Ej: --sim "--l1i_size=128kB"')
    args = ap.parse_args()

    # Verificaciones basicas
    for ruta in (GEM5_BIN, GEM5_CFG, WORKLOAD):
        if not os.path.exists(ruta):
            sys.exit(f"No se encontro '{ruta}'. Ejecuta el script desde la carpeta de gem5.")

    if args.sim is None:
        sims = pedir_parametros_interactivo(N_SIMS)
    else:
        sims = args.sim
        if len(sims) != N_SIMS:
            sys.exit(f"Debes dar exactamente {N_SIMS} opciones --sim (diste {len(sims)}).")

    os.makedirs(RESULTS, exist_ok=True)

    for i, params in enumerate(sims, start=1):
        correr_simulacion(i, params)

    print("\nListo. Archivos en la carpeta:", os.path.abspath(RESULTS))


if __name__ == "__main__":
    main()