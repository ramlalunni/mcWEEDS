"""Make synthetic observations: a WEEDS LTE model with known parameters plus correlated noise.

For every spectrum of an mcWEEDS config, the model given in a "truth" file (same format as
for tools/single_model.py: one entry per component, all five parameters) is computed by
WEEDS on that spectrum's frequency axis, Gaussian noise with a given RMS and correlation
between neighbouring channels is added, and the result is written as a CLASS spectra file.

The frequency axes come from either

  * GRID MODE: a `grid: [fmin, fmax, df]` entry (MHz) for every spectrum of the config
    (for a single spectrum: `grid:` at the top level next to spectrum_file). The spectra
    are created from scratch (CLASS MODEL), with the rest frequency of the spectrum
    (`rest_freq`, otherwise the centre of the grid) at its exact frequency, and are written
    to the `file:` paths of the config itself. So an example config can make its own data:
        python tools/make_synthetic_spectra.py examples/h2co_hc3n_single_spectrum.yaml \\
               examples/truth_h2co_hc3n.yaml
  * TEMPLATE MODE (no `grid:` entries): the existing spectra of the config are used as
    templates (frequency axis and header); their intensities are replaced, and the files
    are written to OUTDIR as <prefix><input name>.30m.

Uses: example data whose true parameters are known; checking whether a setup (lines,
windows, noise) can recover the parameters you are interested in, e.g. before observing.

Usage (from the mcWEEDS folder):
    python tools/make_synthetic_spectra.py config.yaml truth.yaml [OUTDIR]
        [--rms 0.03] [--rho 0.39] [--seed 1] [--source SYNTH] [--position RA DEC]
        [--prefix synth_] [--drop TEXT ...]

    OUTDIR    template mode only (required there)
    --rms     noise RMS [K] (default 0.03)
    --rho     correlation of neighbouring channels, 0 <= rho < 0.5 (default 0.39)
    --seed    random seed (default 1), so the data can be remade exactly
    --source  source name written in the headers (default SYNTH)
    --position RA DEC   template mode: new central position written in the headers (CLASS
              MODIFY POSITION, e.g. 06:00:00.0 +30:00:00); default: keep the template's.
    --prefix, --drop    template mode: output names = prefix + input name without the
              --drop texts (default prefix synth_)

Each written file is read back (offset 0,0) and compared with the intended spectrum. A copy
of the truth file and <truth name>_spectra.npz (noise-free models model_i, noise noise_i,
frequencies freq_i [MHz]) are written next to the spectra. Scratch files go to a temporary
synth_work_* folder, removed at the end.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcweeds.config import Config, PARAMS
from mcweeds.class_model import run_class_model, load_spectrum_commands
from mcweeds.observation import extract_spectrum


def set_model(cfg, model):
    """Fix every parameter of every component to the values in `model`."""
    for m in cfg.molecules:
        n = m["name"]
        if n not in model:
            sys.exit(f"Truth file has no entry for component {n}")
        for p in PARAMS:
            if p not in model[n]:
                sys.exit(f"Truth file, {n}: parameter '{p}' missing")
            v = float(model[n][p])
            cfg._where[(n, p)] = ("const", np.log10(v) if p == "ntot" else v)
    cfg.shared = {}
    return np.array([])


def correlated_noise(n, rms, rho, rng):
    """Gaussian noise with standard deviation rms and lag-1 correlation rho (MA(1) process)."""
    if not 0.0 <= rho < 0.5:
        sys.exit("--rho must be in [0, 0.5)")
    b = 0.0 if rho == 0 else (1.0 - np.sqrt(1.0 - 4.0 * rho**2)) / (2.0 * rho)
    e = rng.normal(size=n + 1)
    x = (e[1:] + b * e[:-1]) / np.sqrt(1.0 + b**2)   # unit variance, lag-1 correlation rho
    return rms * x


def run_class(lines, path, what):
    with open(path, "w") as f:
        f.write("\n".join(lines))
    try:
        res = subprocess.run(["class", "-nw", f"@{path}"], capture_output=True, text=True,
                             stdin=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired:
        sys.exit(f"CLASS did not finish {what} (macro: {path})")
    return res


def make_grid_template(grid, rest_freq, out_path, source, workdir):
    """Create an empty (zero) CLASS spectrum on a regular frequency grid (CLASS MODEL)."""
    fmin, fmax, df = (float(g) for g in grid)
    if not (fmax > fmin and df > 0):
        sys.exit(f"grid {grid}: need fmin < fmax and df > 0 (MHz)")
    n = int(round((fmax - fmin) / df)) + 1
    f0 = float(rest_freq) if rest_freq else 0.5 * (fmin + fmax)
    ref = 1.0 + (f0 - fmin) / df          # channel (1-based, may be fractional) at f0
    macro = [
        f"define double synth[{n}]",
        "let synth 0",
        f"model synth /xaxis {ref:.6f} {f0:.6f} {df:.9f} FREQUENCY /frequency SYNTH {f0:.6f}",
        f"modify source {source}",
        f"file out {out_path} single /overwrite",
        "write",
        "exit",
    ]
    res = run_class(macro, os.path.join(workdir, "make_grid.class"), f"creating {out_path}")
    if not os.path.exists(out_path):
        sys.exit(f"CLASS did not create {out_path}. Output:\n{res.stdout[-3000:]}\n{res.stderr[-1000:]}")
    return n


def write_class_file(spec, values, out_path, source, workdir, position=None):
    """Load the template spectrum in CLASS, replace its intensities and source name, write it."""
    txt = os.path.join(workdir, "synth_values.txt")
    np.savetxt(txt, values, fmt="%.7e")
    macro = load_spectrum_commands(spec) + [
        f"define double synth[{len(values)}]",
        f"accept synth /column {txt}",
        "let ry synth",
        f"modify source {source}",
    ] + ([f"modify position {position[0]} {position[1]}"] if position else []) + [
        f"file out {out_path} single /overwrite",
        "write",
        "exit",
    ]
    res = run_class(macro, os.path.join(workdir, "write_synth.class"), f"writing {out_path}")
    if not os.path.exists(out_path):
        sys.exit(f"CLASS did not write {out_path}. Output:\n{res.stdout[-3000:]}\n{res.stderr[-1000:]}")


def grid_entries(raw):
    """(entry dict, is_top_level) for every spectrum of the raw YAML config."""
    if "spectra" in raw:
        return [(s, False) for s in raw["spectra"]]
    return [(raw, True)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config"); ap.add_argument("truth"); ap.add_argument("outdir", nargs="?")
    ap.add_argument("--rms", type=float, default=0.03)
    ap.add_argument("--rho", type=float, default=0.39)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--source", default="SYNTH")
    ap.add_argument("--position", nargs=2, metavar=("RA", "DEC"), default=None)
    ap.add_argument("--prefix", default="synth_")
    ap.add_argument("--drop", action="append", default=[])
    a = ap.parse_args()

    with open(a.config) as f:
        raw = yaml.safe_load(f)
    with open(a.truth) as f:
        truth = yaml.safe_load(f)
    entries = grid_entries(raw)
    n_grid = sum(1 for e, _ in entries if e.get("grid"))
    if 0 < n_grid < len(entries):
        sys.exit("Give `grid:` for all spectra of the config (grid mode) or for none (template mode)")
    grid_mode = n_grid > 0
    if not grid_mode and not a.outdir:
        sys.exit("Template mode (no `grid:` in the config): give OUTDIR")

    # scratch files (CLASS macros, templates, extracted spectra): a temporary folder
    work = os.path.relpath(tempfile.mkdtemp(prefix="synth_work_", dir="."))
    config_path = a.config
    outputs = []
    if grid_mode:
        # 1. empty spectra on the grids (templates), 2. a copy of the config that reads them
        for k, (e, top) in enumerate(entries):
            fkey = "spectrum_file" if top else "file"
            out = e[fkey]
            if " " in out:
                sys.exit(f"{out}: no spaces in file names (CLASS)")
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            tpl = os.path.join(work, f"template_{k}.30m")
            n = make_grid_template(e["grid"], e.get("rest_freq"), tpl, a.source, work)
            print(f"    grid {e['grid']} MHz -> {n} channels (template for {out})")
            outputs.append(out)
            e[fkey] = tpl
            e.pop("select", None)              # templates: one spectrum at offset (0, 0)
        raw.pop("select", None)
        config_path = os.path.join(work, "config_templates.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(raw, f, sort_keys=False)
        outdir = os.path.dirname(outputs[0]) or "."
    else:
        outdir = a.outdir
        os.makedirs(outdir, exist_ok=True)

    cfg = Config(config_path)
    theta = set_model(cfg, truth)
    cfg.output_dir = work
    cfg.mdl_dir = os.path.join(work, "temp_mdl_files")
    os.makedirs(cfg.mdl_dir, exist_ok=True)

    print(f"[+] WEEDS model of {a.truth} on the {len(cfg.spectra)} spectra of {a.config}")
    models = run_class_model(cfg, theta)
    if models is None:
        sys.exit(f"[!] CLASS failed; see {work}/failed_models.log")

    rng = np.random.default_rng(a.seed)
    saved = {}
    for i, (spec, y) in enumerate(zip(cfg.spectra, models)):
        if grid_mode:
            out = outputs[i]
        else:
            stem = os.path.splitext(os.path.basename(spec["file"]))[0]
            for d in a.drop:
                stem = stem.replace(d, "")
            out = os.path.join(outdir, f"{a.prefix}{stem}.30m")
        noise = correlated_noise(len(y), a.rms, a.rho, rng)
        values = y + noise
        write_class_file(spec, values, out, a.source, work, None if grid_mode else a.position)

        # read back with the default selection (offset 0,0) and compare
        check = dict(spec, file=out, select={"offset": [0.0, 0.0], "match": 0.2})
        x_chk, y_chk, _ = extract_spectrum(cfg, check, f"_check{i}")
        x_tpl, _, df = extract_spectrum(cfg, spec, f"_tpl{i}")
        dy = np.max(np.abs(y_chk - values)) if len(y_chk) == len(values) else np.inf
        dx = np.max(np.abs(x_chk - x_tpl)) if len(x_chk) == len(x_tpl) else np.inf
        # intensities: float32 rounding; frequencies: printed header values, so allow 1 % of a channel
        status = "OK" if (dy < 1e-4 + 1e-5 * np.max(np.abs(values)) and dx < 0.01 * df) else "MISMATCH"
        print(f"    {out}: {len(values)} channels {x_chk.min():.3f}-{x_chk.max():.3f} MHz, "
              f"peak model {np.max(y):.3f} K, noise rms {np.std(noise):.4f} K  "
              f"[read back: {status}, max |dT| {dy:.2e} K, max |dnu| {dx:.2e} MHz = {dx / df:.1e} channel]")
        if status != "OK":
            sys.exit("[!] The written file does not contain the intended spectrum")
        saved[f"model_{i}"] = y
        saved[f"noise_{i}"] = noise
        saved[f"freq_{i}"] = x_chk

    tstem = os.path.splitext(os.path.basename(a.truth))[0]
    np.savez(os.path.join(outdir, f"{tstem}_spectra.npz"), **saved)
    shutil.copy(a.truth, os.path.join(outdir, f"{tstem}.yaml"))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[+] Done: {len(cfg.spectra)} files in {outdir} (truth: {outdir}/{tstem}.yaml; "
          f"rms {a.rms} K, rho {a.rho}, seed {a.seed})")


if __name__ == "__main__":
    main()
