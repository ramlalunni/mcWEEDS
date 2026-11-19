"""Run one WEEDS (CLASS) LTE model for a given set of parameters."""
import os
import random
import subprocess
import time
import uuid
import numpy as np


def load_spectrum_commands(spec):
    """CLASS commands that open spec['file'] and load the selected spectrum into R.

    Used both to extract the observation and to compute every model, so that the model is
    always evaluated on the same spectrum (frequency axis) as the data. With the default
    selection the commands are:
        file in <file> / set match 0.2 / find /offset 0 0 / get f
    The file may be a GILDAS VLM cube (each pixel is a spectrum, offset (0, 0) = centre
    pixel) or a CLASS spectra file (.30m, .cla, ...). The first spectrum found is used.
    """
    sel = spec.get("select") or {"offset": [0.0, 0.0], "match": 0.2}
    cmds = [f"file in {spec['file']}"]
    find = "find"
    if sel.get("offset") is not None:
        if sel.get("match") is not None:
            cmds.append(f"set match {sel['match']:g}")
        find += f" /offset {sel['offset'][0]:g} {sel['offset'][1]:g}"
    if sel.get("number") is not None:
        find += f" /number {sel['number'][0]} {sel['number'][1]}"
    for key in ("source", "line", "telescope"):
        if sel.get(key) is not None:
            find += f" /{key} {sel[key]}"
    cmds += [find, "get f"]
    return cmds


def _species_field(name):
    """Species name as written in the WEEDS model file.

    A catalogue name that contains a space or a comma (CDMS entries such as 'SO, v=0') is
    enclosed in double quotes; without them WEEDS stops with
    "E-MODSOURCE READMDL, Incorrect input". Other names (e.g. H2CO) are written as they are.
    """
    return f'"{name}"' if (" " in name or "," in name) else name


def write_model_file(cfg, theta, mdl_name, only=None):
    """Write the WEEDS model file (.mdl) for theta: one line per component."""
    lines = ["! species Ntot Tex source_size v_off width"]
    for k, (mol, v) in enumerate(zip(cfg.molecules, cfg.expand(theta))):
        if only is not None and k not in only:
            continue
        # Enough digits that the likelihood is a smooth function of every parameter
        # (e.g. a source size rounded to 0.01" would change in steps of 4 % at 0.24").
        lines.append(f"{_species_field(mol['species'])} {10**v['log_ntot']:.6e} {v['tex']:.4f} "
                     f"{v['source_size']:.5f} {v['v_off']:.4f} {v['width']:.4f}")
    with open(mdl_name, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_model_macro(cfg, theta, mdl_name, out_names, only=None):
    """Write the WEEDS model file for theta and return the list of CLASS commands that computes it.

    The same WEEDS model file is evaluated on every spectrum in cfg.spectra:
    for each spectrum, load it, run MODSOURCE (with that spectrum's telescope
    size) and write the model channels to out_names[i].
    only: None (all components) or a list of component indices to include in the
    model file (used to plot the components separately).
    The model file is written by Python (write_model_file), so that species names with
    spaces or commas can be quoted (see _species_field).
    """
    s0 = cfg.spectra[0]
    write_model_file(cfg, theta, mdl_name, only)
    macro = load_spectrum_commands(s0) + ["set unit f", f"use in {cfg.catalog}"]
    macro += [
        f"modsource {mdl_name} {s0['telescope_size']} /verbose", "retrieve tb_model",
        f"sic output {out_names[0]}",
        "for i 1 to channels", "say 'ry[i]'", "next", "sic output",
    ]
    for spec, out_name in zip(cfg.spectra[1:], out_names[1:]):
        macro += load_spectrum_commands(spec) + [
            f"modsource {mdl_name} {spec['telescope_size']} /verbose", "retrieve tb_model",
            f"sic output {out_name}",
            "for i 1 to channels", "say 'ry[i]'", "next", "sic output",
        ]
    macro.append("exit")
    return macro


def run_class_model(cfg, theta, only=None):
    """Executes a single CLASS model using unique file names to prevent overwrites.

    Returns a list with one model spectrum (array) per entry of cfg.spectra,
    or None if CLASS failed. only: see build_model_macro.
    """
    uid = uuid.uuid4().hex
    macro_name = f"{cfg.mdl_dir}/run_{uid}.class"
    mdl_name = f"{cfg.mdl_dir}/model_{uid}.mdl"
    out_names = [f"{cfg.mdl_dir}/out_{uid}.txt"] + \
                [f"{cfg.mdl_dir}/out_{uid}_{i}.txt" for i in range(1, len(cfg.spectra))]
    failed_log = f"{cfg.output_dir}/failed_models.log"

    y_mod = None
    max_retries = 10  # retries after a CLASS start-up failure (docs/methods.md, Section 8.3)

    for attempt in range(max_retries):
        try:
            # (re)written on every attempt: the finally block below deletes the model file
            macro = build_model_macro(cfg, theta, mdl_name, out_names, only)
            with open(macro_name, "w") as f:
                f.write("\n".join(macro))

            result = subprocess.run(
                ["class", "-nw", f"@{macro_name}"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=60  # seconds per CLASS run
            )

            # Check if CLASS failed to produce all output text files
            if not all(os.path.exists(o) for o in out_names):
                raw_log = result.stdout + "\n" + result.stderr

                # CLASS start-up failure (F-MESSAGE_INIT or errno message): retry
                if "F-MESSAGE_INIT" in raw_log or "errno" in raw_log:
                    # wait until the next second, plus a random 50-250 ms
                    time_to_next_second = 1.0 - (time.time() % 1.0)
                    time.sleep(time_to_next_second + random.uniform(0.05, 0.25))
                    continue
                else:
                    # any other failure: log it
                    with open(failed_log, "a") as err_log:
                        err_log.write(f"Failed Theta: {theta}\nReason: {raw_log}\n---\n")
                    return None

            # all output files exist: read the model spectra
            y_mod = [np.atleast_1d(np.loadtxt(o)) for o in out_names]
            break  # Exit the retry loop

        except subprocess.TimeoutExpired:
            # log the timeout (the model counts as failed)
            with open(failed_log, "a") as err_log:
                err_log.write(f"Timeout Theta: {theta}\n---\n")
            return None
        except Exception as e:
            print(f"\n[!] Python Error: {e}")
            return None
        finally:
            # delete this model's files (also before a retry)
            for f in [macro_name, mdl_name] + out_names:
                if os.path.exists(f): os.remove(f)

    return y_mod
