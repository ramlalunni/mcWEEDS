"""Check that a local catalogue copy gives the same WEEDS models as the online catalogue.

Computes a few models with random parameters inside the priors of an mcWEEDS config, once
with the online catalogue (e.g. `cdms`) and once with the local file of the config
(`catalog:`), and prints the largest difference and the time per model.

Usage (from the mcWEEDS folder):
    python tools/check_catalog.py config.yaml [online_catalog] [n_models]
    e.g. python tools/check_catalog.py examples/h2co_hc3n_single_spectrum.yaml cdms 3
"""
import os
import sys
import time

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcweeds import Config
from mcweeds.class_model import run_class_model


def main(config_path, online="cdms", n_models=3):
    cfg = Config(config_path)
    local = cfg.catalog
    if local == online:
        sys.exit(f"The config already uses the online catalogue '{online}'; set catalog: to the local file")
    os.makedirs(cfg.mdl_dir, exist_ok=True)
    rng = np.random.default_rng(1)
    for k in range(int(n_models)):
        theta = np.array([rng.uniform(lo, hi) for lo, hi in cfg.bounds])
        cfg.catalog = online
        t0 = time.time(); y_on = run_class_model(cfg, theta); t_on = time.time() - t0
        cfg.catalog = local
        t0 = time.time(); y_loc = run_class_model(cfg, theta); t_loc = time.time() - t0
        if y_on is None or y_loc is None:
            print(f"set {k}: a CLASS run failed (see {cfg.output_dir}/failed_models.log)")
            continue
        diff = max(np.max(np.abs(a - b)) for a, b in zip(y_on, y_loc))
        peak = max(np.max(np.abs(a)) for a in y_on)
        print(f"set {k}: max |diff| = {diff:.3g} K (peak {peak:.3g} K) | "
              f"online {t_on:.2f} s, local {t_loc:.2f} s")


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 4:
        sys.exit("Usage: python tools/check_catalog.py config.yaml [online_catalog] [n_models]")
    main(*sys.argv[1:])
