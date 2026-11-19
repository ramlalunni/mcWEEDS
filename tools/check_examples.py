"""Check that the example data are reproduced exactly by WEEDS at their true parameters.

For each example: the WEEDS model at the true values (examples/truth_*.yaml) is computed on
the example spectra, exactly as in a fit; the residual (data - model) is compared with the
noise that was added when the data were made (data/truth_*_spectra.npz), and the
chi-square is compared with its reference value. If GILDAS/WEEDS, the catalogue and mcWEEDS
work as intended, the residual equals the stored noise to rounding precision and the
chi-square equals the reference. The chi-square is printed per fitted channel (chi2/N),
which is close to 1 when the model matches the data up to the noise.
Plots: SINGLE_example_*/single_model_spectrum.png.

Usage (from the mcWEEDS folder):   python tools/check_examples.py
"""
import os
import subprocess
import sys

import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcweeds.config import Config, PARAMS
from mcweeds.observation import init_observation
from mcweeds.class_model import run_class_model
from mcweeds.fitting import _chi2_one_spectrum

EXAMPLES = [  # config, truth, reference chi-square at the true values
    ("examples/h2co_hc3n_single_spectrum.yaml", "examples/truth_h2co_hc3n.yaml", 96.4),
    ("examples/so_2comp_multi_spectrum.yaml", "examples/truth_so_2comp.yaml", 296.4),
]


def set_truth(cfg, truth):
    for m in cfg.molecules:
        for p in PARAMS:
            v = float(truth[m["name"]][p])
            cfg._where[(m["name"], p)] = ("const", np.log10(v) if p == "ntot" else v)
    cfg.shared = {}
    return np.array([])


def check(config, truth_file, chi2_ref):
    print(f"\n=== {config}")
    cfg = Config(config)
    with open(truth_file) as f:
        theta = set_truth(cfg, yaml.safe_load(f))
    cfg.output_dir = f"SINGLE_check_{cfg.run_name}"
    cfg.mdl_dir = os.path.join(cfg.output_dir, "temp_mdl_files")
    os.makedirs(cfg.mdl_dir, exist_ok=True)
    stored = np.load(os.path.join("data",
                                  os.path.splitext(os.path.basename(truth_file))[0] + "_spectra.npz"))
    obs_list = init_observation(cfg)
    models = run_class_model(cfg, theta)
    if models is None:
        print(f"[!] CLASS failed: see {cfg.output_dir}/failed_models.log")
        return False
    ok, chi2, n_tot = True, 0.0, 0
    for i, (obs, y) in enumerate(zip(obs_list, models)):
        resid = obs.y_obs - y
        noise = stored[f"noise_{i}"]
        dmax = np.max(np.abs(resid - noise)) if len(resid) == len(noise) else np.inf
        c = _chi2_one_spectrum(cfg, obs, y)
        n = int(obs.line_mask.sum())
        chi2 += c
        n_tot += n
        good = dmax < 1e-4
        ok &= good
        print(f"    {obs.label:40s} peak {np.max(y):6.3f} K   max|residual - stored noise| = {dmax:.1e} K "
              f"{'OK' if good else 'MISMATCH'}   chi2/N = {c / n:.2f} ({n} channels)")
    good = abs(chi2 - chi2_ref) < 0.1
    ok &= good
    print(f"    chi2/N = {chi2 / n_tot:.3f} for N = {n_tot} channels (pure noise: 1 +- {np.sqrt(2.0 / n_tot):.2f}; "
          f"reference {chi2_ref / n_tot:.3f}: {'OK' if good else 'DIFFERENT'})")
    return ok


def main():
    results = [check(*e) for e in EXAMPLES]
    print("\n[+] Plots of the models at the true values:")
    for config, truth, _ in EXAMPLES:
        subprocess.run([sys.executable, "tools/single_model.py", config, truth],
                       stdout=subprocess.DEVNULL, check=False)
        print(f"    SINGLE_{Config(config).run_name}/single_model_spectrum.png")
    print("\n" + ("ALL OK: the example data are reproduced exactly by WEEDS at the true parameters."
                  if all(results) else "[!] Some checks failed: see above."))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
