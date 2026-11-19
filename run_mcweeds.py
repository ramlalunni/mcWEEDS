"""Run an mcWEEDS fit.

Usage:
    python run_mcweeds.py config.yaml
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"  # one thread per worker; set before numpy is imported
import shutil
import sys

from mcweeds import (__version__, Config, init_observation, run_class_model, run_mcmc,
                     summarize, save_chains, write_summary, save_best_fit, plot_spectrum,
                     plot_corner)


def main(config_path):
    print(f"[+] mcWEEDS {__version__}: {config_path}")
    cfg = Config(config_path)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.mdl_dir, exist_ok=True)
    # Keep a copy of the configuration with the results
    shutil.copy(config_path, os.path.join(cfg.output_dir, "config_used.yaml"))

    obs = init_observation(cfg)
    sampler = run_mcmc(cfg, obs)
    res = summarize(cfg, sampler)
    save_chains(cfg, sampler, res)
    write_summary(cfg, sampler, res, config_path)

    print("[+] Synthesizing Maximum A Posteriori (MAP) and Marginalized Median models...")
    y_map_mod = run_class_model(cfg, res["map_fit"])
    y_median_mod = run_class_model(cfg, res["median_fit"])

    # MAP model of each component on its own (only when there are several components)
    y_components = None
    if len(cfg.molecules) > 1:
        y_components = [run_class_model(cfg, res["map_fit"], only=[k]) for k in range(len(cfg.molecules))]

    save_best_fit(cfg, obs, res, y_map_mod, y_median_mod, y_components)
    plot_spectrum(cfg, obs, res["map_fit"], res["median_fit"], y_map_mod, y_median_mod, y_components)
    plot_corner(cfg, res["flat_samples"], res["median_fit"], res["map_fit"])
    print(f"[+] Complete! Check '{cfg.output_dir}/' for the corner plot, spectrum and chains.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python run_mcweeds.py config.yaml")
    main(sys.argv[1])
