"""Compute and plot ONE WEEDS model (no MCMC) for the spectra of an mcWEEDS config.

The spectra, windows, catalogue, telescope size and chi-square method come from the
mcWEEDS config; the model parameters come from a small YAML file with one entry per
molecule/component of the config (all five parameters, linear N):

    SO_c: {ntot: 5.0e15, tex: 100.0, source_size: 0.5, v_off: 12.5, width: 3.5}
    SO_e: {ntot: 3.0e14, tex: 40.0,  source_size: 4.0, v_off: 10.0, width: 7.0}

These values replace the priors / fixed / shared settings of the config.

Output (in SINGLE_<run_name>/): single_model_spectrum.png with the same panel layout as
the mcWEEDS best-fit plot (observed, total model, each component on its own), and
single_model_spectra.npz with the model spectra, and weeds_verbose.log with the
WEEDS MODSOURCE /VERBOSE output (line-centre opacity of every line and component). The chi-square of every spectrum
(same definition as the MCMC likelihood) is printed and shown in the panel titles.

Usage (from the mcWEEDS folder):
    python tools/single_model.py examples/so_2comp_multi_spectrum.yaml examples/model_so_2comp.yaml [output.png]
"""
import os
import sys

import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mcweeds.config import Config, PARAMS
from mcweeds.observation import init_observation
from mcweeds.class_model import run_class_model, build_model_macro
from mcweeds.fitting import _chi2_one_spectrum
from mcweeds.plotting import FONT, TRANSITION_FONTSIZE, COMPONENT_COLORS, C_KMS

# ---- plot options (the chi-square is always printed in the terminal)
SHOW_CHI2 = False        # True: chi-square in the panel titles and the figure title
DROP_EDGE_CHANNELS = 1   # channels not plotted at each end of every spectrum (edge channels of the
                         # cut-outs drop sharply); 0 plots all channels. Plot only: chi2 is unchanged.


def set_model(cfg, model):
    """Fix every parameter of every molecule to the values in `model` (dict per molecule name)."""
    names = [m["name"] for m in cfg.molecules]
    missing = [n for n in names if n not in model]
    if missing:
        sys.exit(f"Model file has no entry for: {', '.join(missing)} (config molecules: {', '.join(names)})")
    for n in names:
        for p in PARAMS:
            if p not in model[n]:
                sys.exit(f"Model file, {n}: parameter '{p}' missing")
            v = float(model[n][p])
            cfg._where[(n, p)] = ("const", np.log10(v) if p == "ntot" else v)
    cfg.shared = {}
    return np.array([])          # no free parameters left


def model_text(cfg):
    out = []
    for m, v in zip(cfg.molecules, cfg.expand(np.array([]))):
        out.append(f"{m['label']}: N = {10**v['log_ntot']:.2e} cm$^{{-2}}$, $T_{{ex}}$ = {v['tex']:.0f} K, "
                   rf"$\theta_s$ = {v['source_size']:.2f}$''$, $v$ = {v['v_off']:.2f}, "
                   rf"$\Delta V$ = {v['width']:.2f} km s$^{{-1}}$")
    return "\n".join(out)


def plot(cfg, obs_list, y_tot, y_comp, chi2, out_png):
    n = len(obs_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    with plt.rc_context(FONT):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows + 0.6), squeeze=False)
        for i, obs in enumerate(obs_list):
            ax = axes[i // ncols][i % ncols]
            spec = cfg.spectra[i]
            f0 = spec.get("rest_freq")
            if f0:
                to_x = lambda f, f0=f0: C_KMS * (1.0 - np.asarray(f) / f0)
                xlabel = r'$v_\mathrm{LSR}$ [km s$^{-1}$]'
            else:
                to_x = lambda f: np.asarray(f, dtype=float)
                xlabel = 'Frequency [MHz]'
            e = DROP_EDGE_CHANNELS
            keep = np.zeros(len(obs.x_obs), bool)
            keep[e:len(obs.x_obs) - e] = True
            x = to_x(obs.x_obs)
            ax.step(x[keep], obs.y_obs[keep], where='mid', color='black', linewidth=1.2, label='Observed')
            if y_tot is not None:
                ax.step(x[keep], y_tot[i][keep], where='mid', color='red', linewidth=1.6, label='Model')
            if y_comp and len(y_comp) > 1:
                for k, yc in enumerate(y_comp):
                    if yc is not None:
                        ax.step(x[keep], yc[i][keep], where='mid', color=COMPONENT_COLORS[k % 3],
                                linewidth=1.1, label=cfg.molecules[k]['label'])
            for fmin, fmax in obs.line_windows:
                a, b = sorted(to_x([fmin, fmax]))
                ax.axvspan(a, b, color='gray', alpha=0.15, zorder=0)

            flo, fhi = obs.x_obs[keep].min(), obs.x_obs[keep].max()
            if spec.get("plot_range"):
                flo, fhi = spec["plot_range"]
            elif fhi - flo > 100.0:
                flo = max(flo, min(w[0] for w in obs.line_windows) - 20.0)
                fhi = min(fhi, max(w[1] for w in obs.line_windows) + 20.0)
            xlo, xhi = sorted(to_x([flo, fhi]))
            for rest_freq, label in cfg.transitions:
                f_obs = rest_freq * (1.0 - (cfg.v_sys / C_KMS))
                if flo <= f_obs <= fhi:
                    xm = float(to_x(f_obs))
                    ax.axvline(x=xm, color='dimgray', linestyle=':', alpha=0.6, linewidth=1.0, zorder=1)
                    ax.text(xm, 0.97, label, rotation=90, fontsize=TRANSITION_FONTSIZE,
                            transform=ax.get_xaxis_transform(), verticalalignment='top',
                            horizontalalignment='right', alpha=0.9, zorder=2)
            ax.set_xlim(xlo, xhi)
            vis = (obs.x_obs >= flo) & (obs.x_obs <= fhi) & keep
            curves = [obs.y_obs[vis]] + ([y_tot[i][vis]] if y_tot is not None else [])
            ymin, ymax = min(c.min() for c in curves), max(c.max() for c in curves)
            pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)
            nfit = int(obs.line_mask.sum())
            ttl = obs.label if (chi2[i] is None or not SHOW_CHI2) else \
                f"{obs.label}\n$\\chi^2$ = {chi2[i]:.0f} ({nfit} channels)"
            ax.set_title(ttl, fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.25)
            if i % ncols == 0:
                ax.set_ylabel(r'$T_\mathrm{MB}$ [K]')
            ax.set_xlabel(xlabel)
            if not f0:
                ax.ticklabel_format(axis='x', useOffset=False)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        axes[0][0].legend(loc='best', fontsize=10)
        tot = sum(c for c in chi2 if c is not None)
        fig.suptitle(model_text(cfg) + (f"\ntotal $\\chi^2$ = {tot:.0f} ({cfg.chi_sq_method})" if SHOW_CHI2 else ""),
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
    print(f"[+] Plot: {out_png}")


def save_weeds_verbose(cfg, theta, log_path):
    """Run the model once more and keep the CLASS terminal output.

    MODSOURCE /VERBOSE prints, per component and per spectrum, the partition function and,
    for each line in the band, frequency, E_up, g_up, A_ul and the opacity at line centre.
    mcWEEDS itself discards this output; here it is saved to log_path.
    """
    import subprocess
    import uuid
    uid = uuid.uuid4().hex
    mdl = f"{cfg.mdl_dir}/model_{uid}.mdl"
    outs = [f"{cfg.mdl_dir}/out_{uid}_{i}.txt" for i in range(len(cfg.spectra))]
    macro_path = f"{cfg.mdl_dir}/run_{uid}.class"
    with open(macro_path, "w") as f:
        f.write("\n".join(build_model_macro(cfg, theta, mdl, outs)))
    res = subprocess.run(["class", "-nw", f"@{macro_path}"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL, timeout=120)
    with open(log_path, "w") as f:
        f.write("# WEEDS MODSOURCE /VERBOSE output (one block per spectrum)\n")
        f.write("# spectra in order: " + " | ".join(s["label"] for s in cfg.spectra) + "\n\n")
        f.write(res.stdout)
        if res.stderr.strip():
            f.write("\n# stderr\n" + res.stderr)
    for f_ in [macro_path, mdl] + outs:
        if os.path.exists(f_):
            os.remove(f_)
    print(f"[+] WEEDS line list with opacities (MODSOURCE /VERBOSE): {log_path}")


def main(config_path, model_path, out_png=None):
    cfg = Config(config_path)
    cfg.output_dir = f"SINGLE_{cfg.run_name}"
    cfg.mdl_dir = os.path.join(cfg.output_dir, "temp_mdl_files")
    os.makedirs(cfg.mdl_dir, exist_ok=True)
    with open(model_path) as f:
        model = yaml.safe_load(f)
    theta = set_model(cfg, model)
    print("[+] Model:\n    " + model_text(cfg).replace("$", "").replace("\n", "\n    "))

    obs_list = init_observation(cfg)
    y_tot = run_class_model(cfg, theta)
    if y_tot is None:
        sys.exit(f"[!] CLASS failed; see {cfg.output_dir}/failed_models.log")
    y_comp = [run_class_model(cfg, theta, only=[k]) for k in range(len(cfg.molecules))] \
        if len(cfg.molecules) > 1 else None

    chi2 = []
    print("[+] Chi-square per spectrum (line windows only):")
    for obs, y in zip(obs_list, y_tot):
        c = _chi2_one_spectrum(cfg, obs, y) if len(y) == len(obs.y_obs) else None
        chi2.append(c)
        n = int(obs.line_mask.sum())
        print(f"    {obs.label:55s} chi2 = {c:9.1f}  ({n} channels)" if c is not None else
              f"    {obs.label:55s} model length mismatch")
    print(f"    total chi2 = {sum(c for c in chi2 if c is not None):.1f}")

    np.savez(os.path.join(cfg.output_dir, "single_model_spectra.npz"),
             **{f"obs_x_{i}": o.x_obs for i, o in enumerate(obs_list)},
             **{f"obs_y_{i}": o.y_obs for i, o in enumerate(obs_list)},
             **{f"model_{i}": y for i, y in enumerate(y_tot)},
             **({f"model_{i}_{cfg.molecules[k]['name']}": yc[i]
                 for k, yc in enumerate(y_comp) if yc is not None for i in range(len(obs_list))}
                if y_comp else {}))
    save_weeds_verbose(cfg, theta, os.path.join(cfg.output_dir, "weeds_verbose.log"))
    plot(cfg, obs_list, y_tot, y_comp, chi2,
         out_png or os.path.join(cfg.output_dir, "single_model_spectrum.png"))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: python tools/single_model.py config.yaml model.yaml [output.png]")
    main(*sys.argv[1:])
