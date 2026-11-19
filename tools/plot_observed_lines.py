"""Plot the observed spectra of a (multi-spectrum) mcWEEDS config, without models.

Same panel layout and fonts as the mcWEEDS best-fit plot, but:
  - bottom x axis in frequency [MHz], top x axis in LSR velocity [km/s]
    (radio convention, relative to each spectrum's rest_freq),
  - a marker with label at every line (each spectrum's own rest_freq, shifted to
    v_sys, plus any entry of `transitions:` falling in the panel).

Only the spectra part of the config is needed: `molecules:` and the line / noise windows
may still be missing (windows are shaded if given). So this can be run on a new source
before the model part of the config is written; a minimal config is

    spectra:
      - file: data/my_line.vlm
        rest_freq: 345903.916        # optional: velocity axis on top
        # select: {offset: [0, 0]}   # which spectrum of the file (default)
    plot: {v_sys: -9.5}              # optional: shifts the transition markers
    transitions: [[345903.916, 'CH$_3$OH']]   # optional

Usage (from the mcWEEDS folder):
    python tools/plot_observed_lines.py examples/so_2comp_multi_spectrum.yaml [output.png] [--no-windows]
"""
import os
import sys

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mcweeds.config import Config
from mcweeds.observation import extract_spectrum
from mcweeds.plotting import FONT, TRANSITION_FONTSIZE, C_KMS


class _Spectrum:
    def __init__(self, x, y, line_windows):
        self.x_obs, self.y_obs, self.line_windows = x, y, line_windows


def main(config_path, out_png=None, show_windows=True):
    # Data only: `molecules:` and the windows may still be missing (no fit, no noise model)
    cfg = Config(config_path, data_only=True)
    os.makedirs(cfg.output_dir, exist_ok=True)       # the extraction writes its files here
    print("[+] Extracting the observed spectra...")
    obs_list = []
    multi = len(cfg.spectra) > 1
    for i, spec in enumerate(cfg.spectra):
        x, y, df = extract_spectrum(cfg, spec, f"_{i}" if multi else "")
        print(f"    {spec['file']}: {x.min():.3f}-{x.max():.3f} MHz, {len(y)} channels of {df:.4f} MHz")
        obs_list.append(_Spectrum(x, y, spec["line_windows"]))
    out_png = out_png or os.path.join(cfg.output_dir, "observed_lines.png")

    n = len(obs_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    with plt.rc_context(FONT):
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.2 * nrows), squeeze=False)

        for i, (obs, spec) in enumerate(zip(obs_list, cfg.spectra)):
            ax = axes[i // ncols][i % ncols]
            f0 = spec.get("rest_freq")

            ax.step(obs.x_obs, obs.y_obs, where='mid', color='black', linewidth=1.2)
            if show_windows:
                for fmin, fmax in obs.line_windows:
                    ax.axvspan(fmin, fmax, color='gray', alpha=0.15, zorder=0)

            # Frequency range: plot_range if given; whole spectrum, or line windows
            # +/- 20 MHz when the spectrum is much wider than one line
            flo, fhi = obs.x_obs.min(), obs.x_obs.max()
            if spec.get("plot_range"):
                flo, fhi = spec["plot_range"]
            elif fhi - flo > 100.0 and obs.line_windows:
                flo = max(flo, min(w[0] for w in obs.line_windows) - 20.0)
                fhi = min(fhi, max(w[1] for w in obs.line_windows) + 20.0)
            ax.set_xlim(flo, fhi)

            # Line markers: this spectrum's own line, plus `transitions` in range
            markers = []
            if f0:
                own = [lab for rf, lab in cfg.transitions if abs(rf - f0) < 0.01]
                fallback = cfg.molecules[0]["label"] if cfg.molecules else f"{f0:.3f} MHz"
                markers.append((f0, own[0] if own else fallback))
            for rf, lab in cfg.transitions:
                if not (f0 and abs(rf - f0) < 0.01):
                    markers.append((rf, lab))
            for rf, lab in markers:
                f_obs = rf * (1.0 - cfg.v_sys / C_KMS)
                if flo <= f_obs <= fhi:
                    ax.axvline(f_obs, color='dimgray', linestyle=':', alpha=0.7, linewidth=1.0, zorder=1)
                    ax.text(f_obs, 0.97, lab, rotation=90, fontsize=TRANSITION_FONTSIZE,
                            transform=ax.get_xaxis_transform(), va='top', ha='right',
                            alpha=0.9, zorder=2)

            vis = (obs.x_obs >= flo) & (obs.x_obs <= fhi)
            ymin, ymax = obs.y_obs[vis].min(), obs.y_obs[vis].max()
            pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)

            ax.xaxis.set_major_locator(MaxNLocator(5))
            ax.ticklabel_format(axis='x', useOffset=False)
            ax.set_xlabel('Frequency [MHz]')
            if i % ncols == 0:
                ax.set_ylabel(r'$T_\mathrm{MB}$ [K]')
            ax.grid(True, linestyle='--', alpha=0.25)

            # Top axis: LSR velocity relative to this spectrum's rest frequency
            if f0:
                sec = ax.secondary_xaxis('top', functions=(
                    lambda f, f0=f0: C_KMS * (1.0 - np.asarray(f) / f0),
                    lambda v, f0=f0: f0 * (1.0 - np.asarray(v) / C_KMS)))
                sec.set_xlabel(r'$v_\mathrm{LSR}$ [km s$^{-1}$]')
                ax.set_title(spec["label"], pad=10)
                ax.title.set_position((0.5, 1.22))   # title above the velocity axis
            else:
                ax.set_title(spec["label"])

        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    print(f"[+] Saved {out_png}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit("Usage: python tools/plot_observed_lines.py config.yaml [output.png] [--no-windows]")
    main(args[0], args[1] if len(args) > 1 else None, show_windows="--no-windows" not in sys.argv)
