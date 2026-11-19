"""Best-fit spectrum plot and corner plot."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

C_KMS = 299792.458

# Font sizes used in all figures (larger than the matplotlib defaults, for readability)
FONT = {
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 14,
}
TRANSITION_FONTSIZE = 11
COMPONENT_COLORS = ["#d95f02", "#1b9e77", "#7570b3"]   # MAP model of each component (if > 1)


def _fit_summary_lines(cfg, map_fit, median_fit):
    """Text lines with Tex and N of every molecule for the MAP and median models."""
    lines = []
    for title, fit in [("MAP Fit:", map_fit), ("Median Fit:", median_fit)]:
        lines.append(title)
        for mol, v in zip(cfg.molecules, cfg.expand(fit)):
            lines.append(f"  {mol['label']} ($T_{{ex}}$={v['tex']:.0f} K, $N_{{tot}}$={10**v['log_ntot']:.1e})")
    return lines


def plot_spectrum(cfg, obs_list, map_fit, median_fit, y_map_mods, y_median_mods, y_components=None):
    """Best-fit spectrum plot. obs_list / y_*_mods have one entry per spectrum
    (a model list may be None if that CLASS run failed). y_components: None, or one
    model list per component (MAP parameters, that component alone)."""
    with plt.rc_context(FONT):
        if len(obs_list) > 1:
            _plot_spectra_grid(cfg, obs_list, map_fit, median_fit, y_map_mods, y_median_mods, y_components)
        else:
            _plot_single_spectrum(cfg, obs_list[0], map_fit, median_fit,
                                  y_map_mods[0] if y_map_mods is not None else None,
                                  y_median_mods[0] if y_median_mods is not None else None,
                                  [c[0] if c is not None else None for c in y_components]
                                  if y_components else None)


def _plot_single_spectrum(cfg, obs, map_fit, median_fit, y_map_mod, y_median_mod, y_comp=None):
    """One spectrum containing several lines: frequency axis."""
    x_obs, y_obs = obs.x_obs, obs.y_obs

    plt.figure(figsize=(17, 8))
    plt.step(x_obs, y_obs, where='mid', color='black', linewidth=1.2, label='Observed spectrum')

    if y_map_mod is not None:
        plt.step(x_obs, y_map_mod, where='mid', color='red', linewidth=1.5, label="MAP Model (Max Probability)")
    if y_median_mod is not None:
        plt.step(x_obs, y_median_mod, where='mid', color='blue', linestyle='--', linewidth=1.3, label="Median Model (Marginalized)")
    if y_comp:
        for k, yc in enumerate(y_comp):
            if yc is not None:
                plt.step(x_obs, yc, where='mid', color=COMPONENT_COLORS[k % 3], linewidth=1.1,
                         label=f"MAP: {cfg.molecules[k]['label']}")

    for fmin, fmax in obs.line_windows:
        plt.axvspan(fmin, fmax, color='gray', alpha=0.15, zorder=0)

    # Line Annotations (rest frequencies shifted by v_sys)
    v_sys = cfg.v_sys
    shifted_lines = sorted([(rest_freq * (1.0 - (v_sys / C_KMS)), label) for rest_freq, label in cfg.transitions])

    last_x, current_y, x_threshold = -np.inf, 0.98, 6.0
    for obs_freq, label in shifted_lines:
        plt.axvline(x=obs_freq, color='dimgray', linestyle=':', alpha=0.6, linewidth=1.0, zorder=1)
        current_y = current_y - 0.175 if (obs_freq - last_x) < x_threshold and current_y >= 0.4 else 0.98
        plt.text(obs_freq, current_y, label, rotation=90, fontsize=TRANSITION_FONTSIZE,
                 transform=plt.gca().get_xaxis_transform(), verticalalignment='top', horizontalalignment='right',
                 color='black', alpha=0.9, zorder=2)
        last_x = obs_freq

    plt.xlabel('Frequency [MHz]')
    plt.ylabel(r'$T_\mathrm{MB}$ [K]')
    plt.xlim(cfg.plot_fmin - 5.0, cfg.plot_fmax + 5.0)
    # Legend and the MAP / median values to the right of the plot, so they never hide data
    plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    box_lines = _fit_summary_lines(cfg, map_fit, median_fit)
    plt.text(1.01, 0.0, "\n".join(box_lines), transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='lightgray'))

    plt.grid(True, linestyle='--', alpha=0.25)
    plt.tight_layout()
    plt.savefig(f"{cfg.output_dir}/mcmc_best_fit_spectrum.png")
    plt.close()


def _plot_spectra_grid(cfg, obs_list, map_fit, median_fit, y_map_mods, y_median_mods, y_components=None):
    """One panel per spectrum (multi-spectrum fits).

    Panels of spectra with a rest_freq are plotted against LSR velocity
    (radio convention, v = c (1 - f / rest_freq)); others against frequency.
    """
    n = len(obs_list)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.8 * nrows), squeeze=False)

    for i, obs in enumerate(obs_list):
        ax = axes[i // ncols][i % ncols]
        spec = cfg.spectra[i]
        f0 = spec.get("rest_freq")

        # Axis conversion for this panel
        if f0:
            to_x = lambda f, f0=f0: C_KMS * (1.0 - np.asarray(f) / f0)
            xlabel = r'$v_\mathrm{LSR}$ [km s$^{-1}$]'
        else:
            to_x = lambda f: np.asarray(f, dtype=float)
            xlabel = 'Frequency [MHz]'
        x = to_x(obs.x_obs)

        ax.step(x, obs.y_obs, where='mid', color='black', linewidth=1.2, label='Observed')
        if y_map_mods is not None:
            ax.step(x, y_map_mods[i], where='mid', color='red', linewidth=1.6, label='MAP model')
        if y_median_mods is not None:
            ax.step(x, y_median_mods[i], where='mid', color='blue', linestyle='--',
                    linewidth=1.4, label='Median model')
        if y_components:
            for k, yc in enumerate(y_components):
                if yc is not None:
                    ax.step(x, yc[i], where='mid', color=COMPONENT_COLORS[k % 3], linewidth=1.1,
                            label=f"MAP: {cfg.molecules[k]['label']}")
        for fmin, fmax in obs.line_windows:
            a, b = sorted(to_x([fmin, fmax]))
            ax.axvspan(a, b, color='gray', alpha=0.15, zorder=0)

        # Frequency range shown: plot_range [MHz] from the config if given; otherwise the
        # whole spectrum, or the line windows +/- 20 MHz when the spectrum is much wider
        flo, fhi = obs.x_obs.min(), obs.x_obs.max()
        if spec.get("plot_range"):
            flo, fhi = spec["plot_range"]
        elif fhi - flo > 100.0:
            flo = max(flo, min(w[0] for w in obs.line_windows) - 20.0)
            fhi = min(fhi, max(w[1] for w in obs.line_windows) + 20.0)
        xlo, xhi = sorted(to_x([flo, fhi]))

        # Transition markers (rest frequencies shifted to v_sys)
        for rest_freq, label in cfg.transitions:
            f_obs = rest_freq * (1.0 - (cfg.v_sys / C_KMS))
            if flo <= f_obs <= fhi:
                xm = float(to_x(f_obs))
                ax.axvline(x=xm, color='dimgray', linestyle=':', alpha=0.6, linewidth=1.0, zorder=1)
                ax.text(xm, 0.97, label, rotation=90, fontsize=TRANSITION_FONTSIZE,
                        transform=ax.get_xaxis_transform(), verticalalignment='top',
                        horizontalalignment='right', alpha=0.9, zorder=2)

        ax.set_xlim(xlo, xhi)
        vis = (obs.x_obs >= flo) & (obs.x_obs <= fhi)
        curves = [obs.y_obs[vis]] + [m[i][vis] for m in (y_map_mods, y_median_mods) if m is not None]
        ymin, ymax = min(c.min() for c in curves), max(c.max() for c in curves)
        pad = 0.08 * (ymax - ymin if ymax > ymin else 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_title(obs.label)
        ax.grid(True, linestyle='--', alpha=0.25)
        if i % ncols == 0:
            ax.set_ylabel(r'$T_\mathrm{MB}$ [K]')
        ax.set_xlabel(xlabel)
        if not f0:
            ax.ticklabel_format(axis='x', useOffset=False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    # Legend in the first unused panel if there is one, otherwise in the first panel
    if n < nrows * ncols:
        spare = axes[n // ncols][n % ncols]
        spare.set_visible(True)
        spare.axis('off')
        spare.legend(*axes[0][0].get_legend_handles_labels(), loc='center')
    else:
        axes[0][0].legend(loc='best')
    fig.suptitle("   ".join(_fit_summary_lines(cfg, map_fit, median_fit)).replace(":   ", ": "))
    fig.tight_layout()
    fig.savefig(f"{cfg.output_dir}/mcmc_best_fit_spectrum.png")
    plt.close(fig)


def plot_corner(cfg, flat_samples, median_fit, map_fit):
    # Median values in blue, MAP values as dotted red lines
    with plt.rc_context(FONT):
        fig = corner.corner(
            flat_samples,
            labels=cfg.free_labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            truths=median_fit,
            truth_color='b',
            label_kwargs={"fontsize": 15},
            title_kwargs={"fontsize": 13},
        )
        for ax in fig.get_axes():
            ax.tick_params(labelsize=12)
        corner.overplot_lines(fig, map_fit, color='r', linestyle=':')
        fig.savefig(f"{cfg.output_dir}/mcmc_corner.png", bbox_inches="tight")
        plt.close()
