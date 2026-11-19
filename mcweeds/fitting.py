"""Prior, likelihood, the emcee run and the post-run summary."""
import time
from multiprocessing import Pool
import numpy as np
import scipy.linalg
import emcee

from .class_model import run_class_model
from .config import display_name

# Config and Observation used by log_probability. They are set once per process
# (main process and every Pool worker) by init_worker().
_CFG = None
_OBS = None


def init_worker(cfg, obs):
    global _CFG, _OBS
    _CFG, _OBS = cfg, obs


# --- Log-likelihood functions ---
def log_prior(theta):
    """Uniform prior: checks if parameters are within defined bounds."""
    for val, bounds in zip(theta, _CFG.bounds):
        if not (bounds[0] <= val <= bounds[1]):
            return -np.inf
    return 0.0


def _chi2_one_spectrum(cfg, obs, y_mod):
    """Chi-square of one spectrum, using the selected chi_sq_method."""
    chi2 = 0.0

    if cfg.chi_sq_method == 'whitened':
        # Cholesky decomposition method for correlated noise
        for w_mask, L in obs.window_whiteners:
            r = obs.y_obs[w_mask] - y_mod[w_mask]
            z = scipy.linalg.solve_triangular(L, r, lower=True)
            chi2 += np.sum(z**2)

    elif cfg.chi_sq_method == 'standard':
        # Standard channel-by-channel comparison assuming independent channels
        r = obs.y_obs[obs.line_mask] - y_mod[obs.line_mask]
        chi2 = np.sum((r / obs.rms_estimate)**2)

    elif cfg.chi_sq_method == 'integrated':
        # Area-based (integrated line flux) comparison per window
        for fmin, fmax in obs.line_windows:
            w_mask = (obs.x_obs > fmin) & (obs.x_obs < fmax)
            n_ch = np.sum(w_mask)
            if n_ch == 0: continue

            # Integrated area = sum of flux * channel width
            obs_area = np.sum(obs.y_obs[w_mask]) * obs.df_mhz
            mod_area = np.sum(y_mod[w_mask]) * obs.df_mhz

            # Error on the integrated area, including the channel correlation:
            # var(sum of n channels) = rms^2 * sum_ij rho(|i-j|) = rms^2 * (n + 2 sum_k (n-k) rho(k))
            k = np.arange(1, min(n_ch, len(obs.acf)))
            n_eff = n_ch + 2.0 * np.sum((n_ch - k) * obs.acf[k])
            area_err = obs.rms_estimate * np.sqrt(n_eff) * obs.df_mhz

            chi2 += ((obs_area - mod_area) / area_err)**2

    return chi2


def log_likelihood(theta):
    """Log-likelihood: -chi2/2, with chi2 summed over all spectra."""
    cfg, obs_list = _CFG, _OBS
    y_mods = run_class_model(cfg, theta)

    # Guard against CLASS failures AND truncated array lengths
    if y_mods is None or len(y_mods) != len(obs_list):
        return -np.inf
    if any(len(y) != len(o.y_obs) for y, o in zip(y_mods, obs_list)):
        return -np.inf

    chi2 = 0.0
    for obs, y_mod in zip(obs_list, y_mods):
        chi2 += _chi2_one_spectrum(cfg, obs, y_mod)

    return -0.5 * chi2


def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)


# --- MCMC run ---
def initial_positions(cfg):
    """Walker start positions: uniform over the priors (default), or a ball around
    cfg.init_model (half-width init_scatter x prior range, clipped to the priors)."""
    pos = np.zeros((cfg.n_walkers, cfg.ndim))
    if not cfg.init_model:
        for i, bounds in enumerate(cfg.bounds):
            pos[:, i] = np.random.uniform(bounds[0], bounds[1], cfg.n_walkers)
        return pos
    import yaml
    with open(cfg.init_model) as f:
        model = yaml.safe_load(f)
    print(f"[+] Walkers start around {cfg.init_model} (scatter {cfg.init_scatter:g} x prior range)")
    for i, ((mol, p), (lo, hi)) in enumerate(zip(cfg.free_params, cfg.bounds)):
        if mol not in model or p not in model[mol]:
            raise ValueError(f"init_model: no value for {mol} {p}")
        c = np.log10(float(model[mol][p])) if p == "ntot" else float(model[mol][p])
        if not lo <= c <= hi:
            raise ValueError(f"init_model: {mol} {p} = {model[mol][p]} is outside the prior")
        half = cfg.init_scatter * (hi - lo)
        a, b = max(lo, c - half), min(hi, c + half)
        pos[:, i] = np.random.uniform(a, b, cfg.n_walkers)
        print(f"    {cfg.free_keys[i]:22s} start {a:.3g} - {b:.3g}")
    return pos


def run_mcmc(cfg, obs):
    print(f"[+] Starting MCMC sampling...")
    print(f"[+] Free parameters ({cfg.ndim}): {', '.join(cfg.free_keys)}")
    print(f"[+] Configuration: {cfg.n_walkers} walkers, {cfg.n_steps} steps "
          f"({cfg.n_walkers*cfg.n_steps} total models)")

    pos = initial_positions(cfg)

    start_time = time.time()
    with Pool(cfg.n_threads, initializer=init_worker, initargs=(cfg, obs)) as pool:
        sampler = emcee.EnsembleSampler(cfg.n_walkers, cfg.ndim, log_probability, pool=pool)
        for i, result in enumerate(sampler.sample(pos, iterations=cfg.n_steps)):
            if (i + 1) % 10 == 0:
                print(f"    Progress: {i + 1}/{cfg.n_steps} steps completed...")

    print(f"[+] MCMC Finished in {(time.time() - start_time)/60:.1f} minutes.")
    return sampler


# --- Post-processing ---
def summarize(cfg, sampler):
    """Convergence diagnostics, burn-in/thinning, median ± 1 sigma and MAP.

    Returns a dict with flat_samples, flat_log_prob, median_fit, map_fit, burnin, thin.
    """
    print("\n[+] Calculating convergence diagnostics...")
    n_steps = cfg.n_steps

    # Calculate Autocorrelation Time dynamically with safety clamps
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        max_tau = np.max(tau)

        # Explicitly check if emcee returned NaN due to a very short or stuck chain
        if np.isnan(max_tau):
            raise ValueError("tau is NaN")

        tau_ratio = n_steps / max_tau
        print(f"    Max Autocorrelation Time (tau): {max_tau:.1f} steps")
        print(f"    Chain length to tau ratio: {tau_ratio:.1f}")

        if tau_ratio < 50:
            print("    [!] Warning: Chain is too short to fully trust tau (ratio < 50).")

        # Clamp burn-in to a maximum of 50% of the chain to prevent empty arrays
        burnin = min(int(2 * max_tau), int(0.5 * n_steps))
        thin = max(1, int(0.5 * max_tau))  # Ensure thin is at least 1
        print(f"    Applying dynamic burn-in: {burnin} steps, thinning: {thin}")

    except (emcee.autocorr.AutocorrError, ValueError):
        print("    [!] Chain too short to calculate tau accurately. Falling back to 30% burn-in.")
        burnin = int(0.3 * n_steps)
        thin = 1

    # Check Acceptance Fraction
    mean_acceptance = np.mean(sampler.acceptance_fraction)
    print(f"    Mean Acceptance Fraction: {mean_acceptance:.3f}")
    if mean_acceptance < 0.2 or mean_acceptance > 0.5:
        print("    [!] Warning: Acceptance fraction is outside the optimal range (0.2 - 0.5).")

    # Extract the thinned, flat chains and their corresponding likelihood probabilities
    flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burnin, thin=thin, flat=True)

    eff_samples = len(flat_samples)
    print(f"    Effective independent samples: {eff_samples}")
    if eff_samples < 100:
        print("    [!] Warning: Effective sample size is critically low for reliable 1-sigma bounds.")

    # Median and 1-Sigma bounds for formal reporting
    percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
    median_fit = percentiles[1]
    lower_err = median_fit - percentiles[0]
    upper_err = percentiles[2] - median_fit

    print("\n" + "="*65)
    print(" BEST FIT PARAMETERS (Marginal Median ± 1-Sigma) ")
    print("="*65)
    for i, key in enumerate(cfg.free_keys):
        val, err_minus, err_plus = median_fit[i], lower_err[i], upper_err[i]
        label, unit = display_name(key)
        if key.endswith("_log_ntot"):
            # Fitted as log10(ntot): convert back to linear, with asymmetric errors
            val_lin = 10**val
            err_plus_lin = 10**(val + err_plus) - val_lin
            err_minus_lin = val_lin - 10**(val - err_minus)
            print(f" {label:17s}: {val_lin:.3e} (+{err_plus_lin:.2e} / -{err_minus_lin:.2e}) {unit}")
        else:
            print(f" {label:17s}: {val:.4g} (+{err_plus:.2g} / -{err_minus:.2g}) {unit}")
    for line in cfg.describe_non_free():
        print(line)
    print("="*65 + "\n")

    # Maximum A Posteriori (MAP) sample
    if not np.any(np.isfinite(flat_log_prob)):
        raise RuntimeError("All posterior samples are -inf. Check failed_models.log!")
    map_fit = flat_samples[np.argmax(flat_log_prob)]

    print("="*65)
    print(" MAXIMUM A POSTERIORI (MAP) PARAMETERS ")
    print("="*65)
    for i, key in enumerate(cfg.free_keys):
        val = map_fit[i]
        label, unit = display_name(key)
        if key.endswith("_log_ntot"):
            print(f" {label:17s}: {10**val:.3e} {unit}  (log10: {val:.3f})")
        else:
            print(f" {label:17s}: {val:.4g} {unit}")
    print("="*65 + "\n")

    return dict(flat_samples=flat_samples, flat_log_prob=flat_log_prob,
                median_fit=median_fit, map_fit=map_fit, burnin=burnin, thin=thin)


def save_chains(cfg, sampler, res):
    """Save the full and the thinned/flattened chains so they can be re-analysed later."""
    path = f"{cfg.output_dir}/mcmc_chains.npz"
    np.savez(path,
             chain=sampler.get_chain(),              # (n_steps, n_walkers, ndim)
             log_prob=sampler.get_log_prob(),        # (n_steps, n_walkers)
             flat_samples=res["flat_samples"],       # after burn-in and thinning
             flat_log_prob=res["flat_log_prob"],
             param_names=np.array(cfg.free_keys),
             burnin=res["burnin"], thin=res["thin"])
    print(f"[+] Chains saved to {path}")


def save_best_fit(cfg, obs_list, res, y_map, y_median, y_components=None):
    """Write the MAP and median models as ASCII spectra and as model files.

    best_fit_spectrum.txt (one spectrum) or best_fit_spectrum_<i>.txt (several, in the
    order of the config): one row per channel with the frequency, the velocity (if the
    spectrum has a rest_freq), the observed intensity, the MAP and median models, the MAP
    model of each component (if several) and a flag for the channels used in the fit.
    map_model.yaml, median_model.yaml: the parameters in the model-file format
    (for tools/single_model.py or mcmc: init_model).
    """
    from . import __version__
    C_KMS = 299792.458
    names = [m["name"] for m in cfg.molecules]
    several = len(obs_list) > 1
    for i, obs in enumerate(obs_list):
        spec = cfg.spectra[i]
        f0 = spec.get("rest_freq")
        nan = np.full(len(obs.x_obs), np.nan)
        pick = lambda models: models[i] if models is not None and models[i] is not None \
            and len(models[i]) == len(obs.x_obs) else nan
        cols, heads, fmts = [obs.x_obs], ["freq_MHz"], ["%.6f"]
        if f0:
            cols.append(C_KMS * (1.0 - obs.x_obs / f0)); heads.append("v_kms"); fmts.append("%.4f")
        cols += [obs.y_obs, pick(y_map), pick(y_median)]
        heads += ["T_obs", "T_MAP", "T_median"]; fmts += ["%.6g"] * 3
        if y_components:
            for k, yc in enumerate(y_components):
                cols.append(pick(yc)); heads.append(f"T_MAP_{names[k]}"); fmts.append("%.6g")
        cols.append(obs.line_mask.astype(int)); heads.append("in_fit"); fmts.append("%d")
        meta = [f"mcWEEDS {__version__}  run: {cfg.run_name}",
                f"spectrum: {obs.label}   file: {spec.get('file', '')}",
                (f"rest_freq [MHz]: {f0}   (v_kms: LSR velocity relative to rest_freq, radio convention)"
                 if f0 else "rest_freq: none (frequency axis only)"),
                f"RMS noise used in the fit [K]: {obs.rms_estimate:.5g}",
                "line windows [MHz]: " + ", ".join(f"[{a:.3f}, {b:.3f}]" for a, b in obs.line_windows),
                "T in K (main-beam temperature); in_fit = 1 for the channels used in the fit"]
        path = f"{cfg.output_dir}/best_fit_spectrum{f'_{i}' if several else ''}.txt"
        np.savetxt(path, np.column_stack(cols), fmt=fmts, header=" ".join(heads) + "\n" + "\n".join(meta))
    for tag, fit in [("map", res["map_fit"]), ("median", res["median_fit"])]:
        with open(f"{cfg.output_dir}/{tag}_model.yaml", "w") as f:
            f.write(f"# {tag.upper()} model of run {cfg.run_name} (mcWEEDS {__version__}); ntot in cm^-2, "
                    "tex in K, source_size in arcsec, v_off and width in km/s\n")
            for name, v in zip(names, cfg.expand(fit)):
                f.write(f"{name}: {{ntot: {10**v['log_ntot']:.6e}, tex: {v['tex']:.4f}, "
                        f"source_size: {v['source_size']:.4f}, v_off: {v['v_off']:.4f}, "
                        f"width: {v['width']:.4f}}}\n")
    print(f"[+] Best-fit spectra (ASCII) and map_model.yaml / median_model.yaml written to {cfg.output_dir}/")


def write_summary(cfg, sampler, res, config_path=None):
    """Write the results of the run to <output_dir>/fit_summary.txt (plain text)."""
    from . import __version__
    flat = res["flat_samples"]
    p16, p50, p84 = np.percentile(flat, [16, 50, 84], axis=0)
    lines = [f"mcWEEDS {__version__}  run: {cfg.run_name}",
             f"config: {config_path or '-'}",
             f"spectra: {len(cfg.spectra)}   chi_sq_method: {cfg.chi_sq_method}   catalog: {cfg.catalog}",
             f"walkers x steps: {cfg.n_walkers} x {cfg.n_steps}   burn-in: {res['burnin']}   "
             f"thin: {res['thin']}   samples: {len(flat)}   "
             f"mean acceptance: {np.mean(sampler.acceptance_fraction):.3f}",
             "",
             f"{'parameter':24s} {'median':>12s} {'-1sigma':>11s} {'+1sigma':>11s} {'MAP':>12s}"]
    for i, key in enumerate(cfg.free_keys):
        m, lo, hi, mp = p50[i], p16[i], p84[i], res["map_fit"][i]
        label, unit = display_name(key)
        if key.endswith("_log_ntot"):     # fitted as log10(ntot); reported linear
            lines.append(f"{label:24s} {10**m:12.4e} {10**m - 10**lo:11.3e} "
                         f"{10**hi - 10**m:11.3e} {10**mp:12.4e}   [{unit}]")
        else:
            lines.append(f"{label:24s} {m:12.5g} {m - lo:11.3g} {hi - m:11.3g} {mp:12.5g}   [{unit}]")
    lines += [""] + [l.strip() for l in cfg.describe_non_free()]
    path = f"{cfg.output_dir}/fit_summary.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Summary written to {path}")
