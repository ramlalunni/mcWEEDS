"""Extract the observed spectrum with CLASS and characterise the noise.

This runs once, before the MCMC starts.
"""
import os
import subprocess
import numpy as np
import scipy.linalg

from .class_model import load_spectrum_commands


class Observation:
    """One observed spectrum plus everything the likelihood needs for it."""

    def __init__(self, label, x_obs, y_obs, df_mhz, line_windows, line_mask,
                 rms_estimate, window_whiteners=None):
        self.label = label
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.df_mhz = df_mhz
        self.line_windows = line_windows
        self.line_mask = line_mask
        self.rms_estimate = rms_estimate
        self.window_whiteners = window_whiteners
        self.acf = None    # noise ACF (whitened and integrated chi-square)


def _acf_sums(y_data, noise_mask, max_lag=50):
    """Lag-product sums and pair counts within contiguous noise runs (where noise_mask is True)."""
    in_noise = noise_mask
    edges = np.diff(in_noise.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if in_noise[0]: starts = np.r_[0, starts]
    if in_noise[-1]: ends = np.r_[ends, len(in_noise)]

    lag_sums, lag_counts = np.zeros(max_lag), np.zeros(max_lag)
    for s, e in zip(starts, ends):
        seg = y_data[s:e] - np.mean(y_data[s:e])
        n = len(seg)
        for k in range(min(max_lag, n)):
            prod = seg[:n-k] * seg[k:]
            lag_sums[k] += np.sum(prod)
            lag_counts[k] += len(prod)
    return lag_sums, lag_counts


def segmented_acf(y_data, noise_mask, max_lag=50):
    """Compute ACF only within contiguous pure noise runs (where noise_mask is True)."""
    lag_sums, lag_counts = _acf_sums(y_data, noise_mask, max_lag)
    acf_out = lag_sums / lag_counts
    acf_out /= acf_out[0]
    return acf_out


def pooled_acf(items, max_lag=50):
    """ACF pooled over several spectra: items = [(y, noise_mask, rms), ...].

    Each spectrum's lag products are divided by its own variance before pooling,
    so spectra with different noise levels contribute equally per channel pair.
    For a single spectrum this is identical to segmented_acf.
    """
    if len(items) == 1:
        y, mask, _ = items[0]
        return segmented_acf(y, mask, max_lag)
    tot_sums, tot_counts = np.zeros(max_lag), np.zeros(max_lag)
    for y, mask, rms in items:
        sums, counts = _acf_sums(y, mask, max_lag)
        tot_sums += sums / rms**2
        tot_counts += counts
    with np.errstate(invalid="ignore", divide="ignore"):
        acf_out = np.where(tot_counts > 0, tot_sums / np.maximum(tot_counts, 1), 0.0)
    acf_out /= acf_out[0]
    return acf_out


def truncate_acf(acf_array, n_samples, n_sigma=2.0):
    threshold = n_sigma / np.sqrt(n_samples)
    acf_trunc = acf_array.copy()
    below = np.where(np.abs(acf_array) < threshold)[0]
    if len(below) > 0 and below[0] > 0:
        acf_trunc[below[0]:] = 0.0
    return acf_trunc


def check_acf(acf, n_ch):
    """Stop if the ACF does not give a valid (positive-definite) covariance for n_ch channels.

    A truncated ACF estimate is not necessarily a valid autocorrelation function; the
    Toeplitz matrix built from it can then have negative eigenvalues, and the chi-square
    is meaningless. n_ch is the largest line window: smaller windows have no smaller
    eigenvalue (eigenvalues of nested Toeplitz matrices interlace).
    """
    if n_ch < 2:
        return
    rho = np.zeros(n_ch)
    m = min(n_ch, len(acf))
    rho[:m] = acf[:m]
    lam_min = np.linalg.eigvalsh(scipy.linalg.toeplitz(rho)).min()
    if lam_min < -1e-6:
        raise RuntimeError(
            f"The noise ACF {np.round(acf[:np.count_nonzero(acf)], 3)} does not give a valid "
            f"covariance matrix for {n_ch} channels (smallest eigenvalue {lam_min:.3g} < 0), so "
            f"the chi-square would be meaningless. Give noise_acf in the config (e.g. measured "
            f"on a larger line-free region), or use more or wider noise windows.")


def build_window_whiteners(acf_array, rms, windows, x_data):
    whiteners = []
    for fmin, fmax in windows:
        w_mask = (x_data > fmin) & (x_data < fmax)
        n_ch = np.sum(w_mask)
        if n_ch == 0: continue

        lags = np.arange(n_ch)
        rho = np.array([acf_array[k] if k < len(acf_array) else 0.0 for k in lags])
        C = rms**2 * scipy.linalg.toeplitz(rho)

        evals, evecs = scipy.linalg.eigh(C)
        if np.min(evals) <= 0:
            threshold = 1e-8 * np.max(evals)
            evals[evals < threshold] = threshold
            C = (evecs * evals) @ evecs.T
            C = 0.5 * (C + C.T)  # Enforce exact symmetry against float64 roundoff

        C += 1e-8 * rms**2 * np.eye(n_ch)
        L = scipy.linalg.cholesky(C, lower=True)
        whiteners.append((w_mask, L))
    return whiteners


def _extract_spectrum(cfg, spec, tag):
    """Run CLASS once to dump the header and channels of one spectrum.

    The spectrum is loaded with the same commands as in every model run
    (class_model.load_spectrum_commands). Returns frequencies [MHz], intensities and the
    channel width [MHz]; prints how many spectra of the file matched the selection.
    """
    out = cfg.output_dir
    header_file = f"{out}/obs_header{tag}.txt"
    spectrum_file = f"{out}/obs_spectrum{tag}.txt"
    found_file = f"{out}/obs_found{tag}.txt"
    for f_ in (header_file, spectrum_file, found_file):   # never read results of an earlier run
        if os.path.exists(f_):
            os.remove(f_)
    load = load_spectrum_commands(spec)
    macro = load[:-1] + [                   # all but "get f": record the number of matches first
        f"sic output {found_file}", "say 'found'", "sic output",
        "if (found.eq.0) then", "exit", "endif",   # nothing matched: stop (GET would pause CLASS)
        load[-1], "set unit f",
        f"sic output {header_file}",
        "say 'frequency'", "say 'reference'", "say 'freq_step'", "sic output",
        f"sic output {spectrum_file}",
        "for i 1 to channels", "say 'rx[i]' \" \" 'ry[i]'", "next", "sic output", "exit"
    ]
    with open(f"{out}/init{tag}.class", "w") as f:
        f.write("\n".join(macro))
    try:
        res = subprocess.run(["class", "-nw", f"@{out}/init{tag}.class"], capture_output=True,
                             text=True, stdin=subprocess.DEVNULL, timeout=60)
    except subprocess.TimeoutExpired as e:
        log = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        raise RuntimeError(f"CLASS did not finish reading {spec['file']} within 60 s (it probably "
                           f"stopped on an error). Macro: {out}/init{tag}.class. Its output was:\n"
                           f"{log[-3000:]}") from None

    n_found = None
    if os.path.exists(found_file):
        try:
            n_found = int(float(open(found_file).read().split()[0]))
        except (ValueError, IndexError):
            pass
    if n_found == 0:
        raise RuntimeError(f"{spec['file']}: no spectrum matches the selection "
                           f"({' '.join(load[1:-1])}). Check `select:` in the config "
                           f"(e.g. the offset of the spectrum).")
    if not (os.path.exists(header_file) and os.path.exists(spectrum_file)):
        raise RuntimeError(f"CLASS could not extract {spec['file']}. Its output was:\n"
                           f"{res.stdout[-3000:]}\n{res.stderr[-2000:]}")
    if n_found is not None and n_found > 1:
        print(f"    [!] {spec['file']}: {n_found} spectra match the selection; the first is used "
              f"(narrow it down with `select:`).")

    header = np.loadtxt(header_file)
    f0, ref_chan, df_mhz = header[0], header[1], abs(header[2])

    y_obs = np.atleast_2d(np.loadtxt(spectrum_file, comments='!'))[:, 1]
    channels = np.arange(1, len(y_obs) + 1)
    x_obs_freq = f0 + (channels - ref_chan) * header[2]
    return x_obs_freq, y_obs, df_mhz


# Public name: extract one spectrum without building the noise model (used by tools)
extract_spectrum = _extract_spectrum


def _mask(x, windows):
    m = np.zeros_like(x, dtype=bool)
    for fmin, fmax in windows:
        m |= (x > fmin) & (x < fmax)
    return m


def init_observation(cfg):
    """Extract every spectrum and build its noise model. Returns a list of Observation."""
    print("[+] Extracting baseline observation data...")
    multi = len(cfg.spectra) > 1
    obs_list, noise_masks, measured_rms = [], [], []

    for i, spec in enumerate(cfg.spectra):
        tag = f"_{i}" if multi else ""
        kind = "VLM cube" if spec["file"].lower().endswith(".vlm") else "CLASS file"
        sel = spec["select"]
        where = ("first spectrum" if sel["offset"] is None
                 else f"offset ({sel['offset'][0]:g}, {sel['offset'][1]:g})")
        print(f"    {'[' + str(i) + '] ' if multi else ''}{spec['file']} ({kind}, {where})"
              + (f": {spec['label']}" if multi else ""))
        x_obs_freq, y_obs, df_mhz = _extract_spectrum(cfg, spec, tag)

        # 1. Standard line_mask for target fitting
        line_mask = _mask(x_obs_freq, spec["line_windows"])
        # 2. Pure noise mask built exclusively from hand-picked line-free windows
        noise_acf_mask = _mask(x_obs_freq, spec["noise_windows"])

        # RMS: from the config (noise_rms) if given, else from the pure noise windows
        # (one value per spectrum). The measured value is also used to normalise the ACF.
        measured = float(np.std(y_obs[noise_acf_mask])) if noise_acf_mask.sum() > 1 else None
        rms_estimate = spec.get("noise_rms") or measured
        if rms_estimate is None:
            raise RuntimeError(f"{spec['file']}: fewer than 2 channels in the noise windows; "
                               f"check them, or give noise_rms")
        ind = "        " if multi else "    "
        if multi:
            print(f"        {x_obs_freq.min():.3f}-{x_obs_freq.max():.3f} MHz, "
                  f"{len(y_obs)} channels of {df_mhz:.4f} MHz; "
                  f"{line_mask.sum()} fitted / {noise_acf_mask.sum()} noise channels")
        if spec.get("noise_rms"):
            meas = f"{measured:.4f} K" if measured is not None else "no noise channels"
            print(f"{ind}RMS noise from the config (noise_rms): {rms_estimate:.4f} K "
                  f"(noise windows: {meas})")
        else:
            print(f"{ind}Estimated RMS noise (pure noise windows): {rms_estimate:.4f} K")
        for fmin, fmax in spec["line_windows"]:
            if not np.any((x_obs_freq > fmin) & (x_obs_freq < fmax)):
                print(f"    [!] Warning: line window {fmin}-{fmax} MHz contains no channels "
                      f"of {spec['label']}")

        obs_list.append(Observation(spec["label"], x_obs_freq, y_obs, df_mhz,
                                    spec["line_windows"], line_mask, rms_estimate))
        noise_masks.append(noise_acf_mask)
        measured_rms.append(measured)

    if cfg.chi_sq_method in ('whitened', 'integrated'):
        # The noise ACF is pooled over all spectra with the same channel width
        # (same spectral setup); with a single spectrum it is that spectrum's ACF.
        groups = {}
        for i, o in enumerate(obs_list):
            groups.setdefault(round(o.df_mhz, 6), []).append(i)

        for df_key, members in groups.items():
            # ACF from the noise windows, each spectrum normalised by its measured variance
            items = [(obs_list[i].y_obs, noise_masks[i], measured_rms[i]) for i in members
                     if measured_rms[i] is not None]
            n_noise_samples = int(sum(noise_masks[i].sum() for i in members))
            if cfg.noise_acf is not None:
                # ACF given in the config (e.g. measured on a larger noise sample);
                # avoids a noisy estimate from few noise channels being truncated to zero.
                acf = np.zeros(50)               # lags 0-49, as for the estimated ACF
                acf[:len(cfg.noise_acf)] = cfg.noise_acf
                print("    Noise ACF taken from the config (noise_acf).")
            elif not items:
                raise RuntimeError(f"Spectra {members}: no noise channels to estimate the noise "
                                   f"correlation; give noise windows or noise_acf")
            else:
                raw_acf = pooled_acf(items, max_lag=50)
                acf = truncate_acf(raw_acf, n_noise_samples, n_sigma=2.0)

            corr_length = np.count_nonzero(acf)
            if multi:
                print(f"    Channel width {df_key} MHz (spectra {members}):")
            print(f"    Noise ACF (first 5 lags): {np.round(acf[:5], 3)}")
            if corr_length > 1:
                print(f"    [+] Correlated noise detected (effective length: {corr_length} channels).")
            else:
                print("    [+] No significant correlation between channels.")

            # the ACF must give a valid covariance matrix (largest line window of the group)
            n_max = max(int(((obs_list[i].x_obs > a) & (obs_list[i].x_obs < b)).sum())
                        for i in members for a, b in obs_list[i].line_windows)
            check_acf(acf, n_max)

            for i in members:
                o = obs_list[i]
                o.acf = acf
                if cfg.chi_sq_method != 'whitened':
                    continue
                try:
                    o.window_whiteners = build_window_whiteners(acf, o.rms_estimate,
                                                                o.line_windows, o.x_obs)
                except scipy.linalg.LinAlgError as e:
                    raise RuntimeError(f"Noise covariance matrix not positive-definite. {e}")
    else:
        print(f"    [+] Using '{cfg.chi_sq_method}' method for chi-square calculation.")

    return obs_list
