"""Read the YAML configuration and work out which parameters are fitted.

Each molecule has five physical parameters: ntot, tex, source_size, v_off, width.
In the YAML file each one can be

    ntot:  [8.0e13, 9.0e15]        # free, uniform prior between the two bounds
    tex:   {fixed: 150.0}          # held fixed at this value (a plain number also works)
    v_off: {shared_with: CH3CN}    # always equal to the same parameter of CH3CN

ntot is given in linear units (cm^-2) but is fitted as log10(ntot).

Each entry of `molecules:` is one emission component. `name` must be unique; the
optional `species` is the catalogue entry used by WEEDS (default: `name`). Two
components of the same species therefore look like

    - name: CH3OH_narrow
      species: CH3OH
    - name: CH3OH_broad
      species: CH3OH

Spectra: either one file holding several lines (spectrum_file + windows at the top level)
or a list `spectra:` of files, each with its own windows. Any file that CLASS opens with
FILE IN can be used: GILDAS VLM cubes (one spectrum per pixel) or CLASS spectra files
(.30m, .cla, ...). Which spectrum of the file is used is set by `select:` (globally and/or
per spectrum; see SELECT_KEYS). By default, the first spectrum at offset (0, 0) is used
(for a VLM cube: the pixel at the projection centre).
"""
import os
import numpy as np
import yaml

PARAMS = ["ntot", "tex", "source_size", "v_off", "width"]  # order of the .mdl columns
UNITS = {"ntot": "cm^-2", "tex": "K", "source_size": "arcsec", "v_off": "km/s", "width": "km/s"}
MAX_MOLECULES = 3
CHI_SQ_METHODS = ["whitened", "standard", "integrated"]
C_KMS = 299792.458

# Selection of the spectrum inside a file (CLASS FIND options; the first match is used).
#   offset:    [x, y] in arcsec (default [0, 0]; the centre pixel of a VLM cube), or "any"
#   match:     tolerance on the offsets [arcsec] (CLASS SET MATCH; default 0.2)
#   number:    observation number n, or [n1, n2]    (FIND /NUMBER)
#   source, line, telescope: names as in the CLASS header (FIND /SOURCE /LINE /TELESCOPE)
SELECT_KEYS = ("offset", "match", "number", "source", "line", "telescope")
SELECT_DEFAULT = {"offset": [0.0, 0.0], "match": 0.2}


def _positive(v, what):
    """None, or a positive float (for noise_rms)."""
    if v is None:
        return None
    v = float(v)
    if not v > 0:
        raise ValueError(f"{what} must be positive")
    return v


def _read_select(sel, where):
    """Check and normalise a `select:` block (dict) -> dict with all SELECT_KEYS."""
    sel = dict(sel or {})
    bad = [k for k in sel if k not in SELECT_KEYS]
    if bad:
        raise ValueError(f"{where}: unknown select option(s) {bad}; use {list(SELECT_KEYS)}")
    out = {k: None for k in SELECT_KEYS}
    out.update(sel)
    off = out["offset"]
    if isinstance(off, str):
        if off.lower() not in ("any", "none", "all"):
            raise ValueError(f"{where}: select offset must be [x, y] or 'any'")
        out["offset"] = None
    elif off is not None:
        if not (isinstance(off, (list, tuple)) and len(off) == 2):
            raise ValueError(f"{where}: select offset must be [x, y] (arcsec) or 'any'")
        out["offset"] = [float(off[0]), float(off[1])]
    out["match"] = float(out["match"]) if out["match"] is not None else None
    num = out["number"]
    if num is not None:
        num = [int(num), int(num)] if isinstance(num, (int, float)) else [int(n) for n in num]
        if len(num) != 2:
            raise ValueError(f"{where}: select number must be n or [n1, n2]")
        out["number"] = num
    for k in ("source", "line", "telescope"):
        if out[k] is not None:
            out[k] = str(out[k])
            if " " in out[k]:
                raise ValueError(f"{where}: select {k} must not contain spaces")
    return out


def _param_key(mol_name, param):
    """Name used for a free parameter, e.g. 'CH3CN_log_ntot' or 'CCH_tex'."""
    return f"{mol_name}_log_ntot" if param == "ntot" else f"{mol_name}_{param}"


def display_name(key):
    """Name and unit of a parameter key for printed results: 'X_log_ntot' -> ('X_ntot', 'cm^-2')."""
    if key.endswith("_log_ntot"):
        return key[:-len("_log_ntot")] + "_ntot", UNITS["ntot"]
    for p in PARAMS[1:]:
        if key.endswith("_" + p):
            return key, UNITS[p]
    return key, ""


def _param_label(mol_label, param):
    """LaTeX label for plots, e.g. '$\\log N$(CH$_3$CN)'."""
    symbol = {"ntot": r"$\log N$", "tex": r"$T_\mathrm{ex}$",
              "source_size": r"$\theta_\mathrm{s}$",
              "v_off": r"$V_\mathrm{off}$", "width": r"$\Delta V$"}[param]
    return f"{symbol}({mol_label})"


class Config:
    """Plain container for everything read from the YAML file.

    data_only=True reads only what is needed to extract and plot the spectra (files,
    selection, windows if given, plot settings): `molecules:` and the windows may be
    missing or incomplete. Used by tools/plot_observed_lines.py to look at the data
    before the model part of the config is written. Such a Config cannot be fitted.
    """

    def __init__(self, path, data_only=False):
        self.data_only = data_only
        with open(path) as f:
            raw = yaml.safe_load(f)

        # --- General settings ---
        self.run_name = str(raw.get("run_name", "run"))
        self.output_dir = raw.get("output_dir", f"MCMC_{self.run_name}")
        self.mdl_dir = os.path.join(self.output_dir, "temp_mdl_files")
        self.telescope_size = raw.get("telescope_size")   # default for all spectra
        self.catalog = raw.get("catalog", "jpl")
        # Default spectrum selection for all spectra (each spectrum may override keys)
        glob_sel = dict(SELECT_DEFAULT)
        glob_sel.update(raw.get("select") or {})
        self.select_default = glob_sel
        self.chi_sq_method = raw.get("chi_sq_method", "whitened")
        # Optional fixed noise ACF [1, rho_1, rho_2, ...] for the whitened and integrated chi-square
        # (otherwise estimated from the noise windows).
        acf = raw.get("noise_acf")
        self.noise_acf = None if acf is None else [float(a) for a in acf]
        if self.noise_acf is not None and abs(self.noise_acf[0] - 1.0) > 1e-9:
            raise ValueError("noise_acf must start with 1.0")
        if self.noise_acf is not None and len(self.noise_acf) > 50:
            raise ValueError("noise_acf: at most 50 values (lags 0 to 49)")
        if self.noise_acf is not None and any(abs(a) > 1.0 for a in self.noise_acf):
            raise ValueError("noise_acf: every value must be between -1 and 1")
        if self.chi_sq_method not in CHI_SQ_METHODS:
            raise ValueError(f"chi_sq_method must be one of {CHI_SQ_METHODS}")
        # Optional fixed RMS noise [K] for all spectra (a spectrum's own noise_rms overrides
        # it); otherwise the RMS of each spectrum is measured in its noise windows.
        self.noise_rms = _positive(raw.get("noise_rms"), "noise_rms")

        mcmc = raw.get("mcmc", {})
        self.n_walkers = int(mcmc.get("n_walkers", 40))
        self.n_steps = int(mcmc.get("n_steps", 1000))
        self.n_threads = int(mcmc.get("n_threads", 8))
        # Optional start of the walkers in a small ball around a model (same YAML format as
        # tools/single_model.py) instead of uniformly over the priors. init_scatter is the
        # half-width of the ball as a fraction of each prior range.
        self.init_model = mcmc.get("init_model")
        self.init_scatter = float(mcmc.get("init_scatter", 0.05))

        # --- Spectra ---
        # Either one spectrum (spectrum_file + line_windows + noise_windows at the
        # top level) or a list `spectra:`,
        # each entry with its own file and windows. All spectra are modelled with
        # the same physical parameters; their chi-square values are summed.
        if "spectra" in raw:
            spec_list = raw["spectra"]
        else:
            if "spectrum_file" not in raw:
                raise ValueError("Give either spectrum_file (one spectrum) or spectra: (a list)")
            spec_list = [{"file": raw["spectrum_file"],
                          "line_windows": raw.get("line_windows"),
                          "noise_windows": raw.get("noise_windows"),
                          "line_windows_kms": raw.get("line_windows_kms"),
                          "noise_windows_kms": raw.get("noise_windows_kms"),
                          "label": raw.get("spectrum_label", os.path.basename(raw["spectrum_file"]))}]
            # (a top-level noise_rms applies to this spectrum through self.noise_rms)
            if "rest_freq" in raw:
                spec_list[0]["rest_freq"] = raw["rest_freq"]
        self.spectra = [self._read_spectrum(s) for s in spec_list]

        plot = raw.get("plot") or {}
        first = self.spectra[0]
        lw = first["line_windows"]
        self.plot_fmin = plot.get("fmin", min(w[0] for w in lw) if lw else None)
        self.plot_fmax = plot.get("fmax", max(w[1] for w in lw) if lw else None)
        self.v_sys = plot.get("v_sys", 0.0)
        self.transitions = [(float(f), str(lab)) for f, lab in raw.get("transitions", [])]

        # --- Molecules and parameters ---
        if data_only:
            # labels only (if given); priors are not read or checked
            self.molecules = [{"name": m.get("name", f"comp{k}"),
                               "species": m.get("species", m.get("name", "")),
                               "label": m.get("label", m.get("name", ""))}
                              for k, m in enumerate(raw.get("molecules") or [])
                              if isinstance(m, dict)]
            self.ndim = 0
            return
        if "molecules" not in raw:
            raise ValueError("No `molecules:` in the config (needed for a fit)")
        self._read_molecules(raw["molecules"])

    def _read_spectrum(self, s):
        """One entry of `spectra:` -> dict with file, label, windows [MHz], telescope size.

        Windows can be given in frequency (line_windows / noise_windows, MHz) or,
        for single-line spectra, in velocity relative to `rest_freq`
        (line_windows_kms / noise_windows_kms, km/s, radio convention).
        """
        name = s["file"]
        if not os.path.isfile(name):
            raise ValueError(f"Spectrum file not found: {name} (paths are relative to the "
                             f"folder you run from: {os.getcwd()})")
        if " " in name:
            print(f"[!] Warning: {name}: CLASS splits commands at spaces, so it may not open "
                  f"this path; use a path without spaces (e.g. relative to the run folder).")
        sel = dict(self.select_default)
        sel.update(s.get("select") or {})
        select = _read_select(sel, name)
        ts = s.get("telescope_size", self.telescope_size)
        if ts is None and not self.data_only:
            raise ValueError(f"{name}: telescope_size missing (set it globally or per spectrum)")

        def windows(key):
            if s.get(key):
                return [tuple(float(x) for x in w) for w in s[key]]
            if s.get(key + "_kms"):
                if "rest_freq" not in s:
                    raise ValueError(f"{name}: {key}_kms needs rest_freq [MHz]")
                f0 = float(s["rest_freq"])
                out = []
                for v1, v2 in s[key + "_kms"]:
                    fa, fb = (f0 * (1.0 - v / C_KMS) for v in (v1, v2))
                    out.append((min(fa, fb), max(fa, fb)))
                return out
            return []

        line_w, noise_w = windows("line_windows"), windows("noise_windows")
        rms = _positive(s.get("noise_rms"), f"{name}: noise_rms")
        if rms is None:
            rms = self.noise_rms
        if not line_w and not self.data_only:
            raise ValueError(f"{name}: no line windows given")
        if not noise_w and not self.data_only:
            # noise windows give the RMS and (whitened, integrated chi-square) the noise ACF;
            # they can be omitted only if both are given in the config
            if rms is None:
                raise ValueError(f"{name}: noise windows are needed to estimate the RMS noise "
                                 f"(or give noise_rms)")
            if self.chi_sq_method in ("whitened", "integrated") and self.noise_acf is None:
                raise ValueError(f"{name}: noise windows are needed to estimate the noise "
                                 f"correlation (or give noise_acf)")
        return {"file": name, "label": s.get("label", os.path.basename(name)),
                "line_windows": line_w, "noise_windows": noise_w,
                "telescope_size": ts, "plot_range": s.get("plot_range"), "select": select,
                "noise_rms": rms,
                "rest_freq": float(s["rest_freq"]) if "rest_freq" in s else None}

    def _read_molecules(self, mol_list):
        if not 1 <= len(mol_list) <= MAX_MOLECULES:
            raise ValueError(f"Between 1 and {MAX_MOLECULES} molecules can be modelled "
                             f"(got {len(mol_list)})")

        self.molecules = []
        for m in mol_list:
            self.molecules.append({
                "name": m["name"],
                "species": m.get("species", m["name"]),   # catalogue entry used by WEEDS
                "label": m.get("label", m["name"]),
            })
        names = [m["name"] for m in self.molecules]
        if len(set(names)) != len(names):
            raise ValueError("Molecule names must be unique (use `species:` for several "
                             "components of the same catalogue entry)")

        # First pass: free and fixed parameters. Free parameters are ordered
        # molecule by molecule (ntot, tex, source_size, v_off, width).
        self.free_keys, self.free_labels, self.bounds = [], [], []
        self.free_params = []   # (molecule name, parameter) of each free parameter
        where = {}   # (mol, param) -> ("theta", index) or ("const", value)
        shared = {}  # (mol, param) -> other molecule name
        for m, spec in zip(self.molecules, mol_list):
            for p in PARAMS:
                if p not in spec:
                    raise ValueError(f"{m['name']}: missing parameter '{p}'")
                s = spec[p]
                if isinstance(s, (list, tuple)) and len(s) == 2:
                    lo, hi = float(s[0]), float(s[1])
                    if p in ("ntot", "source_size") and lo <= 0:
                        raise ValueError(f"{m['name']} {p}: bounds must be positive")
                    if p == "ntot":
                        lo, hi = np.log10(lo), np.log10(hi)
                    if not lo < hi:
                        raise ValueError(f"{m['name']} {p}: lower bound must be below upper bound")
                    where[(m["name"], p)] = ("theta", len(self.free_keys))
                    self.free_keys.append(_param_key(m["name"], p))
                    self.free_params.append((m["name"], p))
                    self.free_labels.append(_param_label(m["label"], p))
                    self.bounds.append((lo, hi))
                elif (isinstance(s, dict) and "fixed" in s) or isinstance(s, (int, float)):
                    v = float(s["fixed"] if isinstance(s, dict) else s)
                    if p in ("ntot", "source_size") and v <= 0:
                        raise ValueError(f"{m['name']} {p}: value must be positive")
                    where[(m["name"], p)] = ("const", np.log10(v) if p == "ntot" else v)
                elif isinstance(s, dict) and "shared_with" in s:
                    shared[(m["name"], p)] = s["shared_with"]
                else:
                    raise ValueError(f"{m['name']} {p}: use [min, max], value, {{fixed: value}} "
                                     f"or {{shared_with: MOLECULE}}")

        # Second pass: shared parameters point at another molecule's free/fixed value.
        self.shared = {}
        for (mol, p), other in shared.items():
            if other not in names or other == mol:
                raise ValueError(f"{mol} {p}: shared_with must name another molecule in the list")
            if (other, p) not in where:
                raise ValueError(f"{mol} {p}: {other} {p} is itself shared; "
                                 f"share with the molecule that is fitted or fixed")
            where[(mol, p)] = where[(other, p)]
            self.shared[(mol, p)] = other

        self._where = where
        self.ndim = len(self.free_keys)
        if self.ndim == 0:
            raise ValueError("All parameters are fixed or shared: nothing to fit")

    def expand(self, theta):
        """Turn the vector of free parameters into full per-molecule values.

        Returns a list (one entry per molecule) of dicts with keys
        'log_ntot', 'tex', 'source_size', 'v_off', 'width'.
        """
        out = []
        for m in self.molecules:
            vals = {}
            for p in PARAMS:
                kind, x = self._where[(m["name"], p)]
                v = theta[x] if kind == "theta" else x
                vals["log_ntot" if p == "ntot" else p] = v
            out.append(vals)
        return out

    def describe_non_free(self):
        """Human-readable lines for fixed and shared parameters (for the summary)."""
        lines = []
        for m in self.molecules:
            for p in PARAMS:
                key, unit = display_name(_param_key(m["name"], p))
                if (m["name"], p) in self.shared:
                    lines.append(f" {key:17s}: shared with {self.shared[(m['name'], p)]}")
                else:
                    kind, x = self._where[(m["name"], p)]
                    if kind == "const":
                        txt = f"{10**x:.3e}" if p == "ntot" else f"{x:.2f}"
                        lines.append(f" {key:17s}: fixed at {txt} {unit}")
        return lines
