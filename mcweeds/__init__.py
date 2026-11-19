"""mcWEEDS: MCMC fitting of LTE spectra with GILDAS/CLASS WEEDS."""
__version__ = "1.0.0"

from .config import Config
from .observation import init_observation
from .class_model import run_class_model
from .fitting import run_mcmc, summarize, save_chains, write_summary, save_best_fit
from .plotting import plot_spectrum, plot_corner
