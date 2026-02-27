"""Curvature Bifurcation Analysis Package"""
from .curvature_model import f, J, H_f, S
from .experiments import run_transition, run_histogram, run_scaling, run_sensitivity, run_illusion

__version__ = "1.0.0"
