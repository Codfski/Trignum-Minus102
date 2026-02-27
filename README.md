# Curvature Bifurcation in Self-Consistent Neural Loss Landscapes

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Authors:** Moez Abdessattar, Claude-3  
**Institution:** Trignum Project / Epistemic Geometry Lab  
**Date:** February 27, 2026

---

## 📖 Overview

This repository contains the complete code and analysis for the paper:

> **"Curvature Bifurcation Induced by Self-Consistency Coupling in Neural Loss Landscapes"**

We investigate what happens when neural networks try to model themselves through a self-consistency loss of the form:

$$L(\theta) = L_{\text{task}}(\theta) + \alpha \| f_\theta(\theta) - \theta \|^2$$

**Key findings:**

- The Hessian of the self-consistency term decomposes into a positive semidefinite linear part and an indefinite nonlinear part.
- At a critical weight $\alpha_c$, the minimum eigenvalue of the total Hessian crosses zero.
- This bifurcation is reproducible across dimensions ($n=50-200$) and random initializations.
- $\alpha_c = 1.85 \pm 0.11$ under our experimental conditions.
- The phenomenon explains instabilities in meta-learning, world models, and reflective architectures.

---

## 🧪 What Started as -102

This investigation began with an intriguing numerical observation: under certain heuristic scaling, the bifurcation appeared near a fixed value of **-102**. Systematic analysis revealed this was an artifact of scaling choices. The journey from illusory constant to rigorous theory is documented in our paper.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Quick Start

Run the main experiment notebook:

```bash
jupyter notebook notebooks/curvature_bifurcation.ipynb
```

Or execute experiments directly:

```bash
python src/experiments.py
```

---

## 📊 Reproducing Results

Main Experiments:

1. **Curvature Transition (Figure 1)**
   `python src/experiments.py --experiment transition --n 50 --trials 20`
2. **$\alpha_c$ Distribution (Figure 2)**
   `python src/experiments.py --experiment histogram --n 50 --trials 20`
3. **Dimensional Scaling (Figure 3)**
   `python src/experiments.py --experiment scaling --n_list 50 75 100 150 200 --trials 50`
4. **Task Hessian Sensitivity (Figure 4)**
   `python src/experiments.py --experiment sensitivity --n 50 --trials 20 --n_hessians 10`
5. **The -102 Illusion (Figure 5)**
   `python src/experiments.py --experiment illusion --n 50 --trials 20`

All figures are saved to the `figures/` directory.

---

## 📁 Repository Structure

```
Trignum-Minus102/
│
├── README.md                 # This file
├── LICENSE                   # MIT License
├── requirements.txt          # Dependencies
├── .gitignore                # Git ignore rules
│
├── notebooks/
│   └── curvature_bifurcation.ipynb   # Analysis notebook
│
├── src/
│   ├── curvature_model.py    # Core math (f, J, H_f, S)
│   └── experiments.py        # Experiment runners
│
├── figures/                  # Generated figures
├── data/                     # Data placeholder
└── paper/
    └── manuscript.md         # Full paper in Markdown
```

---

## 📚 Citation

If you use this code or ideas in your research, please cite:

```bibtex
@article{abdessattar2026curvature,
  title={Curvature Bifurcation Induced by Self-Consistency Coupling in Neural Loss Landscapes},
  author={Abdessattar, Moez and Claude-3},
  journal={Preprint},
  year={2026},
  note={Code available: https://github.com/Codfski/Trignum-Minus102}
}
```

---

## 📄 License

MIT License. See `LICENSE` file for details.
