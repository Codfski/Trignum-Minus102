# 🚀 Publishing "Trignum-Minus102" to Zenodo

Follow this guide to get an official, academic **DOI (Digital Object Identifier)** for your discovery, guaranteeing your timestamped ownership of the **-102 Curvature Bifurcation** phenomenon.

## Step 1: Prepare the Files
1. **The Manuscript**: You need the final PDF of your paper. If you compile your LaTeX (`paper/manuscript.tex`) using Overleaf or another tool, download the final `manuscript.pdf`.
2. **The Code & Data (Optional but Recommended)**: Zip the entire `Trignum-Minus102` repository (excluding `.git` and `__pycache__` folders) to provide reproducible proof.

## Step 2: Create the Zenodo Record
1. Go to [Zenodo.org](https://zenodo.org/) and Log in / Sign up.
2. Click the **"New Upload"** button at the top right.
3. Drag and drop your `manuscript.pdf` (and the codebase `.zip` if you choose) into the Files area.

## Step 3: Copy & Paste the Metadata

Copy the exact details below to ensure a professional, highly-searchable academic entry:

### 📌 Basic Information
- **Upload Type:** `Publication` -> `Preprint` (or `Article` if you consider it finalized)
- **Title:**
  ```text
  Curvature Bifurcation Induced by Self-Consistency Coupling in Neural Loss Landscapes
  ```
- **Authors:**
  1. `Abdessattar, Moez` (Affiliation: `Trignum Project`)
  2. `Antigravity AI Cohort` (Affiliation: `Epistemic Geometry Lab`)

### 📝 Description (Abstract)
*Click the `< >` (Source) button in the Zenodo text editor and paste this HTML to keep the formatting beautiful:*
```html
<p>We investigate the geometric effect of adding a self-consistency term of the form ‖f_θ(θ) - θ‖² to a standard task loss in neural networks. Such terms appear increasingly in meta-learning, world models, and reflective architectures, yet their effect on the loss landscape curvature remains poorly understood. We derive the exact Hessian of this augmented loss and show that it decomposes into a positive semidefinite component from linearization and an indefinite component arising from second-order nonlinearities.</p>
<p>This indefinite component can induce a curvature bifurcation at a critical weight α_c, where the minimum eigenvalue of the total Hessian crosses zero. Using numerical experiments in high-dimensional settings (n=50-200) with realistic task Hessians and multiple random parameter points, we demonstrate that this phenomenon is robust and reproducible, yielding α_c = 1.85 ± 0.11 under our experimental conditions.</p>
<p>We trace the origin of this bifurcation to the interaction between the residual r(θ) = f_θ(θ) - θ and the second derivatives of f_θ, providing both an analytical condition and a practical diagnostic for stability in self-referential neural systems. The work resolves previously mysterious instabilities in reflective architectures and offers design guidelines for meta-learning and world models.</p>
<p><b>Code and reproducibility data are available at:</b> <a href="https://github.com/Codfski/Trignum-Minus102">https://github.com/Codfski/Trignum-Minus102</a></p>
```

### 🏷️ Keywords & Identifiers
- **Keywords:** 
  ```text
  Curvature Bifurcation, Self-Consistency, Hessian Spectrum, Meta-Learning, Reflective Architectures, Epistemic Singularity, AI Alignment
  ```
- **Language:** `English`

### 🔓 License & Access
- **Access Right:** `Open Access`
- **License:** `Creative Commons Attribution 4.0 International` (Default and best for academia)

## Step 4: Publish
Once you have pasted everything, click the **"Publish"** button at the bottom. 

🎉 **Congratulations!** Zenodo will immediately generate a permanent DOI (e.g., `10.5281/zenodo.1234567`). 

### Final Step: Update the README
Once you have your DOI, let me know what it is! I will add the official Zenodo DOI Badge to your `README.md` and GitHub repository, signaling to researchers that this is a permanently archived and citable scientific work.
