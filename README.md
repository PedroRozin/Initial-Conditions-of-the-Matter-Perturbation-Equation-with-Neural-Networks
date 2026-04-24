# Paper Project

This repository is the clean, paper-facing workspace derived from the thesis pipeline.

## Purpose

- Keep paper-specific code, notebooks, figures, and notes in one place.
- Reuse the thesis virtual environment instead of creating a second one.
- Reuse stable helper code from the thesis repo through `PYTHONPATH`.

## Local layout

- `notebooks/`: paper analysis notebooks.
- `src/`: paper-specific Python modules.
- `scripts/`: small shell helpers and automation.
- `data/`: curated input data and lightweight artifacts.
- `figures/`: exported plots for the paper.
- `references/`: notes, citations, and writing support files.

## Setup

1. Create the local link to the thesis environment if it is not already present:

   ```bash
   ln -s /home/pedrorozin/scripts/venv .venv
   ```

2. Activate it:

   ```bash
   source .venv/bin/activate
   ```

3. Expose the thesis helper modules when you need them:

   ```bash
   export PYTHONPATH="/home/pedrorozin/scripts/python:${PYTHONPATH:-}"
   ```

The main reusable helper module currently lives in `/home/pedrorozin/scripts/python/funciones_tesis.py`.

## VS Code

This repo includes `.vscode/settings.json` so the editor uses the shared environment and can resolve the thesis helper modules.
# Initial-Conditions-of-the-Matter-Perturbation-Equation-with-Neural-Networks
