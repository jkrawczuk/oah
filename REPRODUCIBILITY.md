# Reproducibility Guide (Artifact)

This guide describes how to reproduce the main paper outputs from the public release `v1.0.3`.

## Scope
- Training/evaluation protocol: prequential (predict-then-update)
- Repeats: `10` (configured in experiment scripts)
- Output format: CSV files in `results/` and PNG figures in `figures/datasets/`

## Commands
Run all commands from repository root:

```bash
poetry install
poetry run python scripts/compare_2d.py
poetry run python scripts/compare_real.py
poetry run python scripts/ablation.py
poetry run python scripts/export_final_dataset_figures.py
```

## Mapping: Manuscript Output -> Script -> Files
- 2D benchmark comparison tables (prequential accuracy, ISA@20, K):
  - `scripts/compare_2d.py`
  - `results/oah_baselines_2d.csv`
- Real-dataset comparison tables (prequential accuracy, ISA@20, K):
  - `scripts/compare_real.py`
  - `results/oah_baselines_real.csv`
- Ablation tables:
  - `scripts/ablation.py`
  - `results/oah_ablation.csv`
- Final 2D partition figures:
  - `scripts/export_final_dataset_figures.py`
  - `figures/datasets/oah_plot_xor.png`
  - `figures/datasets/oah_plot_moons.png`
  - `figures/datasets/oah_plot_circles.png`
  - `figures/datasets/oah_plot_gauss_far.png`
  - `figures/datasets/oah_plot_gauss_overlap.png`
  - `figures/datasets/oah_plot_piecewise.png`

## Determinism Notes
- Dataset seeds are fixed inside scripts:
  - `scripts/compare_2d.py`
  - `scripts/compare_real.py`
  - `scripts/export_final_dataset_figures.py`
- Reported values are means/std over repeated stream orderings.
- Minor numeric variation may still occur across platforms and BLAS backends.

## Artifact Checklist
- Public code URL: `https://github.com/jkrawczuk/oah`
- Release tag used by manuscript: `v1.0.3`
- Citation metadata: `CITATION.cff`
- License: MIT
- Dependency lockfile: `poetry.lock`

## Optional archival DOI
For stronger long-term reproducibility, archive release `v1.0.3` in Zenodo and add DOI to paper code-availability text.
