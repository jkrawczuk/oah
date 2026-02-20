# oah

Online Additive Hyperplanes (OAH): code for the model, experiments, and interactive tools.

Release used for the paper artifact: `v1.0.3`

## Contents
- `src/oah` - OAH package (model and datasets)
- `scripts` - experiment and ablation scripts used in the paper
- `results` - CSV summaries used in paper tables
- `figures/datasets` - final dataset figures used in the manuscript

## Environment
- Python: `3.12` (see `pyproject.toml`)
- Dependency management: Poetry

## Quick start
```bash
poetry install
poetry run python -c "from oah import OnlineAdditiveHyperplanes; print('ok')"
```

## Quick reproduce (paper outputs)
Run from repository root:

```bash
poetry run python scripts/compare_2d.py
poetry run python scripts/compare_real.py
poetry run python scripts/ablation.py
poetry run python scripts/export_final_dataset_figures.py
```

Expected outputs:
- `results/oah_baselines_2d.csv`
- `results/oah_baselines_real.csv`
- `results/oah_ablation.csv`
- `figures/datasets/oah_plot_*.png`

## Reproducibility mapping
See `REPRODUCIBILITY.md` for mapping:
- manuscript tables/figures -> script -> output files
- fixed dataset seeds and repeat protocol
- artifact checklist before release

## Citation
Citation metadata is provided in `CITATION.cff`.
