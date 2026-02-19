# oah

Online Additive Hyperplanes (OAH): code for the model, experiments, and interactive tools.

## Contents
- `src/oah` - OAH package (model and datasets)
- `scripts` - experiment/ablation scripts and interactive GUI used in the paper

## Quick start (Poetry)
```bash
poetry install
poetry run python -c "from oah import OnlineAdditiveHyperplanes; print('ok')"
```

## Run examples
```bash
poetry run python scripts/compare_2d.py --help
poetry run python scripts/compare_real.py --help
poetry run python scripts/ablation.py
poetry run python -m oah.gui
poetry run python -m oah.playground
```

## Reproducibility note
- Use a tagged commit/release for paper results.
- Run all commands from the repository root.
- Main experiment outputs are written under `results/`.
- Plot scripts may also write figures under `scripts/img/` or `figures/`.
