# Repository Guidelines

The purpose of this project is as a benchmark implementation of GPT-2 to use for comparison for other research models.

## Project Structure & Module Organization

Core Python scripts live at the repository root: `model.py` defines GPT, `train.py` runs training/evaluation, `sample.py` generates text, and `bench.py` benchmarks performance. `configurator.py` applies configuration files and CLI overrides. Experiment presets live in `config/`. Dataset preparation scripts live under `data/`, including the general Hugging Face loader in `data/huggingface/prepare.py`. Smoke tests are in `tests/test_smoke.py`; documentation images are in `assets/`. Root notebooks explore scaling and model sizing.

## Build, Test, and Development Commands

Run commands from the repository root. There is no separate build step.

- `uv sync --locked --extra cpu`: install the locked environment using the Python 3.11 selection in `.python-version`.
- `uv run --locked --extra cpu python data/huggingface/prepare.py`: prepare a bounded WikiText sample in `data/hf_smoke/`; initial setup requires dataset/tokenizer downloads.
- `OMP_NUM_THREADS=1 uv run --locked --extra cpu python train.py config/smoke.py`: run exactly two CPU optimizer steps with offline W&B logging.
- `uv run --locked --extra cpu python -m unittest discover -s tests -v`: run integration smoke tests.

Keep the same PyTorch extra on `sync` and `run`; substitute `cu128` for CUDA 12.8. When changing dependencies, update `pyproject.toml` and regenerate `uv.lock` with `uv lock`.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for functions/variables/modules, and `PascalCase` for classes. Match surrounding code and keep changes focused. No formatter or linter is configured. Configuration presets use top-level Python assignments; CLI overrides use `--key=value` with matching types, such as `--compile=False`.

## Testing Guidelines

Tests use standard-library `unittest`; name files `test_*.py` and methods `test_*`. No coverage threshold is configured. Add focused regression checks for changed behavior. Existing tests cover local JSON/Parquet preparation, disjoint splits, W&B metrics, checkpoints, resume, and sampling. Use bounded CPU smoke runs; do not launch full training unless requested.

## Commit & Pull Request Guidelines

History mixes descriptive imperative subjects with occasional `fix:` prefixes; no strict commit format is evident. Use concise subjects describing the change. PRs should explain the problem, resulting behavior, relevant issues, and validation commands/results. Document dependency or configuration changes and identify untested GPU paths.

## Configuration & Generated Files

Supply credentials through `HF_TOKEN` and `WANDB_API_KEY`; keep secrets out of code and commits. Use offline W&B for smoke tests. Keep `.venv/`, caches, token binaries, checkpoints, and run logs untracked according to `.gitignore`.

Use Hugging Face's default shared cache at `~/.cache/huggingface`; do not redirect `HF_HOME` or its dataset/Hub cache variables into this repository. Prepared nanoGPT token files remain under `data/`.

Always download and cache OpenWebText before reading it; do not stream it. The bounded Hugging Face preparation script applies example limits to tokenization after downloading the full OpenWebText source.
