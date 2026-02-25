📝 Note on Performance: > The current codebase fully reproduces the results reported in our paper. However, due to compute constraints during development, we were unable to perform extensive rerolling or hyperparameter tuning. We believe this architecture has even higher potential, and our team plans to train and publish a newer, optimized version once additional compute resources become available.

````markdown
# T-SAC — Chunking the Critic (ICLR 2026)

Official code release for the ICLR 2026 paper:

**Chunking the Critic: A Transformer-Based Soft Actor-Critic with N-Step Returns**  
Dong Tian*, Onur Celik, Gerhard Neumann (Karlsruhe Institute of Technology, KIT)  
\*Corresponding author: dong.tian@outlook.de

---

## Note on performance / reproducibility

> The current codebase fully reproduces the results reported in our paper. However, due to compute constraints during development, we were unable to perform extensive rerolling or hyperparameter tuning. We believe this architecture has even higher potential, and our team plans to train and publish a newer, optimized version once additional compute resources become available.

**Public test dashboard (paper runs):**  
https://wandb.ai/dt_team/T-SAC%20results?nw=nwusernwzyq

If you are unable to reproduce the paper or Weights & Biases results **after making a best-effort attempt**, please open an issue (preferred) or email—we’re happy to help.

---

## TL;DR

T-SAC strengthens SAC by “chunking” **inside the critic**:

- **Transformer critic (causal):** conditions on short state–action windows and predicts **prefix-conditioned** Q-values.
- **Multi-horizon N-step TD (no importance sampling):** trains on variable-horizon N-step targets.
- **Gradient-level averaging:** averages gradients across horizons (instead of averaging targets) to reduce variance without diluting sparse long-horizon signal.
- **Non-soft critic targets:** entropy is handled on the **policy side** (critic learns standard action-values).
- **Optional target-free training:** replace Polyak target updates with a **hard-copy + critic freezing** schedule (single hyperparameter `K`).

---

## What’s in this repo

- T-SAC implementation (actor/critic, losses, replay, training loop)
- Training + evaluation scripts
- Configs for the paper benchmarks
- Scripts for launching multi-seed runs (local / Slurm)
- Plotting / aggregation utilities (IQM + bootstrap CIs, etc.)

> If you are packaging this as a library, consider adding a minimal API example in `examples/`.

---

## Installation

### 1) Create a Python environment

```bash
conda create -n tsac python=3.10 -y
conda activate tsac
````

### 2) Install dependencies

Choose **one**:

**Option A (recommended, editable install):**

```bash
pip install -U pip
pip install -e .
```

**Option B (requirements file):**

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3) Install benchmark environments (as needed)

Depending on what you want to run:

* **Gymnasium MuJoCo (v4):** `gymnasium[mujoco]` (+ MuJoCo runtime)
* **Meta-World (ML1):** `metaworld`
* **FANCYGYM Box-Pushing:** `fancy_gym`

> Tip: keep benchmark dependencies optional (extras) to make install lighter.

---

## Quickstart

> Replace the commands below with your actual entrypoints (e.g., `python train.py ...`, `python -m tsac.train ...`, Hydra, etc.).

### Train (example)

```bash
python train.py suite=gymnasium env=Walker2d-v4 algo=tsac seed=0 steps=5_000_000
```

### Evaluate (example)

```bash
python eval.py suite=gymnasium env=Walker2d-v4 ckpt=path/to/checkpoint episodes=20
```

### Log to Weights & Biases (example)

```bash
wandb login
python train.py suite=gymnasium env=Walker2d-v4 algo=tsac seed=0 logger=wandb project="T-SAC"
```

---

## Reproducing paper results

### Benchmarks

We report results on:

* **Meta-World ML1 (50 tasks)** — success counted **only at the final timestep**
* **Gymnasium MuJoCo v4** — Ant, HalfCheetah, Hopper, Walker2d, HumanoidStandup
* **FANCYGYM Box-Pushing** — Dense + Sparse variants

### Suggested reproduction commands (edit to match your scripts/configs)

#### Meta-World ML1

```bash
# single task
python train.py suite=metaworld task=assembly algo=tsac seed=0 steps=5_000_000

# sweep all ML1 tasks (example wrapper)
python scripts/run_metaworld_ml1.py algo=tsac seeds=0,1,2,3,4,5,6,7 steps=5_000_000
```

#### FANCYGYM Box-Pushing

```bash
python train.py suite=fancygym task=box_pushing_dense  algo=tsac seed=0 steps=20_000_000
python train.py suite=fancygym task=box_pushing_sparse algo=tsac seed=0 steps=20_000_000
```

#### Gymnasium MuJoCo

```bash
python train.py suite=gymnasium env=Walker2d-v4 algo=tsac seed=0 steps=5_000_000
```

---

## Key hyperparameters (paper defaults)

These are the main knobs to check first if results differ:

### N-step / windowing

* `min_len = 1`
* `max_len = 16` (standard setting)
* sample horizon `n ~ Uniform{min_len, ..., max_len}`
* (optional) collect multiple windows per environment rollout (paper commonly uses 4)

### Transformer critic (lightweight)

* causal self-attention (no future leakage)
* typically **2 attention layers**
* hidden size **128–256** (task-dependent)
* no dropout (recommended)

### Update-to-data ratio (UTD)

* **UTD ≤ 1** (paper uses a maximum UTD of 1)

### Target-network mode vs target-free mode

* **Default:** Polyak target updates (SAC-style)
* **Target-free option:** hard-copy snapshot + cached targets for `K` critic updates

  * MuJoCo setting: `K = 20` is a good default starting point

---

## Repository structure (recommended)

(Adjust names to match your project.)

```text
.
├── tsac/                 # core algorithm (models, losses, replay, utils)
├── configs/              # experiment configs (Hydra/YAML/etc.)
├── scripts/              # launchers (local/Slurm), sweeps, plotting
├── train.py              # training entrypoint
├── eval.py               # evaluation entrypoint
├── README.md
└── LICENSE
```

---

## Troubleshooting

**Install issues (MuJoCo / rendering):**

* verify `gymnasium[mujoco]` works with a minimal rollout
* on headless servers, you may need EGL / OSMesa

**Results don’t match the paper:**

* confirm evaluation protocol (especially Meta-World final-step success)
* confirm step budget, number of seeds, and deterministic evaluation settings
* confirm `min_len/max_len`, UTD, and policy delay / update schedules

When opening an issue, please include:

* command/config used
* git commit hash
* full logs (or W&B link)
* machine + CUDA/driver + package versions

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{tian2026chunking,
  title     = {Chunking the Critic: A Transformer-Based Soft Actor-Critic with N-Step Returns},
  author    = {Tian, Dong and Celik, Onur and Neumann, Gerhard},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Code: https://github.com/DongTian95/T-SAC-Official}
}
```

---

## License

MIT License (see `LICENSE`).

---

## Contact

* Dong Tian: [dong.tian@outlook.de](mailto:dong.tian@outlook.de)
* Please use GitHub issues for bugs/questions when possible.

[1]: https://github.com/DongTian95/T-SAC-Official/tree/main "GitHub - DongTian95/T-SAC-Official: This is the official release of T-SAC (CHUNKING THE CRITIC: A TRANSFORMER-BASED SOFT ACTOR-CRITIC WITH N-STEP RETURNS, ICLR2026)"
