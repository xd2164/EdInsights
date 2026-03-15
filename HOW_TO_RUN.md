# How to run community_citation_model

## 1. Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Python 3.9+**
- For full runs: GPU (16GB+), 500GB+ RAM, 2TB+ disk, CUDA 11+
- For **demo only**: a normal desktop is enough (no need for full data).

## 2. Create the environment

From the **project root** (`community_citation_model`):

```bash
cd repro
conda env create -f environment.yml
conda activate citationdynamics
```

**Note:** `environment.yml` is built for **Linux** (pinned packages). On **Windows** conda may fail or take long. Options:

- Use **WSL2** (Windows Subsystem for Linux), clone the repo there, and run all steps inside WSL.
- Or create a minimal env and install only what the demo needs (see “Minimal run (demo, no conda)” below).

## 3. Install the project’s libraries

Still inside `repro`, with `citationdynamics` activated:

```bash
pip install -e ./libs/geocitmodel
pip install -e ./libs/xnet
```

(README says `./repo/libs/` but the correct path is `./libs/` under `repro`.)

## 4. Config

The main Snakemake uses `repro/workflow/config.yaml`.

- **Demo:** Default is fine. `data_dir` is `data`, so data lives under `repro/data/`. Demo data is already at `repro/data/demo/preprocessed/`.
- **Full runs:** Edit `repro/workflow/config.yaml` and set:
  - `data_dir` – base directory for all data (needs a lot of space for full runs).
  - Optionally `supp_data_dir`, `aps_data_dir`, `legcit_data_dir`, `uspto_data_dir` if you use separate paths.

## 5. Run the workflow

From **`repro`** (the directory that contains `Snakefile`):

```bash
cd C:\Users\xiaoxuedu\community_citation_model\repro
conda activate citationdynamics
snakemake --cores 4
```

- Replace `4` with your number of CPU cores.
- By default, `Snakefile` uses `DATA_LIST = ["demo"]`, so only the demo dataset runs.
- Outputs go under `repro/results` and figures under `repro/figs`.

To run **all datasets** (aps, legcitv2, uspto), edit `repro/Snakefile`: set  
`DATA_LIST = ["legcitv2", "aps", "uspto"]`  
and comment out the `DATA_LIST = ["demo"]` line. You must have the corresponding data in `data_dir` (see README for data sources).

## 6. Run only a specific dataset’s preprocessing

Each dataset has its own Snakefile under `repro/workflow/<name>/`:

```bash
cd repro/workflow/aps_small    # or legcit, legcitv2, uspto, etc.
snakemake --cores 4
```

Preprocessing writes into the `data_dir` structure expected by the main workflow.

## 7. Minimal run (demo, no conda env)

If you only want to run the **demo** and prefer not to use the full conda env (e.g. on Windows):

1. Create a venv and install Snakemake and the demo dependencies, e.g.:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install snakemake numpy scipy pandas pyyaml
   pip install -e repro/libs/geocitmodel
   pip install -e repro/libs/xnet
   ```
2. From `repro`, run:
   ```bash
   snakemake --cores 2
   ```
   Some rules may still expect packages from the full `environment.yml` (e.g. PyTorch, graph-tool); in that case, use the conda env or WSL.

## 8. Expected runtime

- **Demo:** On a typical desktop, with 4–10 cores, the README reports ~500 minutes with 4 GPUs and 10 CPUs; without GPU it will be slower.
- **Full workflow** (all datasets): Very long; needs the hardware listed in the README (e.g. 500GB+ RAM, 2TB disk).

## Quick reference

| Step              | Command / location |
|-------------------|--------------------|
| Create env        | `cd repro` → `conda env create -f environment.yml` |
| Activate          | `conda activate citationdynamics` |
| Install libs      | `pip install -e ./libs/geocitmodel` and `./libs/xnet` (from `repro`) |
| Config            | `repro/workflow/config.yaml` (default OK for demo) |
| Run (demo)        | From `repro`: `snakemake --cores N` |
| Results           | `repro/results`, `repro/figs` |
