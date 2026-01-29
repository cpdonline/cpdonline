# CPD Online (Entropy CUSUM) — Reference Implementation

This repository provides a cleaned and modularized **reference implementation**
of CPD Online that reflects the experimental pipeline used in the accompanying paper.
The original research code evolved over several months; to improve clarity and usability,
the codebase was reorganized with AI-assisted refactoring while preserving the underlying
algorithms, data flow, and evaluation logic.

The repository is intended to facilitate inspection, understanding, and extension of the
method

## Quick start (main results)

```bash
# Main detection + locality results for a single model (LLaMA-2-7B)
bash run_detection_timing.sh llama-7b

# Multi-model table (LLaMA-2-7B/13B + Vicuna-7B/13B)
bash run_detection_timing.sh
```

## Setup

**Requirements**
- Python 3.9+ (tested with 3.9/3.10)
- A CUDA-capable GPU is strongly recommended for token-stat extraction and LLaMA Guard timing.

**Install**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Model access**
- Some models (e.g., LLaMA-2 chat weights) require accepting the model license on Hugging Face and setting credentials in your environment.

**Device selection**
- Default device is configured in `config/models.yaml` (`cuda:0`).
- Override per run:
```bash
DEVICE=cuda:1  bash run_detection_timing.sh llama-7b
DEVICE=cpu     bash run_detection_timing.sh llama-7b
DEVICE=auto    bash run_detection_timing.sh llama-7b
```

## Directory layout

- `compute/`: dataset assembly, token statistics, detector scoring, and evaluation
- `CPD/`: CPD Online implementation (`python -m CPD.run_cpd_batch`)
- `visualize/`: paper-ready plotting and export scripts
- `config/`: model names + chat templates
- `data/`: prompt CSVs used by the scripts (small, paper-scale benchmarks)

Generated artifacts (created by the scripts):
- `data/<model>_dataset_<tag>.csv`
- `stats/<model>_<tag>_token_stats.csv`
- `results/changepoints/<...>_pp.csv` and `results/changepoints/<...>_cpd_scan.csv`
- `results/<model>_<tag>_k_<k>/...` (metrics + plots)

## Main benchmark (detection + locality)

Run the end-to-end pipeline (dataset → token stats → PP/WPP → CPD → metrics):

```bash
# LLaMA-2-7B only
bash run_detection_timing.sh llama-7b

# Multi-model table (defaults to LLaMA-2-7B/13B + Vicuna-7B/13B)
bash run_detection_timing.sh
```

To avoid re-running stages when outputs already exist:
```bash
bash run_detection_timing.sh --skip-existing llama-7b
```

## PP-gap ablation

This script uses the bundled `data/benign_mix_ppgap{1,2,3}_800.csv` files (no resampling step).

```bash
bash run_ppgap_ablation.sh
```

## Regenerating sampled datasets (optional)

This bundle includes the final sampled datasets under `data/` for convenience.
The sampling procedure is deterministic for a fixed seed, and the sampling code is included (`compute/sample_openorca_pp.py`), but regenerating the exact sampled datasets from scratch requires building large token-stat CSVs for the benign source pools (TyDiQA/OpenOrca). These token-stat files are not included in the bundle due to size.

### Step 0: Build an adversarial-only base dataset + token stats (LLaMA-2-7B)

```bash
# adversarial-only dataset (no benign prompts)
python compute/prepare_dataset.py --model llama-7b --global_csv "" --out_csv data/llama-7b_adv_only.csv

# token stats for adversarial prompts (used as the PP target distribution)
python compute/compute_token_stats.py \
  --model llama-7b \
  --input_csv data/llama-7b_adv_only.csv \
  --output_csv stats/llama-7b_adv_only_token_stats.csv
```

### Step 1: Build benign source pools and their token stats

If you have network access, you can build pool CSVs via Hugging Face datasets:

```bash
python compute/build_source_pools.py --out-dir data/pools --seed 7
```

Then compute token stats for each pool (these outputs can be multi-GB):

```bash
python compute/compute_token_stats.py --model llama-7b --input_csv data/pools/tydiqa_pool.csv   --output_csv stats/tydiqa_sample_token_stats.csv
python compute/compute_token_stats.py --model llama-7b --input_csv data/pools/openorca_pool.csv --output_csv stats/openorca_sample_token_stats.csv
```

Notes:
- For TyDiQA, `compute_token_stats.py` preserves the `language` column, enabling language filtering during sampling.
- OpenOrca is treated as English by default in the sampling script.

### Step 2: Regenerate PP-gap benign datasets

```bash
bash run_sample_ppgap_datasets.sh
```

This produces:
- `data/benign_mix_ppgap5_700.csv` (historical name; contains 800 rows)
- `data/benign_mix_ppgap1_800.csv`
- `data/benign_mix_ppgap2_800.csv`
- `data/benign_mix_ppgap3_800.csv`

### Step 3: Regenerate the TyDiQA stream for LLaMA Guard gating (optional)

```bash
# Default uses MALICIOUS_RATE=0.04 and produces N=17376 benign prompts for 724 adversarial prompts
bash run_sample_tydiqa_realworld_lg.sh
```

## Locality panel (plots-only)

Given an existing `detection_counts.csv`, render the 2-panel locality plot:

```bash
python visualize/plot_locality_panel.py \
  --counts-csv results/<run_tag>/detection_counts.csv \
  --out-pdf results/<run_tag>/locality_panel_overallf1_fpr10.pdf \
  --windows 5 20 \
  --metric-right fpr10
```

## ROC figure export (vector PDF)

Export paper-ready ROC figures from cached per-prompt score CSVs (no model runs):

```bash
python visualize/export_paper_roc_figures.py \
  --changepoints-dir results/changepoints \
  --out-dir paper_figures
```

## LLaMA Guard baseline + detector gating

1) Assemble the imbalanced dataset (CSV merge only):
```bash
python compute/prepare_dataset.py \
  --model llama-7b \
  --global_csv data/tydiqa_pp_matched_17376_lang_x1p1.csv \
  --autodan_csv data/full_prompt_dataset.csv \
  --out_csv data/llama-7b_dataset_tydiqa_pp17376_lang_x1p1.csv
```

2) Run LLaMA Guard on the imbalanced stream:
```bash
mkdir -p results/llama_guard
./run_llama_guard_robust.sh \
  --input-csv data/llama-7b_dataset_tydiqa_pp17376_lang_x1p1.csv \
  --output-csv results/llama_guard/lg_results.csv \
  --models lg1 lg2 \
  --batch-size 4
```

3) Produce detector scores on the same dataset tag (example uses `k=0`):
```bash
GLOBAL_CSV=data/tydiqa_pp_matched_17376_lang_x1p1.csv \
DATASET_TAG=tydiqa_pp17376_lang_x1p1 \
MODELS="llama-7b" \
K_LIST="0" \
  bash run_detection_timing.sh --skip-existing
```

4) Compute call savings vs. hybrid detection performance:
```bash
python compute/analyze_hybrid_guard_savings.py \
  --guard-csv results/llama_guard/lg_results.csv \
  --guard-version lg2 \
  --cpd-csv results/changepoints/llama-7b_tydiqa_pp17376_lang_x1p1_k_0_cpd_scan.csv \
  --pp-csv  results/changepoints/llama-7b_tydiqa_pp17376_lang_x1p1_pp.csv \
  --output-csv results/hybrid_guard_savings.csv \
  --output-dir results/figures/hybrid_guard
```

## Code provenance and reproducibility notes

- The released code mirrors the experimental structure and algorithms described in the paper,
- Some scripts may require minor environment- or path-specific adjustments depending on hardware,
  model access, and GPU memory.
- The bundled CSV files correspond to the final paper-scale benchmarks used in the experiments.