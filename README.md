# FedHera: Towards Drift-Resilient Federated Fine-tuning with Heterogeneous Resources

FedHera is a federated fine-tuning framework for large language models under heterogeneous client resources and heterogeneous tasks. This public repository contains the FedHera code path only.

Paper status: published on 30 April 2026 and accepted as an ICML 2026 regular paper.

## What This Repository Contains

- `main.py`: FedHera training entry point.
- `fed_utils/`: client training, server aggregation, and rank allocation logic.
- `utils/preprocess_fedhera_data.py`: dataset conversion and client partitioning script.
- `utils/setup_evaluate.py`: helper to fetch the local Hugging Face `evaluate` checkout used by the evaluation code.
- `templates/`: prompt templates used to build instruction-tuning examples.

## Environment Setup

Run all commands from the repository root.

Unless explicitly marked as `PowerShell`, command blocks below use Linux or macOS shell syntax. If your Linux environment maps Python 3 to `python3` and `pip3`, use those names instead of `python` and `pip`.

### Option 1: pip

```bash
pip install -r requirements.txt
```

### Option 2: conda

```bash
conda env create -f environment.yml
conda activate fedhera
```

Notes:
- `bitsandbytes` is marked optional on Windows in `requirements.txt`.
- Gated Hugging Face models such as Llama require authentication, e.g. `huggingface-cli login`.
- If you want the optional CIDEr metric during evaluation, install `pycocoevalcap`. If it is unavailable, the code will skip CIDEr and keep the other metrics.

## Evaluation Metric Setup

`fed_utils/client.py` loads ROUGE, BLEU, METEOR, and NIST from a local checkout of Hugging Face `evaluate`.

Recommended setup:

Linux or macOS:

```bash
python3 utils/setup_evaluate.py
```

Windows PowerShell:

```powershell
python utils/setup_evaluate.py
```

This clones `https://github.com/huggingface/evaluate.git` into `./evaluate` and installs it in editable mode.

Manual setup is also fine:

Linux or macOS:

```bash
git clone https://github.com/huggingface/evaluate.git evaluate
pip install -e evaluate
```

Windows PowerShell:

```powershell
git clone https://github.com/huggingface/evaluate.git evaluate
pip install -e .\evaluate
```

If you place the checkout somewhere else, set:

Linux or macOS:

```bash
export FEDHERA_EVALUATE_ROOT=/path/to/evaluate
```

On Windows PowerShell:

```powershell
$env:FEDHERA_EVALUATE_ROOT = 'D:\path\to\evaluate'
```

## Data Preparation

The preprocessing script converts a source dataset into FedHera's JSON format:

```text
<output_root>/<num_clients>/
  local_training_0.json
  local_eval_0.json
  local_test_0.json
  ...
```

Supported `--task` values in the current code are:
- `metamathqa`
- `commonsense`
- `e2e_nlg`
- `gsm8k`
- `svamp`
- `boolq`
- `piqa`
- `hellaswag`
- `alpaca`

You can load data either from Hugging Face Hub with `--hf_dataset` or from a local file with `--data_files`.

The multi-line examples in this section use Linux or macOS line continuation with `\`. On Windows PowerShell, either write each example on a single line or replace `\` with PowerShell's backtick line continuation.

### Example 1: GSM8K from Hugging Face

```bash
python utils/preprocess_fedhera_data.py \
  --task gsm8k \
  --hf_dataset openai/gsm8k \
  --hf_config main \
  --split train \
  --output_root ./data/gsm8k \
  --num_clients 200 \
  --max_examples 10000
```

### Example 2: Alpaca-format local JSON

```bash
python utils/preprocess_fedhera_data.py \
  --task alpaca \
  --data_files /path/to/alpaca_data.json \
  --output_root ./data/alpaca \
  --num_clients 200
```

The local Alpaca-style file should contain records with `instruction`, `input`, and `output` fields.

### Example 3: Non-IID split with Dirichlet partitioning

```bash
python utils/preprocess_fedhera_data.py \
  --task boolq \
  --hf_dataset boolq \
  --split train \
  --output_root ./data/boolq \
  --num_clients 200 \
  --alpha 0.5
```

Important:
- `main.py` expects `--data_path` to point to `./data/<task_name>`, not the nested client-count directory.
- For example, if preprocessing writes into `./data/gsm8k/200/...`, then training should use `--data_path ./data/gsm8k --num_clients 200`.

## Running FedHera

Current code only exposes the `fedhera` aggregation path.

The command blocks in this section are shown in Linux or macOS shell syntax. On Windows PowerShell, you can run the same arguments on one line or replace `\` with backticks.

### Smoke Test Example

```bash
python main.py \
  --global_model <hf_or_local_causal_lm> \
  --data_path ./data/gsm8k \
  --num_clients 200 \
  --num_communication_rounds 3 \
  --client_selection_frac 0.2 \
  --local_num_epochs 1 \
  --local_batch_size 4 \
  --local_micro_batch_size 2 \
  --session_name gsm8k_smoke \
  --use_atw
```

### Paper-Style Example: Unbiased Server Aggregation

```bash
python main.py \
  --global_model <hf_or_local_causal_lm> \
  --data_path ./data/alpaca \
  --num_clients 200 \
  --hetero_mode setting_B \
  --fedhera_server_agg unbiased \
  --use_atw \
  --session_name alpaca_unbiased
```

For best out-of-the-box behavior, use a Llama-, Mistral-, Gemma-, or GPT-2-style causal LM. For other architectures, pass `--lora_target_modules` explicitly.

Useful flags:
- `--hetero_mode {setting_A,setting_B}`: client resource distribution preset.
- `--use_atw`: enable Adaptive Tail Warm-up.
- `--fedhera_server_agg {original,unbiased}`: choose the server aggregation rule.
- `--fedhera_coupled`: force `r_tot == r_main` for all layers.
- `--ablation {uniform,random}`: replace the default water-filling allocator.
- `--profile_system_costs`: log client latency, peak memory, and server SVD timing.
- `--resume_epoch N`: resume from an existing run.

To inspect all available arguments:

Linux or macOS:

```bash
python3 main.py --help
python3 utils/preprocess_fedhera_data.py --help
```

Windows PowerShell:

```powershell
python main.py --help
python utils/preprocess_fedhera_data.py --help
```

## Evaluation Behavior

Evaluation is performed during training on each client's `local_eval_*.json` split through `GeneralClient.test()` in `fed_utils/client.py`.

Metrics used by the current code are:
- token-level accuracy
- ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum
- BLEU
- METEOR
- NIST
- CIDEr if `pycocoevalcap` is installed

## Output Layout

A run with `--session_name <name>` writes artifacts under:

```text
<session_name>/
  logs/
  fedhera/<num_clients>/
```

Inside the FedHera output directory, the main artifacts are:
- per-client local adapter checkpoints under `client_id/local_output_epoch_*`
- server push packages under `client_id/server_push_epoch_*`
- dense FedHera aggregates as `fedhera_dense_global_epoch_*.pt`
- optional `adapter_model.bin` if `--save_model` is enabled

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{xiao2026fedhera,
  title={FedHera: Towards Drift-Resilient Federated Fine-tuning with Heterogeneous Resources},
  author={Xiao, Ke and Wang, Qiyuan and Anagnostopoulos, Christos and Tan, Zhuoran and Li, Wenhao},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

Accepted as an ICML 2026 regular paper. Published: 30 April 2026.

## License

This project is distributed under the Apache-2.0 License. See `LICENSE`.

## Acknowledgement

This implementation builds on ideas and code structure inherited from FederatedScope.
