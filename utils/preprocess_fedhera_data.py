import argparse
import json
import os
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from datasets import load_dataset, Dataset


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _partition_data_dirichlet(
    records: List[Dict[str, Any]],
    num_clients: int,
    alpha: float,
    seed: int = 42,
    n_classes: int = 10,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    rng = np.random.default_rng(seed)
    
    print(f"Executing Dirichlet partition with alpha={alpha}...")

    corpus = [r["instruction"] + " " + r["input"] for r in records]
    
    print("Vectorizing text for clustering...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(corpus)
    
    print(f"Clustering data into {n_classes} topics...")
    kmeans = MiniBatchKMeans(n_clusters=n_classes, random_state=seed, batch_size=256)
    labels = kmeans.fit_predict(X)
    
    idxs_by_class = {k: [] for k in range(n_classes)}
    for idx, label in enumerate(labels):
        idxs_by_class[label].append(idx)
        
    min_size = 0
    
    client_data_idxs = [[] for _ in range(num_clients)]
    
    for k in range(n_classes):
        idx_k = idxs_by_class[k]
        rng.shuffle(idx_k)
        
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        
        proportions = np.array([p * (1 if len(idx_k) < num_clients else len(idx_k)) for p in proportions])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        splits = np.split(idx_k, proportions)
        for cid in range(num_clients):
            client_data_idxs[cid].extend(splits[cid].tolist())

    out: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    
    for cid in range(num_clients):
        client_indices = client_data_idxs[cid]
        rng.shuffle(client_indices)
        
        client_records = [records[i] for i in client_indices]
        
        n = len(client_records)
        if n == 0:
            print(f"Warning: Client {cid} received no data!")
            out[cid] = {"train": [], "eval": [], "test": []}
            continue
            
        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)

        train = client_records[:n_train]
        eval_ = client_records[n_train:n_train + n_eval]
        test = client_records[n_train + n_eval:]

        out[cid] = {
            "train": train,
            "eval": eval_,
            "test": test,
        }
        
    return out

def _to_fedhera_example_mathqa(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map a MetaMathQA-style record to (instruction, input, output)."""
    question = (
        example.get("query")
        or example.get("question")
        or example.get("problem")
        or example.get("input")
        or ""
    )
    answer = (
        example.get("response")
        or example.get("solution")
        or example.get("answer")
        or example.get("output")
        or ""
    )
    instruction = (
        "Solve the following math problem and provide the final answer. "
        "Show intermediate reasoning if it helps."
    )
    return {
        "instruction": instruction,
        "input": question,
        "output": answer,
        "category": "MetaMathQA",
    }


def _to_fedhera_example_commonsense(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map a commonsense QA-style record to (instruction, input, output)."""
    context = (
        example.get("context")
        or example.get("passage")
        or example.get("story")
        or example.get("sentence")
        or ""
    )
    question = (
        example.get("question")
        or example.get("query")
        or example.get("input")
        or ""
    )
    # Label / answer field names vary widely across commonsense datasets.
    answer = (
        example.get("answer")
        or example.get("label")
        or example.get("target")
        or example.get("output")
        or ""
    )

    if context:
        inp = f"Context: {context}\nQuestion: {question}"
    else:
        inp = question

    instruction = (
        "Answer the following commonsense reasoning question based on the given context."
    )
    return {
        "instruction": instruction,
        "input": inp,
        "output": answer,
        "category": "Commonsense",
    }


def _to_fedhera_example_e2e(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map an E2E NLG-style record to (instruction, input, output)."""
    mr = (
        example.get("meaning_representation")
        or example.get("mr")
        or example.get("input")
        or ""
    )
    # Different variants store references under different keys.
    ref = (
        example.get("human_reference")
        or example.get("reference")
        or example.get("ref")
        or example.get("output")
        or ""
    )

    instruction = (
        "Generate a fluent natural language description for the given meaning representation."
    )
    return {
        "instruction": instruction,
        "input": mr,
        "output": ref,
        "category": "E2E_NLG",
    }


def _to_fedhera_example_gsm8k(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map GSM8K math word problems to (instruction, input, output)."""
    question = example.get("question") or example.get("input") or ""
    answer = example.get("answer") or example.get("output") or ""
    instruction = (
        "Solve the following math word problem and provide the final numeric answer."
    )
    return {
        "instruction": instruction,
        "input": question,
        "output": answer,
        "category": "GSM8K",
    }


def _to_fedhera_example_svamp(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map SVAMP arithmetic reasoning examples."""
    body = example.get("Body") or example.get("body") or ""
    question = example.get("Question") or example.get("question") or ""
    inp = f"{body}\nQuestion: {question}".strip()
    answer = example.get("Answer") or example.get("answer") or ""
    instruction = "Solve the following arithmetic problem and provide the answer."
    return {
        "instruction": instruction,
        "input": inp,
        "output": answer,
        "category": "SVAMP",
    }


def _to_fedhera_example_boolq(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map BoolQ reading comprehension examples."""
    passage = example.get("passage") or ""
    question = example.get("question") or ""
    inp = f"Passage: {passage}\nQuestion: {question}?"
    raw_answer = example.get("answer")
    if isinstance(raw_answer, str):
        ans_lower = raw_answer.lower()
        answer = "True" if ans_lower in {"true", "yes", "1"} else "False"
    else:
        answer = "True" if raw_answer else "False"
    instruction = "Read the passage and answer the question with True or False."
    return {
        "instruction": instruction,
        "input": inp,
        "output": answer,
        "category": "BoolQ",
    }


def _to_fedhera_example_piqa(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map PIQA physical commonsense examples."""
    goal = example.get("goal") or ""
    sol1 = example.get("sol1") or ""
    sol2 = example.get("sol2") or ""
    label = example.get("label")
    try:
        label = int(label)
    except Exception:
        label = None
    correct = sol1 if label == 0 else sol2 if label == 1 else ""
    inp = f"Goal: {goal}\nSolution 1: {sol1}\nSolution 2: {sol2}"
    instruction = "Given a goal and two solutions, identify the correct one."
    return {
        "instruction": instruction,
        "input": inp,
        "output": correct,
        "category": "PIQA",
    }


def _to_fedhera_example_hellaswag(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map HellaSwag examples to a multiple-choice style prompt."""
    ctx_a = example.get("ctx_a") or ""
    ctx_b = example.get("ctx_b") or ""
    context = (ctx_a + " " + ctx_b).strip()
    endings = example.get("endings") or []
    label = example.get("label")
    try:
        label = int(label)
    except Exception:
        label = None
    correct = endings[label] if label is not None and label < len(endings) else ""
    options = []
    for idx, opt in enumerate(endings):
        letter = chr(ord("A") + idx)
        options.append(f"({letter}) {opt}")
    options_text = "\n".join(options)
    prompt_input = f"Context: {context}\nOptions:\n{options_text}\nChoose the best ending."
    instruction = "Pick the option (A/B/C/D) that best completes the context."
    return {
        "instruction": instruction,
        "input": prompt_input,
        "output": correct,
        "category": "HellaSwag",
    }

def _to_fedhera_example_alpaca(example: Dict[str, Any]) -> Dict[str, Any]:
    """Map Alpaca (Cleaned) examples to (instruction, input, output)."""
    # Alpaca already has 'instruction', 'input', and 'output' fields.
    # We just need to handle cases where input is empty or combine them if needed.
    
    instruction = example.get("instruction") or ""
    inp = example.get("input") or ""
    output = example.get("output") or ""
    
    # Alpaca prompts usually don't need 'input' if it's empty, 
    # but FedHera format often expects separate fields.
    # We will keep them as is, consistent with standard Alpaca formatting.
    
    return {
        "instruction": instruction,
        "input": inp,
        "output": output,
        "category": "Alpaca_Cleaned",
    }


def _split_across_clients(
    records: List[Dict[str, Any]],
    num_clients: int,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """Shuffle and split records into per-client train/eval/test sets."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    per_client = len(records) // num_clients
    remainder = len(records) % num_clients

    out: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    cursor = 0
    for cid in range(num_clients):
        size = per_client + (1 if cid < remainder else 0)
        client_idx = indices[cursor: cursor + size]
        cursor += size
        client_records = [records[i] for i in client_idx]

        n = len(client_records)
        n_train = int(n * train_ratio)
        n_eval = int(n * eval_ratio)
        n_test = n - n_train - n_eval

        train = client_records[:n_train]
        eval_ = client_records[n_train:n_train + n_eval]
        test = client_records[n_train + n_eval:]

        out[cid] = {
            "train": train,
            "eval": eval_,
            "test": test,
        }
    return out


def _save_client_splits(
    splits: Dict[int, Dict[str, List[Dict[str, Any]]]],
    output_root: str,
    num_clients: int,
) -> None:
    base_dir = os.path.join(output_root, str(num_clients))
    _ensure_dir(base_dir)

    for cid, parts in splits.items():
        train_path = os.path.join(base_dir, f"local_training_{cid}.json")
        eval_path = os.path.join(base_dir, f"local_eval_{cid}.json")
        test_path = os.path.join(base_dir, f"local_test_{cid}.json")

        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(parts["train"], f, ensure_ascii=False)
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(parts["eval"], f, ensure_ascii=False)
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(parts["test"], f, ensure_ascii=False)


def _load_source_dataset(task: str, hf_dataset: str | None, data_files: str | None, split: str, hf_config: str | None) -> Dataset:
    """
    Load a source dataset either from Hugging Face Hub (hf_dataset)
    or from local JSON/JSONL/CSV files (data_files).
    """
    if hf_dataset:
        # Some datasets (e.g., openai/gsm8k) require an explicit config ("main"/"socratic").
        load_kwargs = {"split": split}
        if hf_config and str(hf_config).lower() != "none":
            ds = load_dataset(hf_dataset, name=hf_config, **load_kwargs)
        else:
            ds = load_dataset(hf_dataset, **load_kwargs)
    else:
        if data_files is None:
            raise ValueError("Either --hf_dataset or --data_files must be provided.")
        # Let datasets infer the format from extension.
        if data_files.endswith(".csv"):
            ds = load_dataset("csv", data_files=data_files, split="train")
        else:
            ds = load_dataset("json", data_files=data_files, split="train")
    return ds


def preprocess_task(
    task: str,
    hf_dataset: str | None,
    data_files: str | None,
    split: str,
    output_root: str,
    num_clients: int,
    max_examples: int | None,
    seed: int = 42,
    hf_config: str | None = None,
    alpha: float | None = None,
) -> None:
    ds = _load_source_dataset(task, hf_dataset, data_files, split, hf_config)

    if max_examples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(max_examples, len(ds))))

    records: List[Dict[str, Any]] = []
    if task == "metamathqa":
        mapper = _to_fedhera_example_mathqa
    elif task == "commonsense":
        mapper = _to_fedhera_example_commonsense
    elif task == "e2e_nlg":
        mapper = _to_fedhera_example_e2e
    elif task == "gsm8k":
        mapper = _to_fedhera_example_gsm8k
    elif task == "svamp":
        mapper = _to_fedhera_example_svamp
    elif task == "boolq":
        mapper = _to_fedhera_example_boolq
    elif task == "piqa":
        mapper = _to_fedhera_example_piqa
    elif task == "hellaswag":
        mapper = _to_fedhera_example_hellaswag
    elif task == "alpaca":
        mapper = _to_fedhera_example_alpaca
    else:
        raise ValueError(f"Unsupported task: {task}")

    for ex in ds:
        rec = mapper(ex)
        # Ensure we have valid instruction/output for training
        if rec["instruction"] and rec["output"]:
            records.append(rec)

    if alpha is not None and alpha > 0:
        print(f"Partitioning data with Dirichlet alpha={alpha}...")
        splits = _partition_data_dirichlet(records, num_clients=num_clients, alpha=alpha, seed=seed)
    else:
        print("Partitioning data uniformly (IID)...")
        splits = _split_across_clients(records, num_clients=num_clients, seed=seed)
        
    _save_client_splits(splits, output_root=output_root, num_clients=num_clients)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess datasets into Fed-Hera JSON format for multiple clients."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "metamathqa",
            "commonsense",
            "e2e_nlg",
            "gsm8k",
            "svamp",
            "boolq",
            "piqa",
            "hellaswag",
            "alpaca",
        ],
        help="Which task to preprocess.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset name (e.g., meta-math/MetaMathQA).",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default=None,
        help="Optional local data files path (JSON/JSONL/CSV) if not using hf_dataset.",
    )
    parser.add_argument(
        "--hf_config",
        type=str,
        default=None,
        help="Optional dataset config name when loading from Hugging Face (e.g., 'main' for openai/gsm8k).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use when loading from Hugging Face.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Base output directory for this task, e.g., ./data/arithmetic, ./data/commonsense, ./data/nlg.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=20,
        help="Number of federated clients to simulate.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to use (e.g., 10000 for MetaMathQA).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and client splitting.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Dirichlet alpha for non-IID partition. If None, uses uniform IID split. (e.g., 10, 0.5, 0.1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess_task(
        task=args.task,
        hf_dataset=args.hf_dataset,
        data_files=args.data_files,
        split=args.split,
        output_root=args.output_root,
        num_clients=args.num_clients,
        max_examples=args.max_examples,
        seed=args.seed,
        hf_config=args.hf_config,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
