
"""
adult_run_example.py
"""

import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from decaf import DECAF
from decaf.data import DataModule


# ---------- Encoding & post-processing ----------

def encode_adult_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            cats = sorted(out[col].dropna().astype(str).unique().tolist())
            out[col] = pd.Categorical(out[col].astype(str), categories=cats).codes.astype("int64")
    return out


def postprocess_adult_synth(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1) Binary columns
    binary_cols = ["gender", "Class-label"]
    for col in binary_cols:
        if col in df.columns:
            thr = df[col].mean(skipna=True)
            df[col] = (df[col] >= thr).astype(int)

    # 2) Numerical columns (round to 1 decimal)
    numerical_cols = ["age", "hours-per-week", "capital-gain",
                      "capital-loss", "fnlwgt", "educational-num"]
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)

    # 3) Categorical columns (round to 0 decimal, keep as integer codes)
    categorical_cols = ["workclass", "education", "marital-status",
                        "occupation", "relationship", "race", "native-country"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype(int)

    return df


def load_named_edges(json_path: Path, name2i: dict) -> list:
    if json_path is None:
        return []
    edges_named = json.loads(Path(json_path).read_text())
    return [[name2i[u], name2i[v]] for (u, v) in edges_named]


def to_bias_dict(json_path: Path, name2i: dict) -> dict:
    if json_path is None:
        return {}
    edges_named = json.loads(Path(json_path).read_text())
    out = {}
    for (u, v) in edges_named:
        ui, vi = name2i[u], name2i[v]
        out.setdefault(vi, set()).add(ui)
    return out


# ---------- Main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/adult/adult-clean.csv", help="Path to ADULT CSV")
    parser.add_argument(
        "--cols",
        nargs="+",
        default=[
            "capital-gain", "capital-loss", "educational-num", "marital-status",
            "workclass", "education", "occupation", "race", "native-country",
            "Class-label", "relationship", "age", "hours-per-week", "gender",
            "fnlwgt"
        ],
        help="Column order (node list) to use for DAG indexing"
    )
    parser.add_argument("--dag_json", default="tests/adult/dag_edges_adult.json",
                        help="JSON with named DAG edges [[u,v],...]")
    parser.add_argument("--biased_json", default="tests/adult/biased_edges_empty.json",
                        help="JSON with named edges to cut during generation (empty here)")
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=None, help="(Optional) set if your DECAF supports it")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--l1_W", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="If set, generate this many samples from random base.")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # 1) Load + encode
    df = pd.read_csv(args.csv)
    df = encode_adult_numeric_only(df)
    missing = [c for c in args.cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from CSV: {missing}")
    D = df[args.cols].copy()
    X_np = D.values.astype("float32")

    # 2) Indices for edges
    name2i = {c: i for i, c in enumerate(args.cols)}
    # Sanity checks
    _dag_edges_named = json.loads(Path(args.dag_json).read_text())
    _biased_edges_named = json.loads(Path(args.biased_json).read_text())
    missing_dag = sorted({n for e in _dag_edges_named for n in e if n not in name2i})
    missing_biased = sorted({n for e in _biased_edges_named for n in e if n not in name2i})
    if missing_dag or missing_biased:
        raise ValueError(
            f"Edge names not found in columns. Missing in DAG: {missing_dag}; "
            f"Missing in biased: {missing_biased}. Columns seen: {list(name2i.keys())}"
        )
    dag_seed = load_named_edges(Path(args.dag_json), name2i)
    bias_dict = to_bias_dict(Path(args.biased_json), name2i)

    # 3) Data module
    dm = DataModule(X_np.tolist(), batch_size=args.batch_size)

    # 4) Train DECAF
    x_dim = dm.dims[0]
    model = DECAF(
        input_dim=x_dim,
        dag_seed=dag_seed,
        h_dim=args.h_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_privacy=0,
        lambda_gp=10,
        alpha=args.alpha,
        rho=args.rho,
        weight_decay=1e-2,
        grad_dag_loss=False,
        l1_g=0,
        l1_W=args.l1_W,
        # z_dim=args.z_dim, 
    )

    # Trainer config: portable defaults (MPS/CPU/GPU auto)
    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        logger=False
    )
    trainer.fit(model, dm)

    # 5) Generate
    with torch.no_grad():
        base = dm.dataset.x
        if args.n_samples is not None:
            base = torch.randn(args.n_samples, base.shape[1], dtype=base.dtype, device=base.device)
        synth_data = model.gen_synthetic(base, biased_edges=bias_dict).detach().cpu().numpy()

    # 6) Save raw + postprocessed
    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "adult_decaf_synthetic__raw.csv"
    raw = pd.DataFrame(synth_data, columns=args.cols)
    raw.to_csv(raw_path, index=False)

    decoded_df = postprocess_adult_synth(raw)
    out_path = out_dir / "adult_decaf_synthetic_data_postprocessed.csv"
    decoded_df.to_csv(out_path, index=False)

    print("Data generated successfully!",
          {"shape": synth_data.shape, "raw": str(raw_path), "post": str(out_path)})


if __name__ == "__main__":
    main()
