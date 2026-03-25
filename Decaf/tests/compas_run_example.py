"""
compas_run_example.py
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

DEFAULT_COLS = [
    "age_cat",
    "v_score_text",
    "score_text",
    "priors_count",
    "race",
    "c_charge_degree",
    "sex",
    "juv_crime",
    "two_year_recid"
]

BINARY_COLS = [
    "sex",
    "two_year_recid"
]
NUMERICAL_COLS = [
    "priors_count",
    "juv_crime"
]
CATEGORICAL_COLS = [
    "age_cat",
    "race",
    "v_score_text",
    "score_text",
    "c_charge_degree"
]

JUV_COMPONENTS = ["juv_fel_count", "juv_misd_count", "juv_other_count"]


def ensure_juv_crime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "juv_crime" not in out.columns:
        if all(c in out.columns for c in JUV_COMPONENTS):
            out["juv_crime"] = (
                pd.to_numeric(out[JUV_COMPONENTS[0]], errors="coerce").fillna(0) +
                pd.to_numeric(out[JUV_COMPONENTS[1]], errors="coerce").fillna(0) +
                pd.to_numeric(out[JUV_COMPONENTS[2]], errors="coerce").fillna(0)
            )
        else:
            raise ValueError(
                "juv_crime not found and juvenile component columns missing; "
                f"need columns {JUV_COMPONENTS} to compute it."
            )
    else:
        out["juv_crime"] = pd.to_numeric(out["juv_crime"], errors="coerce")
    return out


def encode_compas_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            cats = sorted(out[col].dropna().astype(str).unique().tolist())
            out[col] = pd.Categorical(out[col].astype(str), categories=cats).codes.astype("int64")
    return out


def postprocess_compas_synth(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Binary
    for col in BINARY_COLS:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            thr = x.mean(skipna=True)
            df[col] = (x >= thr).astype(int)

    # Numerical
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)

    # Categorical
    for col in CATEGORICAL_COLS:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/compas-v/compas-scores-two-years-violent_clean.csv",
                        help="Path to COMPAS CSV")
    parser.add_argument("--cols", nargs="+", default=DEFAULT_COLS,
                        help="Column order (node list) to use for DAG indexing; must match CSV exactly.")
    parser.add_argument("--dag_json", default="tests/compas/dag_edges_compas.json",
                        help="JSON with named DAG edges [[u,v],...]")
    parser.add_argument("--biased_json", default="tests/compas/biased_edges_compas.json",
                        help="JSON with named edges to cut during generation (empty [] if none)")
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=None, help="(Optional) if your DECAF supports it")
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

    # 1) Load + ensure juv_crime
    df = pd.read_csv(args.csv)
    df = ensure_juv_crime(df)

    # 2) Encode non-numeric only
    df = encode_compas_numeric_only(df)

    missing = [c for c in args.cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from CSV: {missing}")
    D = df[args.cols].copy()
    X_np = D.values.astype("float32")

    # 3) Edges & bias dict
    name2i = {c: i for i, c in enumerate(args.cols)}
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

    # 4) Data module & train
    dm = DataModule(X_np.tolist(), batch_size=args.batch_size)
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
    trainer = pl.Trainer(
        accelerator="auto",
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

    raw_path = out_dir / "compas_violent_decaf_synthetic__raw.csv"
    raw = pd.DataFrame(synth_data, columns=args.cols)
    raw.to_csv(raw_path, index=False)

    decoded_df = postprocess_compas_synth(raw)
    out_path = out_dir / "compas_violent_decaf_synthetic_data_postprocessed.csv"
    decoded_df.to_csv(out_path, index=False)

    print("Data generated successfully!",
          {"shape": synth_data.shape, "raw": str(raw_path), "post": str(out_path)})


if __name__ == "__main__":
    main()
