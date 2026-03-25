"""
diabetes_run_example.py
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
    "number_inpatient",
    "change",
    "diabetesMed",
    "encounter_id",
    "number_outpatient",
    "number_emergency",
    "admission_source_id",
    "admission_type_id",
    "patient_nbr",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "time_in_hospital",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "age",
    "race",
    "gender",
    "discharge_disposition_id",
    "diag_1",
    "diag_2",
    "diag_3",
    "insulin",
    "nateglinide",
    "glimepiride",
    "rosiglitazone",
    "glyburide",
    "glipizide",
    "metformin",
    "acarbose",
    "pioglitazone",
    "repaglinide",
    "glyburide-metformin",
    "glipizide-metformin",
    "readmitted",
]

NUMERIC_COUNT_COLS = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

ID_COLS = [
    "encounter_id",
    "patient_nbr",
]

BINARY_COLS = [
    "gender",               
    "glipizide-metformin",  
    "change", 
    "diabetesMed", 
    "readmitted",
]


def encode_diabetes(df: pd.DataFrame, cols: list[str]):

    df = df.copy()

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from CSV: {missing}")
    df = df[cols]

    df_enc = pd.DataFrame(index=df.index)
    cat_maps: dict[str, dict] = {}

    for col in cols:
        if col in NUMERIC_COUNT_COLS:
            df_enc[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        else:
            col_str = df[col].astype(str)
            cats = sorted(col_str.dropna().unique().tolist())
            cat2code = {cat: i for i, cat in enumerate(cats)}
            codes = col_str.map(cat2code).astype("int64")

            df_enc[col] = codes

            code2cat = {v: k for k, v in cat2code.items()}
            cat_maps[col] = {
                "categories": cats,
                "code2cat": code2cat,
            }

    return df_enc, cat_maps


def get_type_buckets(cols: list[str]):
    binary_cols = [c for c in cols if c in BINARY_COLS]
    numeric_cols = [c for c in cols if c in NUMERIC_COUNT_COLS]
    categorical_cols = [c for c in cols if c not in numeric_cols]
    return binary_cols, numeric_cols, categorical_cols


def postprocess_diabetes_synth(
    df_raw: pd.DataFrame,
    binary_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    cat_maps: dict[str, dict],
    numeric_bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    df = df_raw.copy()

    # 1. Binary → {0,1}
    for col in binary_cols:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            thr = x.mean(skipna=True)
            df[col] = (x >= thr).astype(int)

    # 2. Numeric count columns
    for col in numeric_cols:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            if numeric_bounds and col in numeric_bounds:
                lo, hi = numeric_bounds[col]
                x = x.clip(lower=lo, upper=hi)
            df[col] = x.round(0).astype(int)

    # 3. Categorical (multi-class + IDs)
    for col in categorical_cols:
        if col not in binary_cols and col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce").round(0).astype(int)

            if col in cat_maps:
                k = len(cat_maps[col]["categories"])
                x = x.clip(lower=0, upper=k - 1)

            df[col] = x

    return df


def load_named_edges(json_path: Path, name2i: dict) -> list[list[int]]:
    edges_named = json.loads(json_path.read_text())
    return [[name2i[u], name2i[v]] for (u, v) in edges_named]


def to_bias_dict(json_path: Path, name2i: dict) -> dict[int, set[int]]:
    edges_named = json.loads(json_path.read_text())
    out: dict[int, set[int]] = {}
    for (u, v) in edges_named:
        ui, vi = name2i[u], name2i[v]
        out.setdefault(vi, set()).add(ui)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/diabetes/diabetes-clean.csv",
                        help="Path to diabetes CSV")
    parser.add_argument(
        "--cols", nargs="+", default=DEFAULT_COLS,
        help="Column order (node list) to use for DAG indexing; must match CSV exactly."
    )
    parser.add_argument("--dag_json", default="tests/diabetes/dag_edges_diabetes.json",
                        help="JSON with named DAG edges [[u,v],...]")
    parser.add_argument("--biased_json", default="tests/diabetes/biased_edges_diabetes.json",
                        help="JSON with named edges to cut during generation")
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)  # smaller LR helps stability
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

    df = pd.read_csv(args.csv)
    df_enc, cat_maps = encode_diabetes(df, args.cols)

    missing = [c for c in args.cols if c not in df_enc.columns]
    if missing:
        raise ValueError(f"Columns missing from encoded data: {missing}")

    D = df_enc[args.cols].copy().astype("float32")
    X_np = D.values

    col_min = X_np.min(axis=0)
    col_max = X_np.max(axis=0)
    denom = col_max - col_min
    denom[denom == 0] = 1.0
    X_scaled = 2.0 * (X_np - col_min) / denom - 1.0

    binary_cols, numeric_cols, categorical_cols = get_type_buckets(args.cols)
    numeric_bounds = {
        col: (float(col_min[i]), float(col_max[i]))
        for i, col in enumerate(args.cols)
        if col in numeric_cols
    }

    name2i = {c: i for i, c in enumerate(args.cols)}
    dag_seed = load_named_edges(Path(args.dag_json), name2i)
    bias_dict = to_bias_dict(Path(args.biased_json), name2i)

    dm = DataModule(X_scaled.tolist(), batch_size=args.batch_size)
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
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model, dm)

    with torch.no_grad():
        base = dm.dataset.x 
        if args.n_samples is not None:
            base = torch.randn(
                args.n_samples,
                base.shape[1],
                dtype=base.dtype,
                device=base.device,
            )
        synth_scaled = (
            model.gen_synthetic(base, biased_edges=bias_dict)
            .detach()
            .cpu()
            .numpy()
        )

    synth_unscaled = 0.5 * (synth_scaled + 1.0) * denom + col_min

    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "diabetes_decaf_synthetic__raw2.csv"
    raw_df = pd.DataFrame(synth_unscaled, columns=args.cols)
    raw_df.to_csv(raw_path, index=False)

    post_df = postprocess_diabetes_synth(
        raw_df,
        binary_cols=binary_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        cat_maps=cat_maps,
        numeric_bounds=numeric_bounds,
    )
    post_path = out_dir / "diabetes_decaf_synthetic_data_postprocessed2.csv"
    post_df.to_csv(post_path, index=False)

    print(
        "Data generated successfully!",
        {"shape": synth_unscaled.shape, "raw": str(raw_path), "post": str(post_path)},
    )


if __name__ == "__main__":
    main()
