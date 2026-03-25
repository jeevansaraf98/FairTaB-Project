import argparse
from pathlib import Path
import json

import pandas as pd
import pytorch_lightning as pl
import torch

from decaf.DECAF import DECAF
from decaf.data import DataModule

DEFAULT_COLS = [
    "country_birth",
    "cur_eco_activity",
    "edu_level",
    "economic_status",
    "age",
    "occupation",
    "sex",
    "household_size",
    "marital_status",
    "household_position",
    "prev_residence_place",
    "citizenship"
]

BINARY_COLS = ["sex", "prev_residence_place", "occupation"]


def encode_dutch_numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            cats = sorted(out[col].dropna().astype(str).unique().tolist())
            out[col] = pd.Categorical(out[col].astype(str), categories=cats).codes.astype("int64")
    return out


def postprocess_dutch_synth(df_raw: pd.DataFrame, cols_order: list[str]) -> pd.DataFrame:
    df = df_raw.copy()

    for col in BINARY_COLS:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            thr = x.mean(skipna=True)
            df[col] = (x >= thr).astype(int)

    for col in cols_order:
        if col in df.columns and col not in BINARY_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype(int)

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
    parser.add_argument("--csv", default="data/dutch/dutch.csv", help="Path to Dutch CSV")
    parser.add_argument("--cols", nargs="+", default=DEFAULT_COLS,
                        help="Column order (node list) to use for DAG indexing; must match CSV exactly.")
    parser.add_argument("--dag_json", default="tests/dutch/dag_edges_dutch.json",
                        help="JSON with named DAG edges [[u,v],...]")
    parser.add_argument("--biased_json", default="tests/dutch/biased_edges_dutch.json",
                        help="JSON with named edges to cut during generation (empty [] here)")
    parser.add_argument("--h_dim", type=int, default=200)
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

    df = pd.read_csv(args.csv)

    df = encode_dutch_numeric_only(df)

    missing = [c for c in args.cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from CSV: {missing}")
    D = df[args.cols].copy()
    X_np = D.values.astype("float32")

    name2i = {c: i for i, c in enumerate(args.cols)}
    dag_seed = load_named_edges(Path(args.dag_json), name2i)
    bias_dict = to_bias_dict(Path(args.biased_json), name2i)

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
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        logger=False
    )
    trainer.fit(model, dm)

    with torch.no_grad():
        base = dm.dataset.x
        if args.n_samples is not None:
            base = torch.randn(args.n_samples, base.shape[1], dtype=base.dtype, device=base.device)
        synth_data = model.gen_synthetic(base, biased_edges=bias_dict).detach().cpu().numpy()

    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "dutch_decaf_synthetic__raw.csv"
    raw_df = pd.DataFrame(synth_data, columns=args.cols)
    raw_df.to_csv(raw_path, index=False)

    post_df = postprocess_dutch_synth(raw_df, args.cols)
    post_path = out_dir / "dutch_decaf_synthetic_data_postprocessed.csv"
    post_df.to_csv(post_path, index=False)

    print("Data generated successfully!",
          {"shape": synth_data.shape, "raw": str(raw_path), "post": str(post_path)})


if __name__ == "__main__":
    main()
