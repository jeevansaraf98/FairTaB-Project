"""
law_run_example.py
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


def encode_law(df: pd.DataFrame, race_col: str = "race", bin_cont: bool = False) -> pd.DataFrame:
    out = df.copy()
    if race_col in out.columns and out[race_col].dtype == object:
        out["race"] = (out[race_col].astype(str) != "White").astype(float)
    for b in ["male", "pass_bar"]:
        if b in out.columns:
            out[b] = out[b].astype(float)
    return out


def load_named_edges(json_path: Path, name2i: dict) -> list:
    if json_path is None:
        return []
    edges_named = json.loads(Path(json_path).read_text())
    return [[name2i[u], name2i[v]] for (u, v) in edges_named]


def to_bias_dict(json_path: Path, name2i: dict) -> dict[int, set[int]]:
    edges_named = json.loads(Path(json_path).read_text())
    out: dict[int, set[int]] = {}
    for (u, v) in edges_named:
        ui, vi = name2i[u], name2i[v]
        out.setdefault(vi, set()).add(ui)
    return out


def postprocess_law_synth(df_raw: pd.DataFrame, df_orig: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    binary_01 = ["pass_bar", "race", "male"]
    binary_12 = ["fulltime"]

    for col in binary_01:
        if col in df.columns and col in df_orig.columns:
            thr = float(df_orig[col].astype(float).mean(skipna=True))
            df[col] = (pd.to_numeric(df[col], errors="coerce") >= thr).astype(int)
    for col in binary_12:
        if col in df.columns and col in df_orig.columns:
            thr = float(df_orig[col].astype(float).mean(skipna=True))
            df[col] = (pd.to_numeric(df[col], errors="coerce") >= thr).astype(int) + 1 

    round_1 = ["decile1b", "decile3", "lsat", "ugpa"]
    round_2 = ["zfygpa", "zgpa"]
    round_0 = ["fam_inc", "tier"]

    def _round_cols(cols, ndigits, to_int=False):
        for c in cols:
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce")
                if to_int:
                    df[c] = np.rint(x).astype("Int64") 
                else:
                    df[c] = x.round(ndigits)

    _round_cols(round_1, 1, to_int=False)
    _round_cols(round_2, 2, to_int=False)
    _round_cols(round_0, 0, to_int=True)

    return df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/law/law_school_clean.csv", help="Path to law_school_clean.csv")
    parser.add_argument(
        "--cols",
        nargs="+",
        default=["race","male","fam_inc","tier","fulltime","lsat","ugpa","decile1b","decile3","zgpa","zfygpa","pass_bar"],
        help="Column order (node list) to use for DAG indexing"
    )
    parser.add_argument("--dag_json", default="tests/law/dag_edges.json",
                        help="Path to JSON file with named DAG edges: [[u,v], ...]")
    parser.add_argument("--biased_json", default="tests/law/biased_edges.json",
                        help="Path to JSON file with named edges to cut during generation: [[u,v], ...]")
    parser.add_argument("--h_dim", type=int, default=200)
    parser.add_argument("--z_dim", type=int, default=None, help="Defaults to x_dim if not set")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--l1_W", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=None, help="If set, generate this many samples (uses random base tensor).")
    parser.add_argument("--bin_continuous", action="store_true", help="If set, bin continuous columns into binary categories per predefined thresholds.")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    df = pd.read_csv(args.csv)
    df = encode_law(df, bin_cont=args.bin_continuous)
    missing = [c for c in args.cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from CSV: {missing}")
    D = df[args.cols].copy()
    X_np = D.values.astype("float32")
    data_tensor = torch.tensor(X_np, dtype=torch.float32)

    if args.n_samples is not None:
        data_tensor = torch.randn(args.n_samples, data_tensor.shape[1], dtype=torch.float32)

    name2i = {c: i for i, c in enumerate(args.cols)}
    def _all_in(names):
        return all(n in name2i for n in names)
    import json as _json
    _dag_edges_named = _json.loads(Path(args.dag_json).read_text())
    _biased_edges_named = _json.loads(Path(args.biased_json).read_text())
    missing_dag = sorted({n for e in _dag_edges_named for n in e if n not in name2i})
    missing_biased = sorted({n for e in _biased_edges_named for n in e if n not in name2i})
    if missing_dag or missing_biased:
        raise ValueError(f"Edge names not found in columns. Missing in DAG: {missing_dag}; Missing in biased: {missing_biased}. Columns seen: {list(name2i.keys())}")
    dag_seed = load_named_edges(Path(args.dag_json), name2i)
    print("DAG SEED: ",dag_seed)
    bias_dict = to_bias_dict(Path(args.biased_json), name2i)
    print("BIAS DICT: ",bias_dict)

    dm = DataModule(X_np.tolist(), batch_size=args.batch_size)

    data_tensor = dm.dataset.x

    x_dim = dm.dims[0]
    z_dim = x_dim 
    lambda_privacy = 0
    lambda_gp = 10
    l1_g = 0
    weight_decay = 1e-2
    grad_dag_loss = False

    model = DECAF(
        input_dim=x_dim,
        dag_seed=dag_seed,
        h_dim=args.h_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_privacy=lambda_privacy,
        lambda_gp=lambda_gp,
        alpha=args.alpha,
        rho=args.rho,
        weight_decay=weight_decay,
        grad_dag_loss=grad_dag_loss,
        l1_g=l1_g,
        l1_W=args.l1_W,
    )

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        max_epochs=args.epochs,
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model, dm)

    with torch.no_grad():
        base = data_tensor
        if args.n_samples is not None:
            base = torch.randn(args.n_samples, base.shape[1], dtype=base.dtype, device=base.device)
        synth_data = model.gen_synthetic(base, biased_edges=bias_dict).detach().cpu().numpy()

    print("Data generated successfully!", {"shape": synth_data.shape})

    out_dir = Path("tests/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"law_decaf_synthetic__raw.csv"
    raw = pd.DataFrame(synth_data, columns=args.cols)
    raw.to_csv(raw_path, index=False)
    decoded_df = postprocess_law_synth(raw, df_orig=df)
    out_path = out_dir / f"law_decaf_synthetic_data_postprocessed.csv"
    decoded_df.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
