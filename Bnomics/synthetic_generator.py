import numpy as np
import pandas as pd

from synthetic import random_dist, downstream_sampler
from ofunc import cpt

# --- STEP 1: Extract adjacency matrix from srch.BN ---
def get_adjmat_from_bn(bn):
    size = len(bn.pnodes)
    adj = np.zeros((size, size), dtype=int)
    for child, parents in enumerate(bn.pnodes):
        for parent in parents:
            adj[child, parent] = 1
    return adj



def synthesize(srch, sample_size: int = 10000):
    """
    Generate Fair synthetic data using random CPTs based on the learned BN structure.
    """
    # Step 1: Extract structure
    adj_mat = get_adjmat_from_bn(srch.BN)
    names = srch.BN.node_names
    arity = srch.arity

    protected_attrs = ["sex"]
    target_attr = "occupation"
    t_idx = names.index(target_attr)

    print(f"\n🎯 Target: {target_attr} (Index: {t_idx})")
    print(f"👀 Initial parents of {target_attr}: {[names[i] for i in srch.BN.pnodes[t_idx]]}")


    # Step 2: Remove any 1-hop edge connecting protected <-> target via any shared node
    for protected_attr in protected_attrs:
        p_idx = names.index(protected_attr)
        print(f"\n🔍 Checking protected attribute: {protected_attr} (Index: {p_idx})")

        # --- Direct edges (both directions) ---
        if p_idx in srch.BN.pnodes[t_idx]:
            srch.BN.pnodes[t_idx].remove(p_idx)
            print(f"🧹 Removed direct edge {protected_attr} → {target_attr}")
        if t_idx in srch.BN.pnodes[p_idx]:
            srch.BN.pnodes[p_idx].remove(t_idx)
            print(f"🧹 Removed direct edge {target_attr} → {protected_attr}")

        # --- 1-hop shared node edges ---
        for inter_idx, parents in enumerate(srch.BN.pnodes):
            inter_name = names[inter_idx]

            # If inter connects to both protected and target (in any direction)
            connected_to_protected = (
                p_idx in srch.BN.pnodes[inter_idx] or  # protected → inter
                inter_idx in srch.BN.pnodes[p_idx]     # inter → protected
            )
            connected_to_target = (
                t_idx in srch.BN.pnodes[inter_idx] or  # target → inter
                inter_idx in srch.BN.pnodes[t_idx]     # inter → target
            )

            if connected_to_protected and connected_to_target:
                # Remove all edges between intermediate ↔ protected and intermediate ↔ target
                if inter_idx in srch.BN.pnodes[p_idx]:
                    srch.BN.pnodes[p_idx].remove(inter_idx)
                    print(f"🧹 Removed {inter_name} → {protected_attr}")
                if p_idx in srch.BN.pnodes[inter_idx]:
                    srch.BN.pnodes[inter_idx].remove(p_idx)
                    print(f"🧹 Removed {protected_attr} → {inter_name}")
                if inter_idx in srch.BN.pnodes[t_idx]:
                    srch.BN.pnodes[t_idx].remove(inter_idx)
                    print(f"🧹 Removed {inter_name} → {target_attr}")
                if t_idx in srch.BN.pnodes[inter_idx]:
                    srch.BN.pnodes[inter_idx].remove(t_idx)
                    print(f"🧹 Removed {target_attr} → {inter_name}")

    # Step 3: Confirm structure
    print(f"\n🛠️  Final parents of {target_attr}: {[names[i] for i in srch.BN.pnodes[t_idx]]}")

    # Step 4: Rebuild adjacency matrix
    adj_mat = get_adjmat_from_bn(srch.BN)

    # Step 5: Save DOT graph
    srch.dot(filename="dutch_debiased_bn", connected_only=False)
    print("🖨️  Saved modified Bayesian Network to 'pdf'")

    # Step 6: Generate CPTs
    node_prob, cond_prob, _ = random_dist(arity, adj_mat)

    # Step 7: Sample synthetic data
    samples = downstream_sampler(node_prob, cond_prob, arity, adj_mat, sample_size=sample_size)
    df = pd.DataFrame(samples, columns=names)

    print("✅ Synthetic data generated using updated BN structure.")
    return df



def generate_random_synthetic_data(srch, sample_size=10000):
    """
    Generate synthetic data using random CPTs based on the learned BN structure.
    """

    print("\n⚙️ Generating synthetic data using RANDOM CPTs...")

    # Extract structure
    adj_mat = get_adjmat_from_bn(srch.BN)
    arity = srch.arity
    names = srch.BN.node_names

    # Generate random priors and CPTs
    node_prob, cond_prob, _ = random_dist(arity, adj_mat)

    # Sample synthetic data
    samples = downstream_sampler(node_prob, cond_prob, arity, adj_mat, sample_size=sample_size)
    df = pd.DataFrame(samples, columns=names)

    print(f"✅ Done. Generated {sample_size} synthetic rows with random CPTs.")
    return df


