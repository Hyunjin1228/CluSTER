import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _load_pt(pt_path: Path):
    payload = torch.load(pt_path, map_location="cpu")
    grads = payload["badge_grad"]
    if isinstance(grads, torch.Tensor):
        grads = grads.detach().cpu().numpy()
    grads = np.asarray(grads, dtype=np.float32)

    idx = payload.get("indices")
    if idx is not None:
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        idx = np.asarray(idx, dtype=np.int64)
    return grads, idx, payload.get("data_info")


def _load_pt_chunks(chunk_paths):
    grad_parts = []
    idx_parts = []
    data_info_parts = {}

    for p in chunk_paths:
        grads, idx, data_info = _load_pt(p)
        grad_parts.append(grads)
        if idx is not None:
            idx_parts.append(idx)
        if isinstance(data_info, dict):
            for k, v in data_info.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                v = np.asarray(v)
                data_info_parts.setdefault(k, []).append(v)

    grads_all = np.concatenate(grad_parts, axis=0) if len(grad_parts) > 0 else np.zeros((0, 0), dtype=np.float32)
    idx_all = np.concatenate(idx_parts, axis=0) if len(idx_parts) > 0 else None
    data_info_all = None
    if len(data_info_parts) > 0:
        data_info_all = {}
        for k, vs in data_info_parts.items():
            try:
                data_info_all[k] = np.concatenate(vs, axis=0)
            except ValueError:
                # Variable-length/padded batch tensors (e.g. input_ids/labels) may have
                # different sequence length across chunks. Keep them as a list.
                data_info_all[k] = vs
    return grads_all, idx_all, data_info_all


def _load_npz(npz_path: Path):
    data = np.load(npz_path)
    return {
        "sample_idx": np.asarray(data["sample_idx"], dtype=np.int64),
        "cluster_labels": np.asarray(data["cluster_labels"], dtype=np.int64),
        "kept_mask": np.asarray(data["kept_mask"], dtype=bool),
    }


def _align_meta(indices, meta):
    if indices is None:
        n = len(meta["cluster_labels"])
        return meta["cluster_labels"][:n], meta["kept_mask"][:n]
    cluster = meta["cluster_labels"][indices]
    kept = meta["kept_mask"][indices]
    return cluster, kept


def _reduce(grads: np.ndarray, method: str, seed: int):
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(grads)
    if method == "tsne":
        return TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, (len(grads) - 1) // 3)),
        ).fit_transform(grads)
    raise ValueError(f"Unknown method: {method}")


def _compute_centroid_metrics(grads: np.ndarray, clusters: np.ndarray, kept: np.ndarray):
    rows = []
    uniq = np.unique(clusters)
    eps = 1e-12
    for c in uniq:
        m = clusters == c
        mk = m & kept
        n_all = int(np.sum(m))
        n_kept = int(np.sum(mk))
        if n_all == 0:
            continue
        mu_all = grads[m].mean(axis=0)
        if n_kept > 0:
            mu_kept = grads[mk].mean(axis=0)
            l2 = float(np.linalg.norm(mu_all - mu_kept))
            na = float(np.linalg.norm(mu_all))
            nk = float(np.linalg.norm(mu_kept))
            cos = float(np.dot(mu_all, mu_kept) / (max(na * nk, eps)))
        else:
            mu_kept = None
            l2 = None
            cos = None
        rows.append(
            {
                "cluster": int(c),
                "n_all": n_all,
                "n_kept": n_kept,
                "cosine_all_vs_kept": cos,
                "l2_all_vs_kept": l2,
            }
        )
    return rows


def _compute_pairwise_cluster_cosines(grads: np.ndarray, clusters: np.ndarray, kept: np.ndarray):
    eps = 1e-12
    uniq = np.unique(clusters)
    cent_all = {}
    cent_kept = {}
    for c in uniq:
        m = clusters == c
        mk = m & kept
        if np.any(m):
            cent_all[int(c)] = grads[m].mean(axis=0)
        if np.any(mk):
            cent_kept[int(c)] = grads[mk].mean(axis=0)

    def _cos(a, b):
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        return float(np.dot(a, b) / max(na * nb, eps))

    pairs = []
    uniq_i = [int(x) for x in uniq.tolist()]
    for i in range(len(uniq_i)):
        for j in range(i + 1, len(uniq_i)):
            ci, cj = uniq_i[i], uniq_i[j]
            row = {
                "cluster_i": ci,
                "cluster_j": cj,
                "cosine_all": _cos(cent_all[ci], cent_all[cj]) if (ci in cent_all and cj in cent_all) else None,
                "cosine_kept": _cos(cent_kept[ci], cent_kept[cj]) if (ci in cent_kept and cj in cent_kept) else None,
            }
            pairs.append(row)
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Plot BADGE gradients with cluster labels.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pt", type=Path, help="Path to a single .pt file")
    group.add_argument(
        "--pt-dir",
        type=Path,
        help="Directory with chunk files (e.g., rank_0_chunk*.pt or rank_0_chunck*.pt)",
    )
    parser.add_argument("--pt-glob", type=str, default="*.pt", help="Glob inside --pt-dir for chunk files")
    parser.add_argument("--npz", type=Path, required=True, help="Path to rank00_cluster_epochXXXX.npz")
    parser.add_argument("--out", type=Path, required=True, help="Output png path")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=50000, help="Randomly subsample for plotting")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional path to save centroid cosine/L2 metrics")
    args = parser.parse_args()

    if args.pt is not None:
        grads, indices, data_info = _load_pt(args.pt)
    else:
        chunk_paths = sorted(args.pt_dir.glob(args.pt_glob))
        if len(chunk_paths) == 0:
            raise ValueError(
                f"No .pt files found in {args.pt_dir} with glob '{args.pt_glob}'. "
                "Try patterns like 'rank*_chunk*.pt' or 'rank*_chunck*.pt'."
            )
        grads, indices, data_info = _load_pt_chunks(chunk_paths)
        print(f"[INFO] loaded {len(chunk_paths)} pt chunks from {args.pt_dir}")

    meta = _load_npz(args.npz)
    clusters, kept = _align_meta(indices, meta)

    if len(grads) != len(clusters):
        raise ValueError(
            f"Length mismatch: grads={len(grads)} clusters={len(clusters)}. "
            "Check that pt/npz come from the same run+epoch."
        )

    # Compute metrics on full high-dimensional gradients before any plotting subsample.
    centroid_metrics = _compute_centroid_metrics(grads, clusters, kept)
    pairwise_cluster_cos = _compute_pairwise_cluster_cosines(grads, clusters, kept)

    rng = np.random.default_rng(args.seed)
    if args.max_points > 0 and len(grads) > args.max_points:
        choice = rng.choice(len(grads), size=args.max_points, replace=False)
        grads = grads[choice]
        clusters = clusters[choice]
        kept = kept[choice]
        if indices is not None:
            indices = indices[choice]

    xy = _reduce(grads, args.method, args.seed)

    uniq = np.unique(clusters)
    cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
    color_map = {c: cmap(i) for i, c in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(10, 8))
    centroid_rows = []
    for c in uniq:
        m = clusters == c
        mk = m & kept
        md = m & (~kept)
        ax.scatter(
            xy[md, 0],
            xy[md, 1],
            s=26,
            alpha=0.75,
            c=[color_map[c]],
            marker="x",
            linewidths=1.2,
        )
        ax.scatter(
            xy[mk, 0],
            xy[mk, 1],
            s=14,
            alpha=0.65,
            c=[color_map[c]],
            marker="o",
            edgecolors="none",
            label=f"cluster {int(c)}",
        )

        # Centroids in projected 2D space
        cen_all = xy[m].mean(axis=0)
        cen_kept = xy[mk].mean(axis=0) if np.any(mk) else None
        ax.scatter(
            [cen_all[0]],
            [cen_all[1]],
            s=120,
            c=[color_map[c]],
            marker="P",
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
        )
        if cen_kept is not None:
            ax.scatter(
                [cen_kept[0]],
                [cen_kept[1]],
                s=140,
                c=[color_map[c]],
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
            )
            ax.plot(
                [cen_all[0], cen_kept[0]],
                [cen_all[1], cen_kept[1]],
                color=color_map[c],
                linewidth=1.1,
                alpha=0.9,
                zorder=5,
            )
        centroid_rows.append((int(c), cen_all, cen_kept))

    ax.set_title(f"BADGE Gradients ({args.method.upper()})")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.5)
    cluster_legend = ax.legend(loc="upper right", ncol=2, fontsize=8, frameon=True, title="Cluster")
    ax.add_artist(cluster_legend)
    style_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markersize=7, label="kept"),
        Line2D([0], [0], marker="x", color="gray", markersize=7, linewidth=0, markeredgewidth=1.2, label="pruned"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="centroid(all)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="centroid(kept)"),
    ]
    ax.legend(handles=style_handles, loc="lower right", frameon=True, title="Mask")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)

    info_keys = sorted(list(data_info.keys())) if isinstance(data_info, dict) else []
    print(f"[OK] saved plot: {args.out}")
    print(f"[INFO] points={len(grads)}, clusters={len(uniq)}, kept={int(kept.sum())}")
    if indices is not None:
        print(f"[INFO] index range: min={int(indices.min())}, max={int(indices.max())}")
    print(f"[INFO] data_info keys: {info_keys}")
    for row in centroid_metrics:
        c = row["cluster"]
        n_all = row["n_all"]
        n_kept = row["n_kept"]
        cos = row["cosine_all_vs_kept"]
        l2 = row["l2_all_vs_kept"]
        if cos is None:
            print(f"[METRIC] cluster={c} n_all={n_all} n_kept={n_kept} cosine=none l2=none")
        else:
            print(f"[METRIC] cluster={c} n_all={n_all} n_kept={n_kept} cosine={cos:.6f} l2={l2:.6f}")
    for row in pairwise_cluster_cos:
        ci = row["cluster_i"]
        cj = row["cluster_j"]
        ca = row["cosine_all"]
        ck = row["cosine_kept"]
        ca_s = "none" if ca is None else f"{ca:.6f}"
        ck_s = "none" if ck is None else f"{ck:.6f}"
        print(f"[PAIR] ({ci},{cj}) cosine_all={ca_s} cosine_kept={ck_s}")
    for c, cen_all, cen_kept in centroid_rows:
        if cen_kept is None:
            print(f"[CENTROID] cluster={c} all=({cen_all[0]:.4f},{cen_all[1]:.4f}) kept=(none)")
        else:
            shift = float(np.linalg.norm(cen_kept - cen_all))
            print(
                f"[CENTROID] cluster={c} "
                f"all=({cen_all[0]:.4f},{cen_all[1]:.4f}) "
                f"kept=({cen_kept[0]:.4f},{cen_kept[1]:.4f}) "
                f"shift={shift:.4f}"
            )
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_points_full": int(len(clusters)),
            "num_clusters": int(len(np.unique(clusters))),
            "metrics": centroid_metrics,
            "pairwise_cluster_cosine": pairwise_cluster_cos,
        }
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] saved metric report: {args.report_json}")


if __name__ == "__main__":
    main()
