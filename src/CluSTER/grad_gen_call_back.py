from transformers import TrainerCallback
from transformers import Trainer
from CluSTER.custom_sampler import CustomDistributedSampler, InterleavedSpecialSampler
import torch
import copy
import numpy as np
from tqdm import tqdm
from torch.distributed import all_gather_object
import random
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from sklearn.cluster import KMeans 
# from k_means_constrained import KMeansConstrained
import torch.nn.functional as F
import itertools
from torch.utils.data import DataLoader
from sklearn.cluster import SpectralClustering
from collections import defaultdict
import math
import torch.distributed as dist
from torch.utils.data import Subset
import random
import types
from typing import List
import time
import transformers
import gc
import os

def _cuda_cleanup(verbose=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose:
            a = torch.cuda.memory_allocated() / (1024**2)
            r = torch.cuda.memory_reserved() / (1024**2)
            print(f"[CUDA] allocated={a:.1f}MB reserved={r:.1f}MB")

class SamplerEpochSetterCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("in callback of sampler epoch setter")        
        sampler = getattr(self.trainer, "_train_sampler_for_callback", None)
        print("check sampler type: ", type(sampler))
        if hasattr(sampler, "set_epoch"):
            print("callback in epoch begins, epoch: ", state.epoch)
            if int(state.epoch) < state.epoch:
                sampler.set_epoch(int(state.epoch)+1)
            else:
                sampler.set_epoch(int(state.epoch))

class ClusteringAndPruningCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("in callback of sampler epoch setter")        
        sampler = getattr(self.trainer, "_train_sampler_for_callback", None)
        print("check sampler type: ", type(sampler))
        if hasattr(sampler, "set_epoch"):
            print("callback in epoch begins, epoch: ", state.epoch)
            sampler.set_epoch(int(state.epoch))

def compute_gradient_diversity(model, batch):
    with torch.no_grad():
        print("copy model")
        ref_model = copy.deepcopy(model).cpu().eval()  # FSDP에서 안전하게 복사
        grads = []

        for i in range(batch['input_ids'].size(0)):
            sample_batch = {k: v[i:i+1].cpu() for k, v in batch.items()}

            for p in ref_model.parameters():
                p.grad = Non

            output = ref_model(**sample_batch)
            loss = output.loss
            loss.backward()

            grad = torch.cat([
                p.grad.detach().flatten() for p in ref_model.parameters() if p.grad is not None
            ])
            grads.append(grad)

        print("stack of grad")
        grads = torch.stack(grads)
        numerator = torch.sum(grads.pow(2)).item()
        total_grad = torch.sum(grads, dim=0)
        denominator = torch.sum(total_grad.pow(2)).item()

        return numerator / (denominator + 1e-12)

def _pairwise_sqdist_to_centers(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # 각 샘플 i에 대해 ||x_i - c_{label_i}||^2
    diffs = X - centers[labels]
    return np.sum(diffs * diffs, axis=1)

def _interleave_equal_clusters(trimmed_by_cluster):
    # [[i0_0, i0_1, ...], [i1_0, i1_1, ...], ...] -> [i0_0, i1_0, ..., iK-1_0, i0_1, i1_1, ...]
    K = len(trimmed_by_cluster)
    lens = []
    for i in trimmed_by_cluster:
        lens.append(len(i))
    m = max(lens) if K > 0 else 0
#    m = len(trimmed_by_cluster[0]) if K > 0 else 0
    out = []
    for j in range(m):
        for c in range(K):
            if len(trimmed_by_cluster[c]) <= j:
                continue
            out.append(trimmed_by_cluster[c][j])
    return out
    
def _concat_clusters(trimmed_by_cluster):
    out = []
    for cluster in trimmed_by_cluster:
        out.extend(cluster)
    return out

def alpha_fit(before_vec: np.ndarray, after_vec: np.ndarray, eps: float = 1e-12):
    """
    before ≈ α * after 최소제곱 α, 재구성오차, 코사인유사도, R^2-like 반환
    """
    b = np.asarray(before_vec); a = np.asarray(after_vec)
    den = float(np.dot(a, a))
    if den < eps:
        alpha = 0.0
        diff = b
        recon_err = float(np.linalg.norm(diff))
        cos_sim = 0.0
        r2_like = 0.0
    else:
        alpha = float(np.dot(b, a) / den)
        diff = b - alpha * a
        recon_err = float(np.linalg.norm(diff))
        nb = float(np.linalg.norm(b)) + eps
        na = float(np.linalg.norm(a)) + eps
        cos_sim = float(np.dot(b, a) / (nb * na))
        r2_like = float(1.0 - (np.dot(diff, diff) / (nb**2)))
    return alpha, recon_err, cos_sim, r2_like

def _pairwise_sqdist_to_centers(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Return squared distance to the (assigned) center for each sample.
      X: (N,D)
      centers: (K,D)
      labels: (N,)
    -> dists: (N,)
    """
    X = np.asarray(X, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    diff = X - centers[labels]
    return np.einsum("nd,nd->n", diff, diff).astype(np.float32)

def _sqdist_matrix_chunk(Xc: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    squared euclidean distances between Xc (M,D) and centers C (K,D) -> (M,K)
    uses ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    """
    x2 = np.sum(Xc * Xc, axis=1, keepdims=True)        # (M,1)
    c2 = np.sum(C * C, axis=1, keepdims=True).T        # (1,K)
    return x2 + c2 - 2.0 * (Xc @ C.T)                  # (M,K)

def kmeans_soft_cap_fit_predict(
    embeddings: np.ndarray,
    world_size: int,
    seed: int = 0,
    n_init: int = 10,
    max_iter: int = 300,
    eps: float = 0.3,          # cap = ceil((1+eps) * N/K)
    topk: int = 8,             # keep top-k nearest centers per point for greedy assignment
    chunk_size: int = 20000,   # distance computation chunk size
    return_full_dist_to_assigned: bool = True,
):
    """
    Drop-in replacement for:
        km = KMeans(n_clusters=world_size, n_init=10, random_state=seed)
        labels = km.fit_predict(embeddings)
        centers = km.cluster_centers_
        dists = _pairwise_sqdist_to_centers(embeddings, centers, labels)

    but with *soft cap* reassignment:
      - fit KMeans normally to get centers
      - reassign labels with an upper capacity cap=(1+eps)*N/K (not perfectly balanced unless eps is small)
      - compute dists to the (final) assigned center

    Returns:
      labels_final: (N,)
      centers: (K,D)
      dists_final: (N,) squared distance to assigned center
      km: fitted KMeans (from sklearn)
      info: dict with cap/counts/labels_kmeans/dists_kmeans
    """
    X = np.asarray(embeddings, dtype=np.float32)
    N, D = X.shape
    K = int(world_size)
    assert 1 <= K <= N

    # 1) vanilla k-means (so you still "refer" to KMeans object as usual)
    km = KMeans(
        n_clusters=K,
        n_init=n_init,
        random_state=seed,
        max_iter=max_iter,
        init="k-means++",
    )
    labels_kmeans = km.fit_predict(X).astype(np.int32)
    centers = km.cluster_centers_.astype(np.float32)
    dists_kmeans = _pairwise_sqdist_to_centers(X, centers, labels_kmeans)

    # 2) soft-cap reassignment (upper bound only)
    cap = int(np.ceil((1.0 + eps) * (N / K)))
    topk = min(int(topk), K)

    # Precompute topk nearest centers per point (indices + squared dists), chunked for memory.
    nn_idx = np.empty((N, topk), dtype=np.int32)
    nn_dist = np.empty((N, topk), dtype=np.float32)

    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        dist = _sqdist_matrix_chunk(X[s:e], centers)  # (M,K)

        idx = np.argpartition(dist, kth=topk - 1, axis=1)[:, :topk]  # (M,topk)
        dsmall = np.take_along_axis(dist, idx, axis=1)

        order = np.argsort(dsmall, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        dsmall = np.take_along_axis(dsmall, order, axis=1)

        nn_idx[s:e] = idx.astype(np.int32)
        nn_dist[s:e] = dsmall.astype(np.float32)

    # Greedy assignment over (sample, candidate-rank) pairs sorted by distance
    labels_final = -np.ones(N, dtype=np.int32)
    counts = np.zeros(K, dtype=np.int32)

    pairs_i = np.repeat(np.arange(N, dtype=np.int32), topk)
    pairs_r = np.tile(np.arange(topk, dtype=np.int32), N)
    pairs_d = nn_dist[pairs_i, pairs_r]

    order = np.argsort(pairs_d, kind="mergesort")  # stable
    pairs_i = pairs_i[order]
    pairs_r = pairs_r[order]

    unassigned = N
    for i, r in zip(pairs_i, pairs_r):
        if labels_final[i] != -1:
            continue
        j = nn_idx[i, r]
        if counts[j] < cap:
            labels_final[i] = j
            counts[j] += 1
            unassigned -= 1
            if unassigned == 0:
                break

    # Fallback: assign remaining points by full distance search until find a center with remaining capacity
    if unassigned > 0:
        rem = np.where(labels_final == -1)[0]
        for s in range(0, len(rem), chunk_size):
            ids = rem[s:s + chunk_size]
            dist = _sqdist_matrix_chunk(X[ids], centers)  # (M,K)
            sort_idx = np.argsort(dist, axis=1)
            for row, i in enumerate(ids):
                for j in sort_idx[row]:
                    if counts[j] < cap:
                        labels_final[i] = int(j)
                        counts[j] += 1
                        unassigned -= 1
                        break

    # Last resort: if cap too strict (eps too small), drop cap for leftovers
    if unassigned > 0:
        rem = np.where(labels_final == -1)[0]
        dist = _sqdist_matrix_chunk(X[rem], centers)
        labels_final[rem] = np.argmin(dist, axis=1).astype(np.int32)

    # 3) dists w.r.t. FINAL labels
    dists_final = _pairwise_sqdist_to_centers(X, centers, labels_final) if return_full_dist_to_assigned else None

    info = {
        "cap": cap,
        "counts_final": counts,
        "labels_kmeans": labels_kmeans,
        "dists_kmeans": dists_kmeans,
    }
    return labels_final, centers, dists_final, km, info

def _kmeans_prune_equalize_interleave(
    embeddings: np.ndarray,
    world_size: int,
    micro: int,
    drop_last: bool,
    seed: int = 42,
    weight_mode: str = "inv",         # "inv" | "sqrt_inv" | "prop" | "none"
    normalize_weights: bool = True,   # 평균 1로 정규화(권장)
    prune_type: str = "far",
    sampling: str = "interleaved",  # "interleaved" | "concat"
    ratio: int = 100,   # 1~100, pruning ratio (% )
):
    N = embeddings.shape[0]
    if world_size <= 1 or N == 0:
        ratios = [1.0]
        w_per_idx = np.ones((N,), dtype=np.float32)
        return list(range(N)), {
            "kept": N, "pruned": 0, "min_cluster_size": N, "m_adj": N,
            "cluster_sizes": [N], "cluster_ratios": ratios,
            "cluster_weights": [1.0], "weight_mean": 1.0,
            "labels": np.zeros((N,), dtype=np.int64),
        }, w_per_idx

    km = KMeans(n_clusters=world_size, n_init=10, random_state=seed)
    labels = km.fit_predict(embeddings)                # (N,)
    centers = km.cluster_centers_
    dists = _pairwise_sqdist_to_centers(embeddings, centers, labels)

    # 클러스터별 정렬(centroid 가까운→먼)
    idxs_by_cluster = [[] for _ in range(world_size)]
    for i, c in enumerate(labels):
        idxs_by_cluster[c].append(i)
    for c in range(world_size):
        idxs_by_cluster[c].sort(key=lambda i: dists[i])

    sizes = [len(lst) for lst in idxs_by_cluster]

    if max(sizes) > sum(sizes) * 1.5 / world_size:
        labels, centers, dists, km, info = kmeans_soft_cap_fit_predict(
            embeddings, world_size=world_size, seed=seed, n_init=10, eps=0.5, topk=8
        )
        idxs_by_cluster = [[] for _ in range(world_size)]
        for i, c in enumerate(labels):
            idxs_by_cluster[c].append(i)
        for c in range(world_size):
            idxs_by_cluster[c].sort(key=lambda i: dists[i])
        sizes = [len(lst) for lst in idxs_by_cluster]

    print("*** size of each cluster:", sizes)
    m = min(sizes)
    sizes_ratio = [size / m for size in sizes]
    print("*** size ratios:", sizes_ratio)
    print("*** min cluster size:", m)
    pruned_ratio = m * ratio // 100
    m_adj = (pruned_ratio // micro) * micro if drop_last else pruned_ratio

    # ---- (A) 프루닝: 가까운 m_adj개만 유지 ----
    if prune_type == "far":
        print("prune far!!!!")
        trimmed = [lst[:m_adj] for lst in idxs_by_cluster]
    elif prune_type == "close":
        print("prune close!!!!")
        trimmed = [lst[-m_adj:] for lst in idxs_by_cluster]
    elif prune_type == "rand":
        print("prune rand!!!!")
        rng = random.Random(seed)
        trimmed = []
        for lst in idxs_by_cluster:
            lst_copy = lst[:]
            rng.shuffle(lst_copy)
            trimmed.append(lst_copy[:m_adj])
    elif prune_type == "mix":
        trimmed = [lst[:m_adj//2] + lst[-(m_adj - m_adj//2):] for lst in idxs_by_cluster]
    elif prune_type == "none":
        trimmed = [lst[:] for lst in idxs_by_cluster]
    elif prune_type == "big":
        grad_norms = np.linalg.norm(embeddings, axis=1)  # (N,)
        trimmed = []
        for lst in idxs_by_cluster:
            # 해당 클러스터 내에서 gradient norm 기준 오름차순 정렬
            lst_sorted = sorted(lst, key=lambda i: grad_norms[i])
            # 가장 큰 norm 기준으로 m_adj 개 선택
            trimmed.append(lst_sorted[-m_adj:])
    else:
        raise ValueError(f"unknown prune_type: {prune_type}")

    for idx, chunk in enumerate(trimmed):     # shuffle은 in-place라 루프가 정석
        rng = random.Random(seed + idx)
        rng.shuffle(chunk)
    if sampling == "interleaved":
        interleaved = _interleave_equal_clusters(trimmed)
    elif sampling == "seq":
        interleaved = _concat_clusters(trimmed)  # 디버깅용
    elif sampling == "rand":
        #interleaved = list(range(len(trimmed)))
        interleaved = _concat_clusters(trimmed)
        rng = random.Random(seed)
        rng.shuffle(interleaved)
    else:
        raise ValueError(f"unknown sampling: {sampling}")
    kept = world_size * m_adj

    # cluster_centroids_before = np.stack(
    #     [X[np.array(idxs_by_cluster[c])].mean(axis=0) for c in range(world_size)]
    # )
    # cluster_centroids_after = np.stack(
    #     [X[np.array(trimmed[c])].mean(axis=0) for c in range(world_size)]
    # )
    # cluster_shift_l2 = np.linalg.norm(cluster_centroids_after - cluster_centroids_before, axis=1)

    # alphas, errs, cos_sims, r2s = [], [], [], []

    # for c in range(world_size):
    #     a, e, cs, r2 = alpha_fit(cluster_centroids_before[c], cluster_centroids_after[c])
    #     alphas.append(a); errs.append(e); cos_sims.append(cs); r2s.append(r2)
    # print(f"[Per-cluster] shift L2 by cluster: {cluster_shift_l2.tolist()} (mean={cluster_shift_l2.mean():.6f})")


    # ---- (B) 가중치: 프루닝 "전" 사이즈로 계산 ----
    #   ratios_k = n_k / N
    #   weight_k: 모드 선택
    ratios = [n / float(N) for n in sizes]
    if prune_type == "none":
        cluster_w = [1.0 for _ in sizes]
    elif weight_mode == "none":
        cluster_w = [1.0 for _ in sizes]
    elif weight_mode == "inv":
        cluster_w = [1.0 / max(n, 1) for n in sizes]
    elif weight_mode == "sqrt_inv":
        cluster_w = [1.0 / max(n, 1)**0.5 for n in sizes]
    elif weight_mode == "prop":
        cluster_w = ratios[:]
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")
    # cluster_w = a
    # print("weights: ", a, "cos_sims: ", cos_sims)

    # 인덱스별 가중치 벡터
    w_per_idx = np.asarray([cluster_w[labels[i]] for i in range(N)], dtype=np.float32)

    # 평균 1로 정규화(권장: 학습 스케일 안정)
    if normalize_weights and kept > 0:
        mean_w = w_per_idx[interleaved].mean()
        if mean_w > 0:
            w_per_idx /= mean_w
    else:
        mean_w = float(w_per_idx.mean()) if N > 0 else 1.0

    diag = {
        "cluster_sizes": sizes,
        "cluster_ratios": ratios,
        "cluster_weights": cluster_w,
        "weight_mean": mean_w,
        "min_cluster_size": m,
        "m_adj": m_adj,
        "kept": kept,
        "pruned": N - kept,
        "labels": labels,    # 필요 시 디버깅/로그용
        "cluster_size_ratio": sizes_ratio,
    }
    return interleaved, diag, w_per_idx

def _badge_kcenter_greedy(cluster_embs, m_adj, seed=42):
    """
    BADGE-style farthest-first selection inside a single cluster.
    cluster_embs: np.ndarray [Nc, D]
    return: list of local indices (0~Nc-1) of selected elements
    """
    Nc = cluster_embs.shape[0]
    if m_adj >= Nc:
        return list(range(Nc))

    rng = np.random.default_rng(seed)

    # Normalize embeddings (optional but usually helpful)
    X = cluster_embs.astype(np.float16) ## float type changed!!
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # ---- 1) First point = max-norm (BADGE style) ----
    norms = np.linalg.norm(X, axis=1)
    first_idx = int(np.argmax(norms))
    selected = [first_idx]

    # Precompute distances
    dist_min = np.sum((X - X[first_idx])**2, axis=1)  # [Nc]
    dist_min[first_idx] = 0.0

    # ---- 2) Farthest-first selection ----
    for _ in range(1, m_adj):
        next_idx = int(np.argmax(dist_min))
        selected.append(next_idx)

        dist_new = np.sum((X - X[next_idx])**2, axis=1)
        dist_min = np.minimum(dist_min, dist_new)
        dist_min[next_idx] = 0.0

    return selected

def _kmeans_badge_equalize_interleave(
    embeddings: np.ndarray,
    world_size: int,
    micro: int,
    drop_last: bool,
    seed: int = 42,
    weight_mode: str = "inv",         # "inv" | "sqrt_inv" | "prop"
    normalize_weights: bool = True,   # 평균 1로 정규화(권장)
    prune_type: str = "far",
    sampling: str = "interleaved",  # "interleaved" | "concat"
    ratio: int = 100,   # 1~100, pruning ratio (% )
):
    N = embeddings.shape[0]
    if world_size <= 1 or N == 0:
        ratios = [1.0]
        w_per_idx = np.ones((N,), dtype=np.float32)
        return list(range(N)), {
            "kept": N, "pruned": 0, "min_cluster_size": N, "m_adj": N,
            "cluster_sizes": [N], "cluster_ratios": ratios,
            "cluster_weights": [1.0], "weight_mean": 1.0,
            "labels": np.zeros((N,), dtype=np.int64),
        }, w_per_idx

    km = KMeans(n_clusters=world_size, n_init=10, random_state=seed)
    labels = km.fit_predict(embeddings)                # (N,)
    centers = km.cluster_centers_
    dists = _pairwise_sqdist_to_centers(embeddings, centers, labels)

    # 클러스터별 정렬(centroid 가까운→먼)
    idxs_by_cluster = [[] for _ in range(world_size)]
    for i, c in enumerate(labels):
        idxs_by_cluster[c].append(i)
    for c in range(world_size):
        idxs_by_cluster[c].sort(key=lambda i: dists[i])

    sizes = [len(lst) for lst in idxs_by_cluster]
    print("*** size of each cluster:", sizes)
    m = min(sizes)
    sizes_ratio = [size / m for size in sizes]
    print("*** size ratios:", sizes_ratio)
    print("*** min cluster size:", m)
    pruned_ratio = m * ratio // 100
    # raw_m_adj = max(1, int(m * ratio // 100))   # 최소 1
    # m_adj = min(raw_m_adj, m)   
    m_adj = (pruned_ratio // micro) * micro if drop_last else pruned_ratio

    # ---- (A) 프루닝: 가까운 m_adj개만 유지 ----
    if prune_type == "far":
        trimmed = [lst[:m_adj] for lst in idxs_by_cluster]
    elif prune_type == "close":
        trimmed = [lst[-m_adj:] for lst in idxs_by_cluster]
    elif prune_type == "rand":
        rng = random.Random(seed)
        trimmed = []
        for lst in idxs_by_cluster:
            lst_copy = lst[:]
            rng.shuffle(lst_copy)
            trimmed.append(lst_copy[:m_adj])
    elif prune_type == "mix":
        trimmed = [lst[:m_adj//2] + lst[-(m_adj - m_adj//2):] for lst in idxs_by_cluster]
    elif prune_type == "none":
        trimmed = [lst[:] for lst in idxs_by_cluster]
    elif prune_type == "badge":
        # ---- (NEW) BADGE-style K-center greedy inside each cluster ----
        print("*** BADGE-style K-center greedy pruning inside each cluster ***")
        trimmed = []
        for c, lst in enumerate(idxs_by_cluster):
            cluster_embs = embeddings[lst]  # subset of embeddings belonging to cluster c
            local_selected = _badge_kcenter_greedy(cluster_embs, m_adj, seed + c)

            # local indices → global indices로 매핑
            global_selected = [lst[i] for i in local_selected]
            trimmed.append(global_selected)
        print("*** done BADGE-style pruning ***")
    else:
        raise ValueError(f"unknown prune_type: {prune_type}")

    for idx, chunk in enumerate(trimmed):     # shuffle은 in-place라 루프가 정석
        rng = random.Random(seed + idx)
        rng.shuffle(chunk)

    interleaved = _concat_clusters(trimmed)
    rng = random.Random(seed) ## 0223
    rng.shuffle(interleaved) ## 0223
    kept = world_size * m_adj

    # ---- (B) 가중치: 프루닝 "전" 사이즈로 계산 ----
    ratios = [n / float(N) for n in sizes]
    if prune_type == "none":
        cluster_w = [1.0 for _ in sizes]
    elif weight_mode == "none":
        cluster_w = [1.0 for _ in sizes]
    elif weight_mode == "inv":
        cluster_w = [1.0 / max(n, 1) for n in sizes]
    elif weight_mode == "sqrt_inv":
        cluster_w = [1.0 / max(n, 1)**0.5 for n in sizes]
    elif weight_mode == "prop":
        cluster_w = ratios[:]  
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")


    # 인덱스별 가중치 벡터
    w_per_idx = np.asarray([cluster_w[labels[i]] for i in range(N)], dtype=np.float32)

    # 평균 1로 정규화(권장: 학습 스케일 안정)
    if normalize_weights and kept > 0:
        mean_w = w_per_idx[interleaved].mean()
        if mean_w > 0:
            w_per_idx /= mean_w
    else:
        mean_w = float(w_per_idx.mean()) if N > 0 else 1.0

    diag = {
        "cluster_sizes": sizes,
        "cluster_ratios": ratios,
        "cluster_weights": cluster_w,
        "weight_mean": mean_w,
        "min_cluster_size": m,
        "m_adj": m_adj,
        "kept": kept,
        "pruned": N - kept,
        "labels": labels,    # 필요 시 디버깅/로그용
        "cluster_size_ratio": sizes_ratio,
    }
    return interleaved, diag, w_per_idx

def _kmeans_uniform_equalize_interleave(
    embeddings: np.ndarray,
    world_size: int,
    micro: int,
    drop_last: bool,
    seed: int = 42,
    weight_mode: str = "inv",         # "inv" | "sqrt_inv" | "prop"
    normalize_weights: bool = True,   # 평균 1로 정규화(권장)
    prune_type: str = "far",
    sampling: str = "interleaved",  # "interleaved" | "concat"
    ratio: int = 100,   # 1~100, pruning ratio (% )
):
    N = embeddings.shape[0]
    if world_size <= 1 or N == 0:
        ratios = [1.0]
        w_per_idx = np.ones((N,), dtype=np.float32)
        return list(range(N)), {
            "kept": N, "pruned": 0, "min_cluster_size": N, "m_adj": N,
            "cluster_sizes": [N], "cluster_ratios": ratios,
            "cluster_weights": [1.0], "weight_mean": 1.0,
            "labels": np.zeros((N,), dtype=np.int64),
        }, w_per_idx

    km = KMeans(n_clusters=world_size, n_init=10, random_state=seed)
    labels = km.fit_predict(embeddings)                # (N,)
    centers = km.cluster_centers_
    dists = _pairwise_sqdist_to_centers(embeddings, centers, labels)

    # 클러스터별 정렬(centroid 가까운→먼)
    idxs_by_cluster = [[] for _ in range(world_size)]
    for i, c in enumerate(labels):
        idxs_by_cluster[c].append(i)
    for c in range(world_size):
        idxs_by_cluster[c].sort(key=lambda i: dists[i])

    sizes = [len(lst) for lst in idxs_by_cluster]
    print("*** size of each cluster:", sizes)

    m = max(sizes)  # 최대 클러스터 크기
    print("*** max cluster size:", m)

    sizes_ratio = [size / m for size in sizes]
    print("*** size ratios:", sizes_ratio)

    # max cluster size 기준으로 ratio 적용
    target = m * ratio // 100
    m_adj = (m // micro) * micro if drop_last else m
    print("*** target size per cluster (after ratio & micro):", m_adj)

    # ---- (B) 업샘플링: 각 클러스터를 m_adj까지 랜덤 복제해서 맞추기 ----
    rng = random.Random(seed)
    balanced = []

    for lst in idxs_by_cluster:
        if not lst:
            balanced.append([])
            continue

        base = lst[:]
        rng.shuffle(base)

        curr_len = len(base)
        if curr_len < m_adj:
            # 부족한 만큼 랜덤 복제 (with replacement)
            needed = m_adj - curr_len
            extra = [rng.choice(base) for _ in range(needed)]
            new_lst = base + extra
        elif curr_len > m_adj:
            # 너무 크면 앞에서 m_adj개만 사용 (원하면 random.sample로 변경 가능)
            new_lst = base[:m_adj]
        else:
            new_lst = base

        balanced.append(new_lst)

    # balanced가 이제 모든 클러스터가 동일한 길이(m_adj)를 갖도록 upsample된 결과
    print("*** balanced cluster sizes:", [len(lst) for lst in balanced])

    for idx, chunk in enumerate(balanced):     # shuffle은 in-place라 루프가 정석
        rng = random.Random(seed + idx)
        rng.shuffle(chunk)
    if sampling == "interleaved":
        interleaved = _interleave_equal_clusters(balanced)
    elif sampling == "seq":
        interleaved = _concat_clusters(balanced)  # 디버깅용
    elif sampling == "rand":
        #interleaved = list(range(len(balanced)))
        interleaved = _concat_clusters(balanced)
        rng = random.Random(seed)
        rng.shuffle(interleaved)
    else:
        raise ValueError(f"unknown sampling: {sampling}")
    kept = world_size * m_adj

    # ---- (B) 가중치: 프루닝 "전" 사이즈로 계산 ----
    #   ratios_k = n_k / N
    #   weight_k: 모드 선택
    ratios = [n / float(N) for n in sizes]
    cluster_w = [1.0 for _ in sizes]

    w_per_idx = np.asarray([cluster_w[labels[i]] for i in range(N)], dtype=np.float32)

    # 평균 1로 정규화(권장: 학습 스케일 안정)
    if normalize_weights and kept > 0:
        mean_w = w_per_idx[interleaved].mean()
        if mean_w > 0:
            w_per_idx /= mean_w
    else:
        mean_w = float(w_per_idx.mean()) if N > 0 else 1.0

    diag = {
        "cluster_sizes": sizes,
        "cluster_ratios": ratios,
        "cluster_weights": cluster_w,
        "weight_mean": mean_w,
        "min_cluster_size": m,
        "m_adj": m_adj,
        "kept": kept,
        "pruned": N - kept,
        "labels": labels,    # 필요 시 디버깅/로그용
        "cluster_size_ratio": sizes_ratio,
    }
    return interleaved, diag, w_per_idx



class SamplerEpochSetterCallback(TrainerCallback):
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("in callback of sampler epoch setter")        
        sampler = getattr(self.trainer, "_train_sampler_for_callback", None)
        print("check sampler type: ", type(sampler))
        if hasattr(sampler, "set_epoch"):
            print("callback in epoch begins, epoch: ", state.epoch)
            if int(state.epoch) < state.epoch:
                sampler.set_epoch(int(state.epoch)+1)
            else:
                sampler.set_epoch(int(state.epoch))

class PruneAndClusteringinwithMeanCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, dataset, remove_mean: bool = True):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.remove_mean = remove_mean
        # 0 means "off": avoid expensive gc/empty_cache per batch by default.
        self.badge_cleanup_interval = int(getattr(self.trainer.args, "badge_cleanup_interval", 0))
        self._badge_cleanup_step = 0
        self.save_badge_grads = bool(getattr(self.trainer.args, "save_badge_grads", False))
        self.badge_grad_save_dir = getattr(self.trainer.args, "badge_grad_save_dir", None)
        self._grad_save_error_reported = False
        self._saved_grad_batches = []
        self._saved_indices = []
        self._saved_data_info = defaultdict(list)

    def _reset_saved_grad_buffers(self):
        self._saved_grad_batches.clear()
        self._saved_indices.clear()
        self._saved_data_info.clear()

    def _maybe_save_grad_batch(self, grad_batch_cpu: torch.Tensor, sample_indices=None, data_batch_cpu=None):
        if not self.save_badge_grads:
            return
        self._saved_grad_batches.append(grad_batch_cpu)
        if sample_indices is not None:
            self._saved_indices.append(torch.as_tensor(sample_indices, dtype=torch.long))
        if data_batch_cpu is not None:
            for k, v in data_batch_cpu.items():
                self._saved_data_info[k].append(v)

    def _flush_saved_grad_file(self, epoch_value):
        if not self.save_badge_grads:
            return
        if len(self._saved_grad_batches) == 0:
            return
        save_dir = self.badge_grad_save_dir or os.path.join(self.trainer.args.output_dir, "badge_grads")
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        try:
            os.makedirs(save_dir, exist_ok=True)
            epoch_int = int(epoch_value) if epoch_value is not None else 0
            payload = {
                "badge_grad": torch.cat(self._saved_grad_batches, dim=0),
                "indices": (
                    torch.cat(self._saved_indices, dim=0)
                    if len(self._saved_indices) > 0
                    else None
                ),
                "data_info": {
                    k: torch.cat(vs, dim=0) for k, vs in self._saved_data_info.items() if len(vs) > 0
                },
            }
            file_path = os.path.join(save_dir, f"rank{rank:02d}_badge_epoch{epoch_int:04d}.pt")
            torch.save(payload, file_path)
        except Exception as e:
            if not self._grad_save_error_reported:
                print(f"[WARN] Failed to save BADGE gradients to {save_dir}: {e}")
                self._grad_save_error_reported = True
        finally:
            self._reset_saved_grad_buffers()

    def _maybe_save_cluster_artifacts(self, diag: dict, interleaved: list, total_n: int, epoch_value):
        if not self.save_badge_grads:
            return
        save_dir = self.badge_grad_save_dir or os.path.join(self.trainer.args.output_dir, "badge_grads")
        try:
            os.makedirs(save_dir, exist_ok=True)
            labels = np.asarray(diag.get("labels", np.zeros((total_n,), dtype=np.int64)), dtype=np.int64)
            kept_idx = np.asarray(interleaved, dtype=np.int64)
            kept_mask = np.zeros((total_n,), dtype=np.bool_)
            if kept_idx.size > 0:
                kept_mask[kept_idx] = True
            epoch_int = int(epoch_value) if epoch_value is not None else 0
            np.savez(
                os.path.join(save_dir, f"rank00_cluster_epoch{epoch_int:04d}.npz"),
                sample_idx=np.arange(total_n, dtype=np.int64),
                cluster_labels=labels,
                kept_mask=kept_mask,
                interleaved_order=kept_idx,
                cluster_sizes=np.asarray(diag.get("cluster_sizes", []), dtype=np.int64),
            )
        except Exception as e:
            if not self._grad_save_error_reported:
                print(f"[WARN] Failed to save cluster artifacts to {save_dir}: {e}")
                self._grad_save_error_reported = True

    # ------------------------- PADDED COLLATE ------------------------- #
    def _get_collate_fn(self):
        # 1) 학습과 동일한 collate 사용
        collate_fn = getattr(self.trainer, "data_collator", None)
        if collate_fn is not None:
            return collate_fn

        # 2) tokenizer.pad 기반 동적 패딩 collate
        tok = self.tokenizer
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token_id = tok.eos_token_id

        def collate_fn(features):
            # feature 값이 텐서일 수도 있어 tokenizer.pad가 리스트를 기대하므로 tolist 처리
            proc = []
            for ex in features:
                ex2 = {}
                for k, v in ex.items():
                    if isinstance(v, torch.Tensor):
                        ex2[k] = v.tolist()
                    else:
                        ex2[k] = v
                proc.append(ex2)
            batch = tok.pad(proc, return_tensors="pt", padding="longest")
            # labels 없으면 생성: LM 학습과 동일하게 pad 위치 -100
            if "labels" not in batch:
                labels = batch["input_ids"].clone()
                if "attention_mask" in batch:
                    labels[batch["attention_mask"] == 0] = -100
                else:
                    attn = (batch["input_ids"] != tok.pad_token_id).long()
                    batch["attention_mask"] = attn
                    labels[attn == 0] = -100
                batch["labels"] = labels
            return batch

        return collate_fn

    # ------------------------- BADGE PROXY --------------------------- #
    from contextlib import nullcontext

    def jvp_proxy_raw(self, model, batch):
        """
        kmeans용 proxy embedding (B,H), 정규화 X
        - vocab 차원 생성 없음 (one_hot/err/einsum 제거)
        - OOM 방지
        """
        with torch.inference_mode():
            use_amp = batch["input_ids"].is_cuda
            amp_dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else torch.float16
            )

            # autocast (deprecated 해결)
            ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                out = model(**batch, output_hidden_states=True)
                h = out.hidden_states[-1]   # (B,T,H)
                logit = out.logits          # (B,T,V)

            # shift: position t predicts token t+1
            h = h[:, :-1, :].contiguous()         # (B,T-1,H)
            logit = logit[:, :-1, :].contiguous() # (B,T-1,V)

            lbl = batch["labels"][:, 1:].contiguous()  # (B,T-1)
            mask = (lbl != -100)

            # gather 안전 처리
            lbl_safe = lbl.masked_fill(~mask, 0)

            # 정답 토큰 logprob만 gather: (B,T-1)
            logp = F.log_softmax(logit, dim=-1)
            logp_y = logp.gather(-1, lbl_safe.unsqueeze(-1)).squeeze(-1)
            p_y = logp_y.exp()

            # CE gradient의 정답 클래스 성분: (p_y - 1)
            coeff = (p_y - 1.0) * mask  # (B,T-1)

            # 토큰 축 aggregation -> (B,H)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # (B,1)
            grad_seq = (coeff.unsqueeze(-1) * h).sum(dim=1) / denom  # (B,H)

            # cleanup
            del out, h, logit, lbl, lbl_safe, mask, logp, logp_y, p_y, coeff, denom
            if self.badge_cleanup_interval > 0:
                self._badge_cleanup_step += 1
                if (self._badge_cleanup_step % self.badge_cleanup_interval) == 0:
                    _cuda_cleanup(verbose=False)

            return grad_seq.float()


    def jvp_proxy_normalized(self, model, batch):
        """
        kmeans용 proxy embedding (B,H), L2 정규화 O
        """
        grad_seq = self.jvp_proxy_raw(model, batch)
        return F.normalize(grad_seq, dim=1)

    @staticmethod
    def _split_batch(batch, chunk_size):
        if chunk_size is None or chunk_size <= 0:
            yield batch
            return
        bsz = batch["input_ids"].size(0)
        if bsz <= chunk_size:
            yield batch
            return
        for s in range(0, bsz, chunk_size):
            e = min(bsz, s + chunk_size)
            yield {k: v[s:e] for k, v in batch.items()}

    # def jvp_proxy_normalized(self, model, batch): # 0222 edit
    #     with torch.inference_mode():
    #         use_amp = batch["input_ids"].is_cuda
    #         amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    #         with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
    #             out = model(**batch, output_hidden_states=True)
    #             h = out.hidden_states[-1]      # (B,T,H)
    #             logit = out.logits             # (B,T,V)

    #         h = h[..., :-1, :].contiguous()
    #         logit = logit[..., :-1, :].contiguous()

    #         p = F.softmax(logit, dim=-1)

    #         lbl = batch["labels"][..., 1:].contiguous()
    #         mask = (lbl != -100)

    #         lbl_safe = lbl.clone()
    #         lbl_safe[~mask] = 0

    #         one_hot = F.one_hot(lbl_safe, num_classes=p.size(-1)).type_as(p)

    #         err = (p - one_hot)
    #         err *= mask.unsqueeze(-1)

    #         grad_tok = torch.einsum("btv,bth->bvh", err, h)
    #         grad_seq = grad_tok.mean(1)

    #         del out, logit, p, one_hot, err, grad_tok, lbl, lbl_safe, mask, h
    #         _cuda_cleanup(verbose=False)

    #         return F.normalize(grad_seq.float(), dim=1) 
            
    # def jvp_proxy_normalized(self, model, batch):
    #     """
    #     기존 동작 유지: (B, H) 방향 임베딩 (L2 정규화된 임베딩)
    #     """
    #     with torch.no_grad():
    #         out = model(**batch, output_hidden_states=True)
    #         h = out.hidden_states[-1]                    # (B,T,H)
    #         logit = out.logits                           # (B,T,V)

    #     h = h[..., :-1, :]
    #     p = F.softmax(logit[..., :-1, :], dim=-1)
    #     lbl = batch["labels"][..., 1:].contiguous()

    #     mask = (lbl != -100)
    #     lbl_safe = lbl.clone()
    #     lbl_safe[~mask] = 0

    #     one_hot = F.one_hot(lbl_safe, num_classes=p.size(-1)).type_as(p)
    #     err = (p - one_hot) * mask.unsqueeze(-1)
    #     err = err.to(h.dtype)

    #     py = p.gather(-1, lbl_safe.unsqueeze(-1)).squeeze(-1)  # (B,T)
    #     err_y = (py - 1.0) * mask                              # (B,T)
    #     grad_seq = torch.einsum("bt,bth->bh", err_y.to(h.dtype), h)  # (B,H)
    #     grad_seq = grad_seq / (mask.sum(dim=1, keepdim=True).clamp_min(1))  # 길이 정규화
        
    #     return F.normalize(grad_seq.float(), dim=1)       # (B,H)
        # (B,T-1,V) & (B,T-1,H) → vocab축 평균으로 hidden proxy
        # 0213 edited
        # grad_tok = torch.einsum("btv,bth->bvh", err, h)   # (B,V,H)
        # grad_seq = grad_tok.mean(1)                       # (B,H)
        # return F.normalize(grad_seq.float(), dim=1)       # (B,H)

    # def jvp_proxy_raw(self, model, batch):
    #     """
    #     remove_mean용 raw 임베딩: 정규화하지 않고 (B,H) 반환
    #     """
    #     # print("in jvp_proxy_raw")
    #     with torch.no_grad():
    #         out = model(**batch, output_hidden_states=True)
    #         # print("in jvp_proxy_raw: got outcome of model")
    #         h = out.hidden_states[-1]                    # (B,T,H)
    #         logit = out.logits                           # (B,T,V)

    #     h = h[..., :-1, :]
    #     p = F.softmax(logit[..., :-1, :], dim=-1)
    #     lbl = batch["labels"][..., 1:].contiguous()

    #     mask = (lbl != -100)
    #     lbl_safe = lbl.clone()
    #     lbl_safe[~mask] = 0

    #     one_hot = F.one_hot(lbl_safe, num_classes=p.size(-1)).type_as(p)
    #     err = (p - one_hot) * mask.unsqueeze(-1)
    #     err = err.to(h.dtype)

    #     grad_tok = torch.einsum("btv,bth->bvh", err, h)   # (B,V,H)
    #     grad_seq = grad_tok.mean(1)                       # (B,H)
    #     return grad_seq.float()                           # (B,H) raw

    # def jvp_proxy_raw(self, model, batch):
    #     with torch.inference_mode():
    #         use_amp = batch["input_ids"].is_cuda
    #         amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    #         with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
    #             out = model(**batch, output_hidden_states=True)
    #             h = out.hidden_states[-1]      # (B,T,H)
    #             logit = out.logits             # (B,T,V)

    #         h = h[..., :-1, :].contiguous()
    #         logit = logit[..., :-1, :].contiguous()

    #         p = F.softmax(logit, dim=-1)

    #         lbl = batch["labels"][..., 1:].contiguous()
    #         mask = (lbl != -100)

    #         lbl_safe = lbl.clone()
    #         lbl_safe[~mask] = 0

    #         one_hot = F.one_hot(lbl_safe, num_classes=p.size(-1)).type_as(p)

    #         err = (p - one_hot)
    #         err *= mask.unsqueeze(-1)

    #         grad_tok = torch.einsum("btv,bth->bvh", err, h)
    #         grad_seq = grad_tok.mean(1)

    #         del out, logit, p, one_hot, err, grad_tok, lbl, lbl_safe, mask, h
    #         _cuda_cleanup(verbose=False)

    #         return grad_seq.float()


    # ------------------- 임베딩 계산 (remove_mean 반영) -------------- #
    def compute_badge_embeddings(
        self,
        model,
        dataset,
        batch_size,
        device,
        forward_chunk_size=None,
        sample_indices=None,
        num_real_to_save=None,
    ):
        model.eval()
        self._reset_saved_grad_buffers()
        num_workers = int(getattr(self.trainer.args, "dataloader_num_workers", 0))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._get_collate_fn(),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )

        if self.remove_mean:
            # 1) raw 수집 → 2) 전체 mean 계산(DDP면 all-reduce) → 3) mean만 제거
            raw_list = []
            seen = 0
            for batch in tqdm(dataloader, desc="Computing BADGE embeddings (raw)"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                for sub_batch in self._split_batch(batch, forward_chunk_size):
                    g_raw = self.jvp_proxy_raw(model, sub_batch)     # (B,H)
                    bsz = int(g_raw.size(0))
                    start = seen
                    end = seen + bsz
                    seen = end

                    g_raw_cpu = g_raw.detach().cpu()
                    save_count = bsz
                    if num_real_to_save is not None:
                        if start >= num_real_to_save:
                            save_count = 0
                        else:
                            save_count = min(end, num_real_to_save) - start
                    if save_count > 0:
                        idx_chunk = sample_indices[start : start + save_count] if sample_indices is not None else None
                        g_save = g_raw_cpu[:save_count]
                        data_info = {
                            k: v[:save_count].detach().cpu()
                            for k, v in sub_batch.items()
                            if isinstance(v, torch.Tensor)
                        }
                        self._maybe_save_grad_batch(g_save, idx_chunk, data_info)
                    raw_list.append(g_raw_cpu)
                    del g_raw, g_raw_cpu, sub_batch
                del batch
            G_raw = torch.cat(raw_list, dim=0).float()       # (N,H)

            # 전역 평균 (DDP면 all_reduce로 전 세계 평균)
            if torch.distributed.is_initialized():
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sum_local = G_raw.to(dev).sum(dim=0, keepdim=True)          # (1,H)
                cnt_local = torch.tensor([G_raw.size(0)], device=dev, dtype=torch.long)
                torch.distributed.all_reduce(sum_local, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cnt_local, op=torch.distributed.ReduceOp.SUM)
                mu = (sum_local / cnt_local.item()).cpu()                   # (1,H)
            else:
                mu = G_raw.mean(dim=0, keepdim=True)

            G_centered = (G_raw - mu)                                       # (N,H) mean-removed
            return G_centered.numpy().astype(np.float32)                     # k-means 입력
        else:
            # 기존 방식: 배치마다 정규화 임베딩 수집 후 concat
            embs = []
            seen = 0
            for batch in tqdm(dataloader, desc="Computing BADGE embeddings"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                for sub_batch in self._split_batch(batch, forward_chunk_size):
                    if self.save_badge_grads:
                        g_raw = self.jvp_proxy_raw(model, sub_batch)
                        bsz = int(g_raw.size(0))
                        start = seen
                        end = seen + bsz
                        seen = end
                        g_raw_cpu = g_raw.detach().cpu()
                        save_count = bsz
                        if num_real_to_save is not None:
                            if start >= num_real_to_save:
                                save_count = 0
                            else:
                                save_count = min(end, num_real_to_save) - start
                        if save_count > 0:
                            idx_chunk = sample_indices[start : start + save_count] if sample_indices is not None else None
                            g_save = g_raw_cpu[:save_count]
                            data_info = {
                                k: v[:save_count].detach().cpu()
                                for k, v in sub_batch.items()
                                if isinstance(v, torch.Tensor)
                            }
                            self._maybe_save_grad_batch(g_save, idx_chunk, data_info)
                        emb = F.normalize(g_raw, dim=1)
                        embs.append(emb.cpu())
                        del g_raw, g_raw_cpu, emb, sub_batch
                    else:
                        emb = self.jvp_proxy_normalized(model, sub_batch)       # (B,H) normalized
                        seen += int(emb.size(0))
                        embs.append(emb.cpu())
                        del emb, sub_batch
                del batch
            return torch.cat(embs, dim=0).numpy().astype(np.float32)

    def _broadcast(self, obj, src=0):
        if dist.is_available() and dist.is_initialized():
            payload = [obj if dist.get_rank() == src else None]
            dist.broadcast_object_list(payload, src=src)
            return payload[0]
        return obj
    
    @staticmethod
    def _gather_varlen_1d(idx_local: torch.Tensor, device):
        """1D LongTensor variable-length all_gather. Returns (idx_all_np) on rank0, else (None)."""
        ws = dist.get_world_size()
        n_local = torch.tensor([idx_local.numel()], device=device, dtype=torch.long)
        n_list = [torch.zeros_like(n_local) for _ in range(ws)]
        dist.all_gather(n_list, n_local)  # 각 랭크 길이 수집
        n_each = [int(t.item()) for t in n_list]
        max_n = max(n_each) if ws > 0 else idx_local.numel()

        pad = torch.full((max_n,), -1, device=device, dtype=idx_local.dtype)
        pad[:n_local.item()] = idx_local
        buf = [torch.empty_like(pad) for _ in range(ws)]
        dist.all_gather(buf, pad)

        if dist.get_rank() == 0:
            parts = [b[:n] for b, n in zip(buf, n_each)]
            idx_all = torch.cat(parts, dim=0).cpu().numpy()
            return idx_all
        return None

    @staticmethod
    def _gather_varlen_2d(emb_local: torch.Tensor, device):
        """2D FloatTensor variable-length all_gather. Returns (emb_all_np) on rank0, else (None)."""
        ws = dist.get_world_size()
        n_local = torch.tensor([emb_local.size(0)], device=device, dtype=torch.long)
        n_list = [torch.zeros_like(n_local) for _ in range(ws)]
        dist.all_gather(n_list, n_local)
        n_each = [int(t.item()) for t in n_list]
        max_n = max(n_each)
        H = emb_local.size(1)

        pad = torch.zeros((max_n, H), device=device, dtype=emb_local.dtype)
        if n_each[dist.get_rank()] > 0:
            pad[:emb_local.size(0)] = emb_local
        buf = [torch.zeros_like(pad) for _ in range(ws)]
        dist.all_gather(buf, pad)

        if dist.get_rank() == 0:
            parts = [b[:n] for b, n in zip(buf, n_each)]
            emb_all = torch.cat(parts, dim=0).cpu().numpy()
            return emb_all
        return None

    def _broadcast_obj(obj, src=0):
        if dist.is_available() and dist.is_initialized():
            payload = [obj if dist.get_rank() == src else None]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
        return obj

    @staticmethod
    def _shuffle_preserving_mod(order: List[int], ws: int, seed: int) -> List[int]:
        """
        전역 인덱스 order를 모듈로 클래스(=rank)별로만 셔플하고,
        다시 라운드로빈으로 interleave하여 반환.
        - pos % ws 는 불변
        - 각 버킷 길이는 그대로라서 drop_last/microbatch 정렬도 유지
        """
        # 1) 모듈로 버킷으로 분할
        buckets = [[] for _ in range(ws)]
        for pos, idx in enumerate(order):
            buckets[pos % ws].append(idx)

        # 2) 버킷 내부 셔플(에폭별 결정론적)
        for r in range(ws):
            rng = random.Random(seed + r)  # rank별 다른 시드(원하면 동일 seed로 바꿔도 됨)
            rng.shuffle(buckets[r])

        # 3) 다시 interleave (라운드로빈)
        #    길이가 완전히 동일하지 않아도 안전하게 zip_longest 스타일로 합침
        out = []
        maxlen = max(len(b) for b in buckets) if buckets else 0
        for j in range(maxlen):
            for r in range(ws):
                if j < len(buckets[r]):
                    out.append(buckets[r][j])
        return out

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        에폭 시작:
        - (모든 랭크) 패딩 포함 로컬 서브셋으로 BADGE 임베딩 계산(계산량 동일화)
        - (all_gather) '실제 인덱스/임베딩'만 모아 rank0에서 전역 정렬
        - (rank0) KMeans(K=world_size) → 최소크기 프루닝(centroid에서 먼 것 제거)
        - (rank0) 인터리브 전역 순서, 프루닝 전 클러스터 비율 → broadcast
        - sampler.update_indices(), rank_weights 저장, DataLoader 재생성
        """
        trainer = self.trainer
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not dist.is_initialized():
            world_size, rank = 1, 0
        else:
            world_size, rank = dist.get_world_size(), dist.get_rank()

        sampler = getattr(trainer, "_train_sampler_for_callback", None)
        assert sampler is not None and hasattr(sampler, "update_indices"), \
            "train sampler에 update_indices가 필요합니다."

        model.eval()

        # -------- (1) 각 랭크 로컬 인덱스: strided real + padding to equal size --------
        N   = len(self.dataset)
        ws  = world_size
        r   = rank
        micro = int(args.per_device_train_batch_size)
        chunk_mult = max(1, int(getattr(args, "badge_forward_chunk_mult", 1)))
        base_badge_chunk = micro * max(ws, 1) * chunk_mult
        drop_last = bool(getattr(args, "dataloader_drop_last", False))  # KMeans 단계에서만 사용

        if ws > 1:
            # 실제(real) 인덱스: r, r+ws, r+2ws, ...
            idx_real = list(range(r, N, ws))
            real_len = len(idx_real)

            # 타겟 길이: 모든 랭크 동일 (선택) 마이크로 배치 배수로 정렬
            target = math.ceil(N / ws)
            target = math.ceil(target / max(micro, 1)) * max(micro, 1)

            # 패딩 값: 마지막 real 인덱스가 있으면 그것, 없으면 N-1
            pad_val = idx_real[-1] if real_len > 0 else max(0, N - 1)
            idx_full = idx_real + [pad_val] * max(0, target - real_len)

            # 이 랭크가 실제로 임베딩을 계산할 서브셋(패딩 포함)
            local_ds = Subset(self.dataset, idx_full)
        else:
            idx_real = list(range(N))
            real_len = N
            target   = N
            local_ds = self.dataset

        # -------- (2) 임베딩 계산: 패딩은 계산하되, gather에는 'real'만 보냄 --------
        # remove_mean=True는 all-reduce로 전역 평균을 쓰는데, 패딩이 포함되면 평균이 왜곡될 수 있음.
        # 패딩 모드에서는 임시로 mean 제거를 끄고(raw/normalized 그대로 사용) 이후 KMeans에 투입.
        saved_remove_mean = getattr(self, "remove_mean", False)
        try:
            if ws > 1:
                self.remove_mean = False  # 패딩 모드: 평균 왜곡 방지
            if args.badge_batch == 1:
                micro = micro * ws 
            else:
                micro = micro * ws * args.badge_batch
            save_indices = idx_full if ws > 1 else idx_real
            save_real = real_len if ws > 1 else len(save_indices)
            emb_full_np = self.compute_badge_embeddings(
                model,
                local_ds,
                micro,
                device,
                forward_chunk_size=base_badge_chunk,
                sample_indices=save_indices,
                num_real_to_save=save_real,
            )  # (target, H)
        finally:
            self.remove_mean = saved_remove_mean
        self._flush_saved_grad_file(state.epoch)

        # real 부분만 사용하여 gather에 보냄
        if ws > 1:
            emb_local = torch.from_numpy(emb_full_np[:real_len]).to(device=device, dtype=torch.float32)
            idx_send  = torch.tensor(idx_real, dtype=torch.long, device=device)
        else:
            emb_local = torch.from_numpy(emb_full_np).to(device=device, dtype=torch.float32)
            idx_send  = torch.arange(0, N, dtype=torch.long, device=device)

        if ws > 1:
            # -------- (3) all_gather (가변 길이) --------
            idx_all = self._gather_varlen_1d(idx_send, device)       # np.int64, 길이 합=N
            emb_all = self._gather_varlen_2d(emb_local, device)      # np.float32, (N, H)

            # -------- (4) rank0에서 전역 정렬 --------
            if r == 0:
                order   = np.argsort(idx_all)
                emb_full = emb_all[order]                            # (N, H)
            else:
                emb_full = None

            # 전 랭크에 전파
            emb = self._broadcast(emb_full, src=0)
        else:
            emb = emb_full_np                                       # 단일 프로세스

        # -------- (5) rank0: KMeans/프루닝/인터리브 & 비율 계산 --------
        if r == 0:
            interleaved, diag, _ = _kmeans_prune_equalize_interleave(
                emb,
                world_size=ws,
                micro=micro,
                drop_last=drop_last,
                seed=args.seed + int(state.epoch),
                prune_type = args.prune,
                sampling = args.sampling_type,
                ratio = args.ratio,
            )
            kept = len(interleaved)
            self._maybe_save_cluster_artifacts(diag, interleaved, N, state.epoch)

            print(diag)
            
            # 프루닝 "전" 클러스터 비율 → 랭크 가중치(평균 1 정규화)
            ratios = diag.get("cluster_ratios")
            size_ratios = diag.get("cluster_size_ratio")
            cluster_weights = diag.get("cluster_weights")

            ## edited: alpagasus-coreset-ratio-7b
            # if ratios is None:
            #     sizes = diag["cluster_sizes"]; total = float(sum(sizes))
            #     ratios = [s / total for s in sizes]
            # mean_r = sum(ratios) / max(len(ratios), 1)
            # rank_weights = [r_ / mean_r for r_ in ratios]

            ## edited: alpagasus-coreset-mean-7b
            sizes = diag["cluster_sizes"]
            total_size = sum(sizes)
            n_gpus = len(sizes)
            # rank_weights = [ 1 + (size - (total_size / n_gpus)) / total_size for size in sizes ]
            rank_weights = [n_gpus * size / total_size for size in sizes]

            if args.weight is False:
                if args.cluster_sizes is None:
                    raise ValueError("You must specify --cluster_sizes when --weight is False.")
                print("Overriding cluster sizes from args.cluster_sizes")
                sizes = args.cluster_sizes
                total_size = sum(sizes)
                n_gpus = len(sizes)
                rank_weights = [n_gpus * size / total_size for size in sizes]
            elif args.weight_mode == "none":
                rank_weights = [1.0 for size in sizes]
        else:
            interleaved, diag, rank_weights = None, None, None

        # -------- (6) 결과 브로드캐스트 --------
        interleaved = self._broadcast(interleaved, src=0)   # list[int]
        diag        = self._broadcast(diag,        src=0)   # dict
        rank_weights= self._broadcast(rank_weights, src=0)  # list[float]
        if dist.is_initialized():
            dist.barrier()

        print(f"Rank {r} after broadcast: got {len(interleaved) if interleaved is not None else 'None'} interleaved indices, diag={diag}, rank_weights={rank_weights}")

        # -------- (7) sampler 갱신 & DataLoader 재생성 --------
        sampler.update_indices(interleaved)
        
        import types
        trainer._train_sampler = sampler
        trainer._get_train_sampler = types.MethodType(lambda self: sampler, trainer)

        trainer.args.rank_weights = rank_weights
        trainer._rank_weight_tensor = torch.tensor(rank_weights, device=device, dtype=torch.float32)
        trainer.__dict__.pop("_train_dataloader", None)

        if r == 0:
            print(
                f"[Epoch {int(state.epoch)}] "
                f"sizes={diag['cluster_sizes']} ratios={[round(x,4) for x in diag['cluster_ratios']]} "
                f"m_min={diag['min_cluster_size']} m_adj={diag['m_adj']} "
                f"kept={diag['kept']} pruned={diag['pruned']} "
                f"rank_weights={[round(w,4) for w in rank_weights]}"
            )

        model.train()



class UniformCallback(TrainerCallback):
    def __init__(self, trainer, tokenizer, dataset, remove_mean: bool = True):
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.remove_mean = remove_mean
        self.badge_cleanup_interval = int(getattr(self.trainer.args, "badge_cleanup_interval", 0))
        self._badge_cleanup_step = 0
        self.save_badge_grads = bool(getattr(self.trainer.args, "save_badge_grads", False))
        self.badge_grad_save_dir = getattr(self.trainer.args, "badge_grad_save_dir", None)
        self._grad_save_error_reported = False
        self._saved_grad_batches = []
        self._saved_indices = []
        self._saved_data_info = defaultdict(list)

    def _reset_saved_grad_buffers(self):
        self._saved_grad_batches.clear()
        self._saved_indices.clear()
        self._saved_data_info.clear()

    def _maybe_save_grad_batch(self, grad_batch_cpu: torch.Tensor, sample_indices=None, data_batch_cpu=None):
        if not self.save_badge_grads:
            return
        self._saved_grad_batches.append(grad_batch_cpu)
        if sample_indices is not None:
            self._saved_indices.append(torch.as_tensor(sample_indices, dtype=torch.long))
        if data_batch_cpu is not None:
            for k, v in data_batch_cpu.items():
                self._saved_data_info[k].append(v)

    def _flush_saved_grad_file(self, epoch_value):
        if not self.save_badge_grads:
            return
        if len(self._saved_grad_batches) == 0:
            return
        save_dir = self.badge_grad_save_dir or os.path.join(self.trainer.args.output_dir, "badge_grads")
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        try:
            os.makedirs(save_dir, exist_ok=True)
            epoch_int = int(epoch_value) if epoch_value is not None else 0
            payload = {
                "badge_grad": torch.cat(self._saved_grad_batches, dim=0),
                "indices": (
                    torch.cat(self._saved_indices, dim=0)
                    if len(self._saved_indices) > 0
                    else None
                ),
                "data_info": {
                    k: torch.cat(vs, dim=0) for k, vs in self._saved_data_info.items() if len(vs) > 0
                },
            }
            file_path = os.path.join(save_dir, f"rank{rank:02d}_badge_epoch{epoch_int:04d}.pt")
            torch.save(payload, file_path)
        except Exception as e:
            if not self._grad_save_error_reported:
                print(f"[WARN] Failed to save BADGE gradients to {save_dir}: {e}")
                self._grad_save_error_reported = True
        finally:
            self._reset_saved_grad_buffers()

    def _maybe_save_cluster_artifacts(self, diag: dict, interleaved: list, total_n: int, epoch_value):
        if not self.save_badge_grads:
            return
        save_dir = self.badge_grad_save_dir or os.path.join(self.trainer.args.output_dir, "badge_grads")
        try:
            os.makedirs(save_dir, exist_ok=True)
            labels = np.asarray(diag.get("labels", np.zeros((total_n,), dtype=np.int64)), dtype=np.int64)
            kept_idx = np.asarray(interleaved, dtype=np.int64)
            kept_mask = np.zeros((total_n,), dtype=np.bool_)
            if kept_idx.size > 0:
                kept_mask[kept_idx] = True
            epoch_int = int(epoch_value) if epoch_value is not None else 0
            np.savez(
                os.path.join(save_dir, f"rank00_cluster_epoch{epoch_int:04d}.npz"),
                sample_idx=np.arange(total_n, dtype=np.int64),
                cluster_labels=labels,
                kept_mask=kept_mask,
                interleaved_order=kept_idx,
                cluster_sizes=np.asarray(diag.get("cluster_sizes", []), dtype=np.int64),
            )
        except Exception as e:
            if not self._grad_save_error_reported:
                print(f"[WARN] Failed to save cluster artifacts to {save_dir}: {e}")
                self._grad_save_error_reported = True

    # ------------------------- PADDED COLLATE ------------------------- #
    def _get_collate_fn(self):
        # 1) 학습과 동일한 collate 사용
        collate_fn = getattr(self.trainer, "data_collator", None)
        if collate_fn is not None:
            return collate_fn

        # 2) tokenizer.pad 기반 동적 패딩 collate
        tok = self.tokenizer
        if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token_id", None) is not None:
            tok.pad_token_id = tok.eos_token_id

        def collate_fn(features):
            # feature 값이 텐서일 수도 있어 tokenizer.pad가 리스트를 기대하므로 tolist 처리
            proc = []
            for ex in features:
                ex2 = {}
                for k, v in ex.items():
                    if isinstance(v, torch.Tensor):
                        ex2[k] = v.tolist()
                    else:
                        ex2[k] = v
                proc.append(ex2)
            batch = tok.pad(proc, return_tensors="pt", padding="longest")
            # labels 없으면 생성: LM 학습과 동일하게 pad 위치 -100
            if "labels" not in batch:
                labels = batch["input_ids"].clone()
                if "attention_mask" in batch:
                    labels[batch["attention_mask"] == 0] = -100
                else:
                    attn = (batch["input_ids"] != tok.pad_token_id).long()
                    batch["attention_mask"] = attn
                    labels[attn == 0] = -100
                batch["labels"] = labels
            return batch

        return collate_fn

    # ------------------------- BADGE PROXY --------------------------- #
    # def jvp_proxy_normalized(self, model, batch):
    #     with torch.inference_mode():
    #         use_amp = batch["input_ids"].is_cuda
    #         amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    #         with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
    #             out = model(**batch, output_hidden_states=True)
    #             h = out.hidden_states[-1]      # (B,T,H)
    #             logit = out.logits             # (B,T,V)

    #         h = h[..., :-1, :].contiguous()
    #         logit = logit[..., :-1, :].contiguous()

    #         p = F.softmax(logit, dim=-1)

    #         lbl = batch["labels"][..., 1:].contiguous()
    #         mask = (lbl != -100)

    #         lbl_safe = lbl.clone()
    #         lbl_safe[~mask] = 0

    #         one_hot = F.one_hot(lbl_safe, num_classes=p.size(-1)).type_as(p)

    #         err = (p - one_hot)
    #         err *= mask.unsqueeze(-1)

    #         grad_tok = torch.einsum("btv,bth->bvh", err, h)
    #         grad_seq = grad_tok.mean(1)

    #         del out, logit, p, one_hot, err, grad_tok, lbl, lbl_safe, mask, h
    #         _cuda_cleanup(verbose=False)

    #         return F.normalize(grad_seq.float(), dim=1) 
            
    # def jvp_proxy_raw(self, model, batch):
    #     with torch.inference_mode():
    #         use_amp = batch["input_ids"].is_cuda
    #         amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    #         with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
    #             out = model(**batch, output_hidden_states=True)
    #             h = out.hidden_states[-1]      # (B,T,H)
    #             logit = out.logits             # (B,T,V)

    #         h = h[..., :-1, :].contiguous()
    #         logit = logit[..., :-1, :].contiguous()

    #         p = F.softmax(logit, dim=-1)

    #         lbl = batch["labels"][..., 1:].contiguous()
    #         mask = (lbl != -100)

    #         lbl_safe = lbl.clone()
    #         lbl_safe[~mask] = 0

    #         one_hot = F.one_hot(lbl_safe, num_classes=p.size(-1)).type_as(p)

    #         err = (p - one_hot)
    #         err *= mask.unsqueeze(-1)

    #         grad_tok = torch.einsum("btv,bth->bvh", err, h)
    #         grad_seq = grad_tok.mean(1)

    #         del out, logit, p, one_hot, err, grad_tok, lbl, lbl_safe, mask, h
    #         _cuda_cleanup(verbose=False)

    #         return grad_seq.float()
    from contextlib import nullcontext

    def jvp_proxy_raw(self, model, batch):
        """
        kmeans용 proxy embedding (B,H), 정규화 X
        - vocab 차원 생성 없음 (one_hot/err/einsum 제거)
        - OOM 방지
        """
        with torch.inference_mode():
            use_amp = batch["input_ids"].is_cuda
            amp_dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else torch.float16
            )

            # autocast (deprecated 해결)
            ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()
            with ctx:
                out = model(**batch, output_hidden_states=True)
                h = out.hidden_states[-1]   # (B,T,H)
                logit = out.logits          # (B,T,V)

            # shift: position t predicts token t+1
            h = h[:, :-1, :].contiguous()         # (B,T-1,H)
            logit = logit[:, :-1, :].contiguous() # (B,T-1,V)

            lbl = batch["labels"][:, 1:].contiguous()  # (B,T-1)
            mask = (lbl != -100)

            # gather 안전 처리
            lbl_safe = lbl.masked_fill(~mask, 0)

            # 정답 토큰 logprob만 gather: (B,T-1)
            logp = F.log_softmax(logit, dim=-1)
            logp_y = logp.gather(-1, lbl_safe.unsqueeze(-1)).squeeze(-1)
            p_y = logp_y.exp()

            # CE gradient의 정답 클래스 성분: (p_y - 1)
            coeff = (p_y - 1.0) * mask  # (B,T-1)

            # 토큰 축 aggregation -> (B,H)
            denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # (B,1)
            grad_seq = (coeff.unsqueeze(-1) * h).sum(dim=1) / denom  # (B,H)

            # cleanup
            del out, h, logit, lbl, lbl_safe, mask, logp, logp_y, p_y, coeff, denom
            if self.badge_cleanup_interval > 0:
                self._badge_cleanup_step += 1
                if (self._badge_cleanup_step % self.badge_cleanup_interval) == 0:
                    _cuda_cleanup(verbose=False)

            return grad_seq.float()


    def jvp_proxy_normalized(self, model, batch):
        """
        kmeans용 proxy embedding (B,H), L2 정규화 O
        """
        grad_seq = self.jvp_proxy_raw(model, batch)
        return F.normalize(grad_seq, dim=1)
        
    # def jvp_proxy_normalized(self, model, batch):
    #     with torch.inference_mode():
    #         use_amp = batch["input_ids"].is_cuda
    #         amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    #         # FutureWarning 해결 (권장)
    #         ctx = torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype) if use_amp else nullcontext()
    #         with ctx:
    #             out = model(**batch, output_hidden_states=True)
    #             h = out.hidden_states[-1]   # (B,T,H)
    #             logit = out.logits          # (B,T,V)

    #         # shift: predict token t+1 from position t
    #         h = h[:, :-1, :].contiguous()         # (B,T-1,H)
    #         logit = logit[:, :-1, :].contiguous() # (B,T-1,V)

    #         lbl = batch["labels"][:, 1:].contiguous()  # (B,T-1)
    #         mask = (lbl != -100)

    #         # 안전 gather: -100 자리는 0으로 치환
    #         lbl_safe = lbl.masked_fill(~mask, 0)

    #         # 정답 토큰의 log-prob만 뽑기: (B,T-1)
    #         logp = F.log_softmax(logit, dim=-1)
    #         logp_y = logp.gather(-1, lbl_safe.unsqueeze(-1)).squeeze(-1)
    #         p_y = logp_y.exp()

    #         # (p_y - 1): CE gradient의 정답 클래스 성분
    #         coeff = (p_y - 1.0) * mask   # (B,T-1)

    #         # 토큰 방향으로 aggregation -> (B,H)
    #         denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # (B,1)
    #         grad_seq = (coeff.unsqueeze(-1) * h).sum(dim=1) / denom

    #         # 정리
    #         del out, logit, logp, logp_y, p_y, lbl, lbl_safe, mask, h, coeff, denom
    #         _cuda_cleanup(verbose=False)

    #         return F.normalize(grad_seq.float(), dim=1)

    # def jvp_proxy_raw(self, model, batch):
    #     with torch.inference_mode():
    #         use_amp = batch["input_ids"].is_cuda
    #         amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    #         with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
    #             out = model(**batch, output_hidden_states=True)
    #             h = out.hidden_states[-1]   # (B,T,H)
    #             logit = out.logits          # (B,T,V)

    #         h = h[:, :-1, :].contiguous()         # (B,T-1,H)
    #         logit = logit[:, :-1, :].contiguous() # (B,T-1,V)

    #         lbl = batch["labels"][:, 1:].contiguous()  # (B,T-1)
    #         mask = (lbl != -100)
    #         lbl_safe = lbl.masked_fill(~mask, 0)

    #         logp = F.log_softmax(logit, dim=-1)
    #         logp_y = logp.gather(-1, lbl_safe.unsqueeze(-1)).squeeze(-1)
    #         p_y = logp_y.exp()

    #         coeff = (p_y - 1.0) * mask  # (B,T-1)

    #         denom = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # (B,1)
    #         grad_seq = (coeff.unsqueeze(-1) * h).sum(dim=1) / denom   # (B,H)

    #         del out, logit, logp, logp_y, p_y, lbl, lbl_safe, mask, h, coeff, denom
    #         if self.badge_cleanup_interval > 0:
    #             self._badge_cleanup_step += 1
    #             if (self._badge_cleanup_step % self.badge_cleanup_interval) == 0:
    #                 _cuda_cleanup(verbose=False)

    #         return grad_seq.float()

    @staticmethod
    def _split_batch(batch, chunk_size):
        if chunk_size is None or chunk_size <= 0:
            yield batch
            return
        bsz = batch["input_ids"].size(0)
        if bsz <= chunk_size:
            yield batch
            return
        for s in range(0, bsz, chunk_size):
            e = min(bsz, s + chunk_size)
            yield {k: v[s:e] for k, v in batch.items()}

    # ------------------- 임베딩 계산 (remove_mean 반영) -------------- #
    def compute_badge_embeddings(
        self,
        model,
        dataset,
        batch_size,
        device,
        forward_chunk_size=None,
        sample_indices=None,
        num_real_to_save=None,
    ):
        model.eval()
        self._reset_saved_grad_buffers()
        num_workers = int(getattr(self.trainer.args, "dataloader_num_workers", 0))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._get_collate_fn(),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )

        if self.remove_mean:
            # 1) raw 수집 → 2) 전체 mean 계산(DDP면 all-reduce) → 3) mean만 제거
            raw_list = []
            seen = 0
            for batch in tqdm(dataloader, desc="Computing BADGE embeddings (raw)"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                for sub_batch in self._split_batch(batch, forward_chunk_size):
                    g_raw = self.jvp_proxy_raw(model, sub_batch)     # (B,H)
                    bsz = int(g_raw.size(0))
                    start = seen
                    end = seen + bsz
                    seen = end
                    g_raw_cpu = g_raw.detach().cpu()
                    save_count = bsz
                    if num_real_to_save is not None:
                        if start >= num_real_to_save:
                            save_count = 0
                        else:
                            save_count = min(end, num_real_to_save) - start
                    if save_count > 0:
                        idx_chunk = sample_indices[start : start + save_count] if sample_indices is not None else None
                        g_save = g_raw_cpu[:save_count]
                        data_info = {
                            k: v[:save_count].detach().cpu()
                            for k, v in sub_batch.items()
                            if isinstance(v, torch.Tensor)
                        }
                        self._maybe_save_grad_batch(g_save, idx_chunk, data_info)
                    raw_list.append(g_raw_cpu)
                    del g_raw, g_raw_cpu, sub_batch
                del batch
            G_raw = torch.cat(raw_list, dim=0).float()       # (N,H)

            # 전역 평균 (DDP면 all_reduce로 전 세계 평균)
            if torch.distributed.is_initialized():
                dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sum_local = G_raw.to(dev).sum(dim=0, keepdim=True)          # (1,H)
                cnt_local = torch.tensor([G_raw.size(0)], device=dev, dtype=torch.long)
                torch.distributed.all_reduce(sum_local, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cnt_local, op=torch.distributed.ReduceOp.SUM)
                mu = (sum_local / cnt_local.item()).cpu()                   # (1,H)
            else:
                mu = G_raw.mean(dim=0, keepdim=True)

            G_centered = (G_raw - mu)                                       # (N,H) mean-removed
            return G_centered.numpy().astype(np.float32)                     # k-means 입력
        else:
            # 기존 방식: 배치마다 정규화 임베딩 수집 후 concat
            embs = []
            seen = 0
            for batch in tqdm(dataloader, desc="Computing BADGE embeddings"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                for sub_batch in self._split_batch(batch, forward_chunk_size):
                    if self.save_badge_grads:
                        g_raw = self.jvp_proxy_raw(model, sub_batch)
                        bsz = int(g_raw.size(0))
                        start = seen
                        end = seen + bsz
                        seen = end
                        g_raw_cpu = g_raw.detach().cpu()
                        save_count = bsz
                        if num_real_to_save is not None:
                            if start >= num_real_to_save:
                                save_count = 0
                            else:
                                save_count = min(end, num_real_to_save) - start
                        if save_count > 0:
                            idx_chunk = sample_indices[start : start + save_count] if sample_indices is not None else None
                            g_save = g_raw_cpu[:save_count]
                            data_info = {
                                k: v[:save_count].detach().cpu()
                                for k, v in sub_batch.items()
                                if isinstance(v, torch.Tensor)
                            }
                            self._maybe_save_grad_batch(g_save, idx_chunk, data_info)
                        emb = F.normalize(g_raw, dim=1)
                        embs.append(emb.cpu())
                        del g_raw, g_raw_cpu, emb, sub_batch
                    else:
                        emb = self.jvp_proxy_normalized(model, sub_batch)       # (B,H) normalized
                        seen += int(emb.size(0))
                        embs.append(emb.cpu())
                        del emb, sub_batch
                del batch
            return torch.cat(embs, dim=0).numpy().astype(np.float32)

    def _broadcast(self, obj, src=0):
        if dist.is_available() and dist.is_initialized():
            payload = [obj if dist.get_rank() == src else None]
            dist.broadcast_object_list(payload, src=src)
            return payload[0]
        return obj
    
    @staticmethod
    def _gather_varlen_1d(idx_local: torch.Tensor, device):
        """1D LongTensor variable-length all_gather. Returns (idx_all_np) on rank0, else (None)."""
        ws = dist.get_world_size()
        n_local = torch.tensor([idx_local.numel()], device=device, dtype=torch.long)
        n_list = [torch.zeros_like(n_local) for _ in range(ws)]
        dist.all_gather(n_list, n_local)  # 각 랭크 길이 수집
        n_each = [int(t.item()) for t in n_list]
        max_n = max(n_each) if ws > 0 else idx_local.numel()

        pad = torch.full((max_n,), -1, device=device, dtype=idx_local.dtype)
        pad[:n_local.item()] = idx_local
        buf = [torch.empty_like(pad) for _ in range(ws)]
        dist.all_gather(buf, pad)

        if dist.get_rank() == 0:
            parts = [b[:n] for b, n in zip(buf, n_each)]
            idx_all = torch.cat(parts, dim=0).cpu().numpy()
            return idx_all
        return None

    @staticmethod
    def _gather_varlen_2d(emb_local: torch.Tensor, device):
        """2D FloatTensor variable-length all_gather. Returns (emb_all_np) on rank0, else (None)."""
        ws = dist.get_world_size()
        n_local = torch.tensor([emb_local.size(0)], device=device, dtype=torch.long)
        n_list = [torch.zeros_like(n_local) for _ in range(ws)]
        dist.all_gather(n_list, n_local)
        n_each = [int(t.item()) for t in n_list]
        max_n = max(n_each)
        H = emb_local.size(1)

        pad = torch.zeros((max_n, H), device=device, dtype=emb_local.dtype)
        if n_each[dist.get_rank()] > 0:
            pad[:emb_local.size(0)] = emb_local
        buf = [torch.zeros_like(pad) for _ in range(ws)]
        dist.all_gather(buf, pad)

        if dist.get_rank() == 0:
            parts = [b[:n] for b, n in zip(buf, n_each)]
            emb_all = torch.cat(parts, dim=0).cpu().numpy()
            return emb_all
        return None

    def _broadcast_obj(self, obj, src=0):
        if dist.is_available() and dist.is_initialized():
            payload = [obj if dist.get_rank() == src else None]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
        return obj

    @staticmethod
    def _shuffle_preserving_mod(order: List[int], ws: int, seed: int) -> List[int]:
        """
        전역 인덱스 order를 모듈로 클래스(=rank)별로만 셔플하고,
        다시 라운드로빈으로 interleave하여 반환.
        - pos % ws 는 불변
        - 각 버킷 길이는 그대로라서 drop_last/microbatch 정렬도 유지
        """
        # 1) 모듈로 버킷으로 분할
        buckets = [[] for _ in range(ws)]
        for pos, idx in enumerate(order):
            buckets[pos % ws].append(idx)

        # 2) 버킷 내부 셔플(에폭별 결정론적)
        for r in range(ws):
            rng = random.Random(seed + r)  # rank별 다른 시드(원하면 동일 seed로 바꿔도 됨)
            rng.shuffle(buckets[r])

        # 3) 다시 interleave (라운드로빈)
        #    길이가 완전히 동일하지 않아도 안전하게 zip_longest 스타일로 합침
        out = []
        maxlen = max(len(b) for b in buckets) if buckets else 0
        for j in range(maxlen):
            for r in range(ws):
                if j < len(buckets[r]):
                    out.append(buckets[r][j])
        return out

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """
        에폭 시작:
        - (모든 랭크) 패딩 포함 로컬 서브셋으로 BADGE 임베딩 계산(계산량 동일화)
        - (all_gather) '실제 인덱스/임베딩'만 모아 rank0에서 전역 정렬
        - (rank0) KMeans(K=world_size) → 최소크기 프루닝(centroid에서 먼 것 제거)
        - (rank0) 인터리브 전역 순서, 프루닝 전 클러스터 비율 → broadcast
        - sampler.update_indices(), rank_weights 저장, DataLoader 재생성
        """
        trainer = self.trainer
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not dist.is_initialized():
            world_size, rank = 1, 0
        else:
            world_size, rank = dist.get_world_size(), dist.get_rank()

        sampler = getattr(trainer, "_train_sampler_for_callback", None)
        assert sampler is not None and hasattr(sampler, "update_indices"), \
            "train sampler에 update_indices가 필요합니다."

        model.eval()

        # -------- (1) 각 랭크 로컬 인덱스: strided real + padding to equal size --------
        N   = len(self.dataset)
        ws  = world_size
        r   = rank
        micro = int(args.per_device_train_batch_size)
        chunk_mult = max(1, int(getattr(args, "badge_forward_chunk_mult", 1)))
        base_badge_chunk = micro * max(ws, 1) * chunk_mult
        drop_last = bool(getattr(args, "dataloader_drop_last", False))  # KMeans 단계에서만 사용

        if ws > 1:
            # 실제(real) 인덱스: r, r+ws, r+2ws, ...
            idx_real = list(range(r, N, ws))
            real_len = len(idx_real)

            # 타겟 길이: 모든 랭크 동일 (선택) 마이크로 배치 배수로 정렬
            target = math.ceil(N / ws)
            target = math.ceil(target / max(micro, 1)) * max(micro, 1)

            # 패딩 값: 마지막 real 인덱스가 있으면 그것, 없으면 N-1
            pad_val = idx_real[-1] if real_len > 0 else max(0, N - 1)
            idx_full = idx_real + [pad_val] * max(0, target - real_len)

            # 이 랭크가 실제로 임베딩을 계산할 서브셋(패딩 포함)
            local_ds = Subset(self.dataset, idx_full)
        else:
            idx_real = list(range(N))
            real_len = N
            target   = N
            local_ds = self.dataset

        # -------- (2) 임베딩 계산: 패딩은 계산하되, gather에는 'real'만 보냄 --------
        # remove_mean=True는 all-reduce로 전역 평균을 쓰는데, 패딩이 포함되면 평균이 왜곡될 수 있음.
        # 패딩 모드에서는 임시로 mean 제거를 끄고(raw/normalized 그대로 사용) 이후 KMeans에 투입.
        saved_remove_mean = getattr(self, "remove_mean", False)
        try:
            if ws > 1:
                self.remove_mean = False  # 패딩 모드: 평균 왜곡 방지
            micro = micro * ws * 4
            save_indices = idx_full if ws > 1 else idx_real
            save_real = real_len if ws > 1 else len(save_indices)
            emb_full_np = self.compute_badge_embeddings(
                model,
                local_ds,
                micro,
                device,
                forward_chunk_size=base_badge_chunk,
                sample_indices=save_indices,
                num_real_to_save=save_real,
            )  # (target, H)
        finally:
            self.remove_mean = saved_remove_mean
        self._flush_saved_grad_file(state.epoch)

        # real 부분만 사용하여 gather에 보냄
        if ws > 1:
            emb_local = torch.from_numpy(emb_full_np[:real_len]).to(device=device, dtype=torch.float32)
            idx_send  = torch.tensor(idx_real, dtype=torch.long, device=device)
        else:
            emb_local = torch.from_numpy(emb_full_np).to(device=device, dtype=torch.float32)
            idx_send  = torch.arange(0, N, dtype=torch.long, device=device)

        if ws > 1:
            # -------- (3) all_gather (가변 길이) --------
            idx_all = self._gather_varlen_1d(idx_send, device)       # np.int64, 길이 합=N
            emb_all = self._gather_varlen_2d(emb_local, device)      # np.float32, (N, H)

            # -------- (4) rank0에서 전역 정렬 --------
            if r == 0:
                order   = np.argsort(idx_all)
                emb_full = emb_all[order]                            # (N, H)
            else:
                emb_full = None

            # 전 랭크에 전파
            emb = self._broadcast(emb_full, src=0)
        else:
            emb = emb_full_np                                       # 단일 프로세스

        # -------- (5) rank0: KMeans/프루닝/인터리브 & 비율 계산 --------
        if r == 0:
            interleaved, diag, _ = _kmeans_uniform_equalize_interleave(
                emb,
                world_size=ws,
                micro=micro,
                drop_last=drop_last,
                seed=args.seed + int(state.epoch),
                prune_type = args.prune,
                sampling = args.sampling_type,
                ratio = args.ratio,
            )
            kept = len(interleaved)
            self._maybe_save_cluster_artifacts(diag, interleaved, N, state.epoch)

            print(diag)
            
            # 프루닝 "전" 클러스터 비율 → 랭크 가중치(평균 1 정규화)
            ratios = diag.get("cluster_ratios")
            size_ratios = diag.get("cluster_size_ratio")
            cluster_weights = diag.get("cluster_weights")
            sizes = diag["cluster_sizes"]

            rank_weights = [1.0 for size in sizes]
        else:
            interleaved, diag, rank_weights = None, None, None

        # -------- (6) 결과 브로드캐스트 --------
        interleaved = self._broadcast(interleaved, src=0)   # list[int]
        diag        = self._broadcast(diag,        src=0)   # dict
        rank_weights= self._broadcast(rank_weights, src=0)  # list[float]
        if dist.is_initialized():
            dist.barrier()

        print(f"Rank {r} after broadcast: got {len(interleaved) if interleaved is not None else 'None'} interleaved indices, diag={diag}, rank_weights={rank_weights}")

        # -------- (7) sampler 갱신 & DataLoader 재생성 --------
        sampler.update_indices(interleaved)
        
        import types
        trainer._train_sampler = sampler
        trainer._get_train_sampler = types.MethodType(lambda self: sampler, trainer)

        trainer.args.rank_weights = rank_weights
        trainer._rank_weight_tensor = torch.tensor(rank_weights, device=device, dtype=torch.float32)
        trainer.__dict__.pop("_train_dataloader", None)

        if r == 0:
            print(
                f"[Epoch {int(state.epoch)}] "
                f"sizes={diag['cluster_sizes']} ratios={[round(x,4) for x in diag['cluster_ratios']]} "
                f"m_min={diag['min_cluster_size']} m_adj={diag['m_adj']} "
                f"kept={diag['kept']} pruned={diag['pruned']} "
                f"rank_weights={[round(w,4) for w in rank_weights]}"
            )

        model.train()
