import torch
from torch.utils.data import Sampler
import numpy as np
import math
import random
import torch.distributed as dist
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class CustomDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, seed=0):
        self.dataset = dataset
        self.num_samples = len(self.dataset)
        self.seed = seed

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank

        self.special_rank = num_replicas - 1  # Only this rank gets (i % N == N-1) indices

        # Precompute the groupings
        self.special_indices = [i for i in range(self.num_samples) if i % num_replicas == self.special_rank]
        self.normal_indices = [i for i in range(self.num_samples) if i % num_replicas != self.special_rank]

        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Shuffle each set independently
        special_perm = torch.randperm(len(self.special_indices), generator=g).tolist()
        normal_perm = torch.randperm(len(self.normal_indices), generator=g).tolist()

        shuffled_special = [self.special_indices[i] for i in special_perm]
        shuffled_normal = [self.normal_indices[i] for i in normal_perm]

        if self.rank == self.special_rank:
            return iter(shuffled_special)
        else:
            # Distribute remaining data across other ranks
            return iter(shuffled_normal[self.rank::self.num_replicas - 1])

    def __len__(self):
        if self.rank == self.special_rank:
            return len(self.special_indices)
        else:
            # Other ranks share normal indices
            return (len(self.normal_indices) + self.num_replicas - 2) // (self.num_replicas - 1)

    def set_epoch(self, epoch):
        self.epoch = epoch


class RatioInterleavedBatchSampler(Sampler):
    def __init__(self, dataset, ratios, per_device_train_batch_size, seed=0):
        self.dataset = dataset
        self.total_size = len(dataset)
        self.ratios = ratios  # e.g., [1, 2, 1]
        self.num_groups = len(ratios)
        self.seed = seed
        self.epoch = 0

        self.num_gpus = torch.cuda.device_count()
        self.per_device_train_batch_size = per_device_train_batch_size
        self.unit = sum(ratios)

        if self.unit != self.num_gpus:
            raise ValueError(
                f"Sum of ratios ({self.unit}) must equal number of GPUs ({self.num_gpus})"
            )

        self.batch_size = self.num_gpus * self.per_device_train_batch_size
        self.samples_per_group = [
            r * self.per_device_train_batch_size for r in self.ratios
        ]

        # print(self.batch_size, self.total_size)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # 그룹 나누기
        groups = [[] for _ in range(self.num_groups)]
        gpu_groups = [[] for _ in range(self.num_gpus)]
        gpu_groups_range = [[] for _ in range(self.num_gpus)]
        
        len_for_one_gpu = self.total_size // self.num_gpus

        for i in range(self.total_size):
            # gpu_groups[i // len_for_one_gpu].append(i) #group by range
            gpu_groups[i % self.num_gpus].append(i) #stripe

        if torch.distributed.get_rank() == 0:
            for i in range(self.num_gpus):
                print(i, len(gpu_groups[i]), len(gpu_groups_range[i]))

        # 셔플
        rng = random.Random(self.seed + self.epoch)
        idx = 0
        for i,r in enumerate(self.ratios):
            for j in range(idx, idx+r):
                groups[i] = groups[i] + gpu_groups[j]
            idx = idx + r

        for i, g in enumerate(groups):
            g.sort(key=lambda idx: len(self.dataset[idx]['input_ids']), reverse=True)
        
        # print(len_for_one_gpu)
        idx_list = [i for i in range(len_for_one_gpu)]
        rng.shuffle(idx_list)

        sorted_groups = []
        for g, ratio in zip(groups, self.ratios):
            if ratio == 1:
                shuffled_g = [g[i] for i in idx_list]
            else:
                shuffled_g = []
                for i in idx_list:
                    block = [g[ratio * i + j] for j in range(ratio)]
                    rng.shuffle(block)
                    shuffled_g.extend(block)
            sorted_groups.append(shuffled_g)

        groups = sorted_groups
        
        ## general shuffling > random
        # for g in groups:
        #     print(len(g))
        #     rng.shuffle(g)

        # 가능한 최대 step 수 계산
        min_steps = self.total_size // self.batch_size

        pointers = [0] * self.num_groups
        result = []

        for _ in range(min_steps):
            batch = []
            for group_id, num_samples in enumerate(self.samples_per_group):
                batch.extend(groups[group_id][pointers[group_id]:pointers[group_id] + num_samples])
                pointers[group_id] += num_samples
            result.extend(batch)

        return iter(result)

    def __len__(self):
        return self.total_size


class NewSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """

    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
    
    def update_indices(self, new_indices):
        self.indices = new_indices

class RangeSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, world_size):
        self.dataset = dataset
        self.world_size = world_size
        self.num_samples = len(dataset)

        # chunk size: dataset 길이 나누기 GPU 개수 (몫, 나머지 고려)
        self.chunk_size = math.ceil(self.num_samples / self.world_size)

        # chunk 단위로 인덱스 나누고 concat
        self.indices = []
        for i in range(self.world_size):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, self.num_samples)
            self.indices.extend(range(start, end))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

    def update_indices(self, new_indices):
        self.indices = new_indices

class CustomSequentialSampler(Sampler):
    def __init__(self, dataset, per_device_train_batch_size, seed):
        self.dataset = dataset
        self.batch_size = per_device_train_batch_size
        self.seed = seed
        self.indices = list(range(len(dataset)))

    def update_indices(self, new_indices):
        self.indices = new_indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
    
class InterleavedSpecialSampler(Sampler):
    def __init__(self, dataset, N, seed=0, partition=0):
        self.dataset = dataset
        self.N = N
        self.seed = seed
        self.epoch = 0
        self.total_size = len(dataset)
        self.partition = partition

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        special = [i for i in range(self.total_size) if i % self.N >= self.N - self.partition]
        normal = [i for i in range(self.total_size) if i % self.N < self.N - self.partition]

        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(normal)
        rng.shuffle(special)

        result = []
        n_idx, s_idx = 0, 0
        while len(result) < self.total_size:
            for _ in range(self.N - 1):
                if n_idx < len(normal):
                    result.append(normal[n_idx])
                    n_idx += 1
            if s_idx < len(special):
                result.append(special[s_idx])
                s_idx += 1

        return iter(result[:self.total_size])

    def __len__(self):
        return self.total_size


class DistributedStridedSampler(Sampler):
    """
    전역 순서에서 rank 오프셋으로 시작해 world_size 간격으로 뽑는 샘플러.
    - update_indices(): 전역 순서를 외부에서 완전히 교체 (프루닝/인터리브 등)
    - set_active_subset(): 전역 순서 중 일부만 사용
    - set_epoch(): (선택) epoch별 셔플
    """
    def __init__(
        self,
        dataset,
        num_replicas: int = None,
        rank: int = None,
        drop_last: bool = False,
        seed: int = 42,
        pad_to_equal_size: bool = False,
        prepartitioned: bool = True,
        sampling: str = None,
    ):
        self.dataset = dataset
        self.N = len(dataset)

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.ws = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.pad_to_equal_size = pad_to_equal_size
        self.prepartitioned = prepartitioned
        self.sampling = sampling

        # 전역 기본 순서(외부 재정렬의 베이스). 기본은 0..N-1
        self._base_order = list(range(self.N))
        # 활성 서브셋(None이면 전체 사용). 값이 있으면 _base_order의 인덱스가 아니라
        # "데이터셋 글로벌 인덱스" 리스트를 직접 담는다.
        self._active = None
        # 현재 에폭에서 실제로 사용할 전역 순서
        self._order = list(self._base_order)

        self._build_local_indices()

    @staticmethod
    def _shuffle_preserving_mod(order, ws: int, seed: int):
        """
        전역 순서 order를 모듈로(ws) 버킷별로만 셔플한 뒤,
        다시 interleave(라운드로빈)로 합쳐서 반환.
        - pos % ws 는 불변
        - 버킷 길이 달라도 안전하게 병합
        """
        if ws <= 1 or not order:
            return list(order)

        # 1) 버킷 분할 (현재 전역 pos 기준)
        buckets = [[] for _ in range(ws)]
        for pos, idx in enumerate(order):
            buckets[pos % ws].append(idx)

        # 2) 버킷 내부 셔플(결정론)
        for r in range(ws):
            rng = random.Random(seed + r)
            rng.shuffle(buckets[r])

        # 3) 라운드로빈 병합
        out = []
        maxlen = max(len(b) for b in buckets) if buckets else 0
        for j in range(maxlen):
            for r in range(ws):
                if j < len(buckets[r]):
                    out.append(buckets[r][j])
        return out
    
    # ---------------- core ---------------- #
    def _current_base(self):
        return self._active if getattr(self, "_active", None) is not None else self._base_order

    def _rebuild_order_for_epoch(self):
        base = self._current_base()
        # self._order = list(base)
        seed = int(self.seed) + int(self.epoch) + 12345
        if self.sampling == "rand":
            rng=random.Random(seed)
            self._order = list(base)
            rng.shuffle(self._order)
        else:
            self._order = self._shuffle_preserving_mod(list(base), self.ws, seed)


    def _build_local_indices(self):
        # 전역 순서 구성
        self._rebuild_order_for_epoch()
        N_eff = len(self._order)                     # ★ 프루닝/인터리브 반영된 길이

        # 등간격 선택
        if self.prepartitioned:
            # rng = random.Random(self.seed + self.epoch)
            # rng.shuffle(self._order)
            local = list(self._order)
        else:
            local = self._order[self.rank:N_eff:self.ws]

        # drop_last: floor(N_eff/ws)
        if self.drop_last and not self.prepartitioned:
            target = N_eff // self.ws
            local = local[:target]

        # pad_to_equal_size: ceil(N_eff/ws)
        if self.pad_to_equal_size and not self.prepartitioned:
            target = math.ceil(N_eff / self.ws) if self.ws > 0 else len(local)
            if len(local) == 0:
                pad_val = self._order[-1] if N_eff > 0 else 0
                local = [pad_val] * target
            elif len(local) < target:
                pad_val = local[-1]
                local = local + [pad_val] * (target - len(local))

        self._local = local

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._build_local_indices()

    def update_indices(self, new_global_order):
        """
        프루닝/인터리브 결과로 바뀐 '전역 순서'를 그대로 반영 (가변 길이 허용).
        """
        # ★ assert 제거
        if not new_global_order:
            self._base_order = list(range(self.N))  # fallback
        else:
            self._base_order = list(map(int, new_global_order))
        self._active = None                         # 전역 교체이므로 subset 해제
        self._build_local_indices()

    def set_active_subset(self, active_global_indices):
        """
        전역 순서 일부만 임시 사용하고 싶을 때 (원 전역 순서 유지).
        """
        self._active = None if active_global_indices is None else list(map(int, active_global_indices))
        self._build_local_indices()

    def __len__(self):
        return len(self._local)                     # ★ 현 랭크가 실제로 순회할 길이
    
    def __iter__(self):
        # print(self._local)
        return iter(self._local)
    