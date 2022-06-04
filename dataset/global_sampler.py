import math
from typing import TypeVar, Optional, Iterator
import os
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataset import Dataset
import torch.distributed as dist
import numpy as np
import random


T_co = TypeVar('T_co', covariant=True)

# This sampler is implemented for multi-GPU training, modified from the DistributedSampler of pytorch.
# The strategy follows VIGOR
class DistributedMiningSampler(DistributedSampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 128, mode = 'similarity', dim=1000, save_path = None) -> None:
        super(DistributedMiningSampler, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dim = dim
        self.batch_size = batch_size * self.num_replicas
        self.queue_length = len(self.dataset)
        self.current_size = len(self.dataset) // self.batch_size * self.batch_size
        self.current_indices = np.arange(self.current_size)
        self.queue_size = 1 # for computing moving average, not used in this implementation
        self.queue = np.zeros([self.queue_length, self.queue_size, self.dim, 2])
        self.queue_ptr = 0
        self.queue_counter = np.zeros(self.queue_length,dtype=np.int)
        self.save_path = save_path
        self.mining_start = 1
        self.mining_pool_size = min(40000, self.queue_length)
        self.mining_save_size = 100
        self.choice_pool = range(self.mining_save_size)
        self.mining_save = np.zeros([self.queue_length, self.mining_save_size],dtype=int)

        self.mode = mode

    def update(self, data_sat, data_grd, indexes):
        data_sat_norm = data_sat / np.linalg.norm(data_sat, axis=1, keepdims=True)
        data_grd_norm = data_grd / np.linalg.norm(data_grd, axis=1, keepdims=True)
        batch_size = data_sat.shape[0]
        # writing in distributed training style, complicated. Update the queue according to the previous index.
        for j in range(self.num_replicas):
            index_j = self.indices_out[j:self.current_size:self.num_replicas]

            for i in range(batch_size // self.num_replicas):
                index = index_j[self.queue_ptr + i]
                assert index == indexes[i + j * (batch_size // self.num_replicas)]
                self.queue[index, self.queue_counter[index] % self.queue_size, :, 0] = \
                data_sat_norm[i + j * (batch_size // self.num_replicas)]
                self.queue[index, self.queue_counter[index] % self.queue_size, :, 1] = \
                    data_grd_norm[i + j * (batch_size // self.num_replicas)]
                self.queue_counter[index] += 1
        self.queue_ptr = (self.queue_ptr + batch_size // self.num_replicas)

    def generate_indices_sim(self):
        self.queue_ptr = 0

        random.seed(7 + self.epoch)
        self.current_indices = np.arange(self.current_size)
        random.shuffle(self.current_indices)

        if self.epoch >= self.mining_start:
            assert self.mining_pool_size <= self.queue_length
            mining_pool = np.array(random.sample(range(self.queue_length), self.mining_pool_size),dtype=int)
            product_train = np.matmul(self.queue[:,:,:,1].mean(axis=1), np.transpose(self.queue[mining_pool,:,:,0].mean(axis=1)))
            product_index = np.argsort(product_train, axis=1)
            ranking = np.zeros(product_train.shape[0])
            # update mining pool
            for i in range(product_train.shape[0]):
                ranking[i] = product_train.shape[1] - 1 - np.where(mining_pool[product_index[i]] == i)[0]
                self.mining_save[i, :] = mining_pool[product_index[i, -self.mining_save_size:]]
            # randomly sample first half
            ori_list = self.current_indices[:self.current_size//2]
            self.current_indices = []
            # global hard mining for the other half
            for i in range(self.current_size//self.batch_size):
                index_s = i * (self.batch_size//2)
                index_e = index_s + min(self.batch_size//2, self.current_size//2 - index_s)
                self.current_indices.extend(ori_list[index_s:index_e])
                hard_list = []
                for j in range(index_s, index_e):
                    idx = random.choice(self.mining_save[ori_list[j]])
                    # random sampling until no overlap in the batch
                    while idx in ori_list[index_s:index_e] or idx in hard_list:
                        idx = random.choice(self.mining_save[ori_list[j]])
                    hard_list.append(idx)
                self.current_indices.extend(hard_list)
        self.current_indices = np.array(self.current_indices, dtype=int)
        assert len(self.current_indices) == self.current_size
        print('sampler updated!')

    def update_epoch(self):
        self.generate_indices_sim()
        if self.rank == 0:
            np.save(os.path.join(self.save_path,'queue.npy'), self.queue)
            np.save(os.path.join(self.save_path,'queue_counter.npy'), self.queue_counter)

    def load(self, path):
        self.queue_counter = np.load(os.path.join(path,'queue_counter.npy'))
        self.queue = np.load(os.path.join(path,'queue.npy'))

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.current_indices), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.current_indices)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.current_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.current_size]
        assert len(indices) == self.current_size

        # subsample
        self.indices_out = self.current_indices[indices].tolist()
        indices = indices[self.rank:self.current_size:self.num_replicas]
        # assert len(indices) == self.num_samples
        # print(indices)
        indices_out = self.current_indices[indices].tolist()
        # print(self.rank, len(indices), len(indices_out))

        return iter(indices_out)


class DistributedMiningSamplerVigor(DistributedSampler[T_co]):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 128, mode = 'similarity', dim=1000, save_path=None) -> None:
        super(DistributedMiningSamplerVigor, self).__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.dim = dim
        self.batch_size = batch_size * self.num_replicas
        self.queue_length = max(dataset.train_data_size, len(dataset.train_sat_cover_list))
        self.current_size = len(self.dataset) // self.batch_size * self.batch_size
        self.current_indices = np.arange(self.current_size)
        self.queue_size = 1 # for computing moving average, not used in this implementation
        self.queue = np.zeros([self.queue_length, self.queue_size, self.dim, 2])
        self.queue_ptr = 0
        self.queue_counter = np.zeros(self.queue_length,dtype=np.int)
        self.save_path = save_path
        self.mining_start = 1
        self.mining_pool_size = min(40000, len(dataset.train_sat_cover_list))
        self.mining_save_size = 100
        self.choice_pool = range(self.mining_save_size)
        self.mining_save = np.zeros([self.queue_length, self.mining_save_size],dtype=int)
        self.mode = mode
        # raise Exception

    def update(self, data_sat, data_grd, indexes):
        data_sat_norm = data_sat / np.linalg.norm(data_sat, axis=1, keepdims=True)
        data_grd_norm = data_grd / np.linalg.norm(data_grd, axis=1, keepdims=True)
        batch_size = data_sat.shape[0]
        # writing in distributed training style, complicated. Update the queue according to the previous index.
        for j in range(self.num_replicas):
            index_j = self.indices_out[j:self.current_size:self.num_replicas]

            for i in range(batch_size // self.num_replicas):
                index = index_j[self.queue_ptr + i] %len(self.dataset.train_sat_cover_list)
                assert indexes[i + j * (batch_size // self.num_replicas)] in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[index]]
                self.queue[index, self.queue_counter[index] % self.queue_size, :, 0] = \
                data_sat_norm[i + j * (batch_size // self.num_replicas)]
                self.queue[indexes[i + j * (batch_size // self.num_replicas)], self.queue_counter[index] % self.queue_size, :, 1] = \
                    data_grd_norm[i + j * (batch_size // self.num_replicas)]
                self.queue_counter[index] += 1
        self.queue_ptr = (self.queue_ptr + batch_size // self.num_replicas)

    def generate_indices_sim(self):
        self.queue_ptr = 0

        random.seed(7 + self.epoch)
        self.current_indices = np.arange(self.current_size) %len(self.dataset.train_sat_cover_list)
        random.shuffle(self.current_indices)

        if self.epoch >= self.mining_start:
            assert self.mining_pool_size <= self.queue_length
            mining_pool = np.array(random.sample(range(len(self.dataset.train_sat_cover_list)), self.mining_pool_size),dtype=int)
            product_train = np.matmul(self.queue[:,:,:,1].mean(axis=1), np.transpose(self.queue[mining_pool,:,:,0].mean(axis=1)))
            product_index = np.argsort(product_train, axis=1)
            # update mining pool
            for i in range(product_train.shape[0]):
                self.mining_save[i, :] = mining_pool[product_index[i, -self.mining_save_size:]]
            # randomly sample the first half
            ori_list = self.current_indices[:self.current_size//2]
            self.current_indices = []
            # global hard mining for the other half
            for i in range(self.current_size//self.batch_size):
                index_s = i * (self.batch_size//2)
                index_e = index_s + min(self.batch_size//2, self.current_size//2 - index_s)
                self.current_indices.extend(ori_list[index_s:index_e])
                hard_list = []
                for j in range(index_s, index_e):
                    grd_id = random.choice(self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[ori_list[j]]])
                    idx = int(random.choice(self.mining_save[grd_id]))
                    # keep random sampling until there is no overlap in the batch, hard coded as VIGOR is complicated
                    while True:
                        flag = False
                        for grd_idx in self.dataset.train_sat_cover_dict[self.dataset.train_sat_cover_list[idx]]:
                            if not self.dataset.check_overlap(ori_list[index_s:index_e],
                                                          grd_idx) or not self.dataset.check_overlap(hard_list, grd_idx):
                                flag = True
                        if flag:
                            idx = random.choice(self.mining_save[grd_id])
                        else:
                            break
                    hard_list.append(idx)
                self.current_indices.extend(hard_list)
        self.current_indices = np.array(self.current_indices, dtype=int)
        assert len(self.current_indices) == self.current_size
        print('sampler updated!')

    def update_epoch(self):
        # if self.epoch >= self.mining_start:
        self.generate_indices_sim()
        if self.rank == 0:
            np.save(os.path.join(self.save_path, 'queue.npy'), self.queue)
            np.save(os.path.join(self.save_path, 'queue_counter.npy'), self.queue_counter)

    def load(self, path):
        self.mining_start = 0
        self.queue_counter = np.load(os.path.join(path, 'queue_counter.npy'))
        self.queue = np.load(os.path.join(path, 'queue.npy'))

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.current_indices), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.current_indices)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.current_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.current_size]
        assert len(indices) == self.current_size

        # subsample
        self.indices_out = self.current_indices[indices].tolist()
        indices = indices[self.rank:self.current_size:self.num_replicas]
        # assert len(indices) == self.num_samples
        # print(indices)
        indices_out = self.current_indices[indices].tolist()
        # print(self.rank, len(indices), len(indices_out))

        return iter(indices_out)