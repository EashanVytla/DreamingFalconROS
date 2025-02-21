import torch
import numpy as np
from torch import multiprocessing as mp
from utils import normalize

class ReplayBuffer:
    def __init__(self, config):
        self.device = config.device

        self.counter = mp.Value('i', 0)
        self.ptr = mp.Value('i', 0)
        self.lock = mp.Lock()
    
        self.capacity = config.replay_buffer.capacity
        self.norm_ranges = config.normalization

        self.states = torch.zeros((self.capacity, 12), dtype=torch.float32, device=self.device).share_memory_()
        self.actions = torch.zeros((self.capacity, config.force_model.action_dim), dtype=torch.float32, device=self.device).share_memory_()
        self.dt = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device).share_memory_()

    def add(self, state: torch.Tensor, action: torch.Tensor, dt: float):
        with self.lock:
            # norm_x_t = torch.zeros_like(state, device=self.device)
            # norm_x_t[0:3] = state[0:3]
            # norm_x_t[3:6] = normalize(state[3:6], self.norm_ranges.velo_min, self.norm_ranges.velo_max)     # Velocity: +- 20
            # norm_x_t[6:9] = normalize(state[6:9], self.norm_ranges.euler_min, self.norm_ranges.euler_max)     # Euler Angles: +- pi
            # norm_x_t[9:12] = normalize(state[9:12], self.norm_ranges.omega_min, self.norm_ranges.omega_max)       # Rotation Rates: +- pi/4
            norm_act = normalize(action, self.norm_ranges.act_min, self.norm_ranges.act_max)

            self.states[self.ptr.value, :] = state
            self.actions[self.ptr.value, :] = norm_act
            self.dt[self.ptr.value] = dt

            # TODO: FIX THIS! This isn't a true FIFO buffer
            self.ptr.value = (self.ptr.value + 1) % self.capacity
            self.counter.value = min(self.counter.value + 1, self.capacity)

    def get_len(self) -> int:
        with self.lock:
            return self.counter.value

    def sample(self, batch_size: int, sequence_length: int):
        with self.lock:
            available_data = self.counter.value
            if available_data < sequence_length:
                raise ValueError(f"Not enough data in buffer. Have {available_data}, need {sequence_length}")

            start_indices = np.random.randint(
                self.ptr.value - self.counter.value, 
                self.ptr.value - sequence_length + 1, 
                size=(batch_size)
            ) % self.capacity
            
            batch_states = torch.zeros((batch_size, sequence_length, self.states.shape[-1]), device=self.device)
            batch_actions = torch.zeros((batch_size, sequence_length, self.actions.shape[-1]), device=self.device)
            batch_dts = torch.zeros((batch_size, sequence_length), device=self.device)
            
            for batch_idx, start_idx in enumerate(start_indices):
                seq_start = start_idx
                seq_end = seq_start + sequence_length
                
                batch_states[batch_idx] = self.states[seq_start:seq_end, :]
                batch_actions[batch_idx] = self.actions[seq_start:seq_end, :]
                batch_dts[batch_idx] = self.dt[seq_start:seq_end]

            return batch_dts, batch_states, batch_actions