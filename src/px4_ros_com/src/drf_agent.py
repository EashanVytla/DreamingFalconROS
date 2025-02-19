#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, ActuatorOutputs
import torch
from utils import quat_to_euler, AttrDict, get_DCM, normalize, denormalize
from replay_buffer import ReplayBuffer
import yaml
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from world_model import WorldModel
from sequence_scheduler import AdaptiveSeqLengthScheduler
from message_filters import Subscriber, ApproximateTimeSynchronizer
import os
import time

class Storage(Node):
    def __init__(self, buffer) -> None:
        super().__init__('storage_node')

        with open('config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)

        self.config = AttrDict.from_dict(config_dict)
        self.buffer = buffer

        self.state_dim = self.config.force_model.state_dim
        self.action_dim = self.config.force_model.action_dim
        self.device = self.config.device
        self.state = torch.zeros((self.state_dim), dtype=torch.float32, device=self.device)
        self.action = torch.zeros((self.action_dim), dtype=torch.float32, device=self.device)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.odometry_subscriber = Subscriber(
            self,
            VehicleOdometry,
            'fmu/out/vehicle_odometry',
            qos_profile=qos_profile,
        )

        self.actuator_subscriber = Subscriber(
            self,
            ActuatorOutputs,
            'fmu/out/actuator_outputs',
            qos_profile=qos_profile,
        )

        self.time_sync = ApproximateTimeSynchronizer(
            [self.odometry_subscriber, self.actuator_subscriber],
            queue_size=5,
            slop=0.05,
            allow_headerless=True
        )

        self.time_sync.registerCallback(self.data_callback)

        self.last_timestamp = None

    def data_callback(self, odo_msg, act_msg):
        current_timestamp = odo_msg.timestamp
        dt = 0.0
        if self.last_timestamp is not None:
            dt = (current_timestamp - self.last_timestamp) / 1e6  # Convert microseconds to seconds
            if (dt < self.config.physics.refresh_rate - (0.05 * self.config.physics.refresh_rate)):
                return

        self.state[6:9] = quat_to_euler(torch.tensor(odo_msg.q, dtype=torch.float32, device=self.device), device=self.device)
        self.state[0:3] = torch.tensor(odo_msg.position, dtype=torch.float32, device=self.device)
        self.state[3:6] = torch.matmul(
            get_DCM(self.state[6], self.state[7], self.state[8]), 
            torch.from_numpy(odo_msg.velocity).to(dtype=torch.float32, device=self.device)
        )

        self.state[9:12] = torch.tensor(odo_msg.angular_velocity, dtype=torch.float32, device=self.device)

        self.action = torch.tensor(act_msg.output[:4], dtype=torch.float32, device=self.device)

        if self.buffer.get_len() <= self.config.replay_buffer.start_learning + 2:
            # print("adding")
            self.buffer.add(self.state, self.action, dt)
        self.last_timestamp = current_timestamp

class WorldModelLearning():
    def __init__(self, buffer):
        self.buffer = buffer
        self.log_directory = "runs/2-17"
        self.model_directory = "models/2-17"

        with open('config.yaml', 'r') as file:
            config_dict = yaml.safe_load(file)

        self.config = AttrDict.from_dict(config_dict)

        self.device = self.config.device
        self.norm_ranges = self.config.normalization

        self.model = WorldModel(self.config, torch.device(self.config.device)).to(self.config.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)

        if self.config.training.cos_lr:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=25, eta_min=self.config.training.min_lr, verbose=True
            )

        self.writer = SummaryWriter(self.log_directory)

        self.seq_scheduler = AdaptiveSeqLengthScheduler(
            initial_length=self.config.training.init_seq_len, 
            max_length=self.config.training.max_seq_len, 
            patience=self.config.training.seq_patience, 
            threshold=self.config.training.seq_sch_thresh, 
            model=self.model, 
            config=self.config
        )
        
    def compute_gradient_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def compute_weight_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def print_gradient_norms(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.data.norm().item()
                print(f"{name:30s} | grad norm: {grad_norm:.2e} | param norm: {param_norm:.2e}")
    
    def save_model(self):
        state = {
            "state_dict": self.model.state_dict()
        }

        os.makedirs(self.model_directory, exist_ok=True)

        torch.save(state, os.path.join(self.model_directory, "model.pt"))
        print("Model saved!")

    def close_writer(self):
        self.writer.close()

    def train(self):
        batch_count = 0
        while(True):
            self.optimizer.zero_grad()
            if self.buffer.get_len() > self.config.replay_buffer.start_learning:
                dts, states, actions = self.buffer.sample(self.config.training.batch_size, 2)
                pred_traj = self.model.rollout(dts, states[:,0,:], actions)

                pred_traj_norm = torch.zeros((pred_traj.shape[0], pred_traj.shape[1], 6), dtype=torch.float32, device=self.device)
                pred_traj_norm[:, :, 0:3] = normalize(pred_traj[:, :, 3:6], self.norm_ranges.velo_min, self.norm_ranges.velo_max)
                pred_traj_norm[:, :, 3:6] = normalize(pred_traj[:, :, 9:12], self.norm_ranges.omega_min, self.norm_ranges.omega_max)
                loss = self.model.loss(pred_traj_norm[:,1:,:], torch.concat((states[:,1:,3:6], states[:,1:, 9:12]), dim=-1))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.seq_scheduler.step(loss.item())

                if batch_count % 25 == 0:
                    x_dn = torch.zeros_like(states)
                    x_dn[:, :, 0:3] = states[:, :, 0:3]
                    x_dn[:, :, 3:6] = denormalize(states[:, :, 3:6], self.norm_ranges.velo_min, self.norm_ranges.velo_max)
                    x_dn[:, :, 6:9] = denormalize(states[:, :, 6:9], self.norm_ranges.euler_min, self.norm_ranges.euler_max)
                    x_dn[:, :, 9:12] = denormalize(states[:, :, 9:12], self.norm_ranges.omega_min, self.norm_ranges.omega_max)
                
                    print(f"Prediction: {torch.mean(pred_traj, dim=(0,1))}")
                    print(f"Truth: {torch.mean(x_dn, dim=(0,1))}")
                    print(f"Error: {torch.mean(pred_traj - x_dn, dim=(0,1))}")
                
                    grad_norm = self.compute_gradient_norm()
                    self.writer.add_scalar("Norms/gradient_norm", grad_norm, batch_count)
                    weight_norm = self.compute_weight_norm()
                    self.writer.add_scalar("Norms/weight_norm", weight_norm, batch_count)
                    self.writer.add_scalar("Loss/train", loss, batch_count)
                
                batch_count += 1
            else:
                print(f"Not enough data yet: {self.buffer.get_len()}")
                time.sleep(1.0)
            

def wm_train_process_fn(buffer):
    try:
        wm_learner = WorldModelLearning(buffer)
        wm_learner.train()
    except KeyboardInterrupt:
        print("Training process interrupted")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        wm_learner.save_model()
        if 'wm_learner' in locals():
            wm_learner.close_writer()

def main(args=None) -> None:
    print('Starting storage node...')
    
    rclpy.init(args=args)

    with open('config.yaml', 'r') as file:
        config_dict = yaml.safe_load(file)

    config = AttrDict.from_dict(config_dict)

    if config.device == "cuda":
        mp.set_start_method('spawn', force=True)
    
    buffer = ReplayBuffer(config)

    storage_node = Storage(buffer)
    train_process = mp.Process(target=wm_train_process_fn, args=[buffer])
    train_process.start()
    
    try:
        rclpy.spin(storage_node)
    except KeyboardInterrupt:
        print("keyboard interrupt")
    finally:
        train_process.terminate()
        train_process.join()
        storage_node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
