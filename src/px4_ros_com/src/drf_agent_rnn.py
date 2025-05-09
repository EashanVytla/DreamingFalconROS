#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, ActuatorOutputs, ActuatorMotors
import torch
from utils import quat_to_euler, AttrDict, denormalize, unwrap_angle, hard_update, diff_flag, flag
from replay_buffer import ReplayBuffer
import yaml
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from models import WorldModelRNN, Actor, Critic
from sequence_scheduler import AdaptiveSeqLengthScheduler
from message_filters import Subscriber, ApproximateTimeSynchronizer
import os
from scipy.spatial.transform import Rotation as R
import time
import re
from datetime import datetime
import copy
import json
import threading
import torch.nn.functional as F

# THIS IS A TEST COMMENT

class Storage(Node):
    def __init__(self, buffer, actor) -> None:
        super().__init__('storage_node')
        self.declare_parameter('config_file', 'config.yaml')
        self.config_file = self.get_parameter('config_file').value

        with open(self.config_file, 'r') as file:
            config_dict = yaml.safe_load(file)

        self.config = AttrDict.from_dict(config_dict)
        self.buffer = buffer
        self.actor = actor

        self.state_dim = self.config.force_model.state_dim
        self.action_dim = self.config.force_model.action_dim
        self.device = self.config.device

        '''
        State:
            1) Position (NED) 0-3
            2) Velocity (body) 12-15
            5) Attitude (Euler) 6-9
            6) Angular Velocity (body) 9-12
        '''

        self.state = torch.zeros((12), dtype=torch.float32, device=self.device)
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

        self.actuator_publisher = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)

        self.start_time = time.time() - 5.0
        self.start_act = False

        self.last_timestamp = None

    def data_callback(self, odo_msg, act_msg):
        current_timestamp = odo_msg.timestamp
        dt = 0.0
        if self.last_timestamp is not None:
            dt = (current_timestamp - self.last_timestamp) / 1e6
            if (dt < self.config.physics.refresh_rate - (0.05 * self.config.physics.refresh_rate)):
                return

        self.state[0:3] = torch.tensor(odo_msg.position, dtype=torch.float32, device=self.device)

        self.state[3:6] = torch.matmul(
            torch.tensor(R.from_quat(odo_msg.q, scalar_first=True).as_matrix().T, dtype=torch.float32, device=self.device),
            torch.tensor(odo_msg.velocity, dtype=torch.float32, device=self.device)
        )

        self.state[6:9] = quat_to_euler(odo_msg.q, device=self.device)

        if not hasattr(self, 'prev_yaw'):
            self.prev_yaw = self.state[8].clone()
        else:
            self.state[8] = unwrap_angle(self.state[8], self.prev_yaw)
            self.prev_yaw = self.state[8].clone()

        self.state[9:12] = torch.tensor(odo_msg.angular_velocity, dtype=torch.float32, device=self.device)

        self.action = torch.tensor(act_msg.output[:4], dtype=torch.float32, device=self.device)

        # if self.buffer.get_len() < 101:
        self.buffer.add(self.state, self.action, dt)
        self.last_timestamp = current_timestamp

        # if time.time() - self.start_time > self.config.chirp_timeout:
        #     if not self.start_act:
        #         print("Starting acting")
        #         self.start_act = True
        #     # This will change if we decide to add back in normalization
        #     act, _ = self.actor(self.state.unsqueeze(0))
        #     msg = ActuatorMotors()
        #     print(f"Act Shape: {act}")
        #     msg.control = [act[0,0].item(), 
        #                    act[0,1].item(), 
        #                    act[0,2].item(), 
        #                    act[0,3].item(), 
        #                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #     self.actuator_publisher.publish(msg)

class Learner():
    def __init__(self, buffer, actor, config_file):
        self.buffer = buffer

        config_index = re.search(r'config_(\d+)\.yaml$', config_file)
        self.run_index = int(config_index.group(1)) if config_index else 0

        self.log_directory = f"runs/run_{self.run_index}"
        self.model_directory = f"models/run_{self.run_index}"

        with open(config_file, 'r') as file:
            config_dict = yaml.safe_load(file)

        self.config = AttrDict.from_dict(config_dict)

        self.history = self.config.rnn_model.history

        pos_threshold = 100.0  # Max position deviation in meters
        vel_threshold = 30.0   # Max velocity in m/s
        att_threshold_z = 6.0   # Max attitude (rad)
        att_threshold_xy = 1.5
        ang_vel_threshold = 2.0  # Max angular velocity (rad/s)

        self.thresholds = torch.tensor([
            pos_threshold, pos_threshold, pos_threshold,     # Position (xyz)
            vel_threshold, vel_threshold, vel_threshold,        # Velocity (uvw)
            att_threshold_xy, att_threshold_xy, att_threshold_z,        # Attitude (euler angles)
            ang_vel_threshold, ang_vel_threshold, ang_vel_threshold         # Angular velocity
        ], device=self.config.device)

        self.device = self.config.device
        self.norm_ranges = self.config.normalization

        self.world_model = WorldModelRNN(self.config, torch.device(self.config.device)).to(self.config.device)

        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.config.training.lr, weight_decay=self.config.training.weight_decay)

        if self.config.training.cos_lr:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.wm_optimizer, T_max=25, eta_min=self.config.training.min_lr, verbose=True
            )

        self.writer = SummaryWriter(self.log_directory)

        self.seq_scheduler = AdaptiveSeqLengthScheduler(
            initial_length=self.config.training.init_seq_len, 
            max_length=self.config.training.max_seq_len, 
            patience=self.config.training.seq_patience, 
            threshold=self.config.training.seq_sch_thresh, 
            model=self.world_model, 
            config=self.config
        )

        self.batch_count = 0
        self.target_state = torch.tensor([0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32, device=self.config.device)

        self.critic = Critic(self.config, self.config.device)
        self.critic_copy = copy.deepcopy(self.critic)
        self.beh_updates = 0

        self.actor = actor

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.behavior_learning.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.behavior_learning.critic_lr)

        for name, param in actor.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, n=name: print(f"NaN detected in gradient of {n}") if torch.isnan(grad).any() else None)

        
    def compute_gradient_norm(self):
        total_norm = 0
        for p in self.world_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def compute_weight_norm(self):
        total_norm = 0
        for p in self.world_model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def print_gradient_norms(self):
        for name, param in self.world_model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                param_norm = param.data.norm().item()
                print(f"{name:30s} | grad norm: {grad_norm:.2e} | param norm: {param_norm:.2e}")
    
    def save_wm(self):
        state = {
            "state_dict": self.world_model.state_dict(),
            # **norm_stats
        }

        os.makedirs(self.model_directory, exist_ok=True)

        torch.save(state, os.path.join(self.model_directory, "model.pt"))
        print("Model saved!")

    def close_writer(self):
        self.writer.close()

    def validate(self, save_to_table=False):
        self.world_model.eval()
        with torch.no_grad():
            batch_size = 1
            if save_to_table:
                batch_size = 128
            print("Validating...")
            rollout_len = 32
            dts, states, actions = self.buffer.sample(batch_size, self.history + rollout_len)
            pred_traj = self.world_model.rollout(dts, states, actions, rollout_len)

            if self.norm_ranges.norm:
                states[:, :, 3:6] = denormalize(states[:, :, 3:6], self.norm_ranges.velo_min, self.norm_ranges.velo_max)
                states[:, :, 6:9] = denormalize(states[:, :, 6:9], self.norm_ranges.euler_min, self.norm_ranges.euler_max)
                states[:, :, 9:12] = denormalize(states[:, :, 9:12], self.norm_ranges.omega_min, self.norm_ranges.omega_max)

            # Absolute error
            abs_error = torch.abs(pred_traj - states[:,self.history+1:,3:])

            # Calculate squared error for RMSE
            squared_error = torch.square(pred_traj - states[:,self.history+1:,3:])
            
            # Calculate means
            truth_mean = torch.mean(states[:,self.history+1:,3:], dim=(0,1))
            pred_mean = torch.mean(pred_traj, dim=(0,1))
            error_mean = torch.mean(abs_error, dim=(0,1))
        
            # Calculate RMSE per state dimension
            rmse = torch.sqrt(torch.mean(squared_error, dim=(0,1)))
            
            # Calculate overall RMSE for key state groups
            # position_rmse = torch.sqrt(torch.mean(squared_error[:,:,0:3]))
            velocity_rmse = torch.sqrt(torch.mean(squared_error[:,:,0:3]))
            attitude_rmse = torch.sqrt(torch.mean(squared_error[:,:,3:6]))
            angular_vel_rmse = torch.sqrt(torch.mean(squared_error[:,:,6:9]))
            overall_rmse = torch.sqrt(torch.mean(squared_error))
            
            print(f"Truth Mean: {truth_mean}")
            print(f"Prediction Mean: {pred_mean}")
            print(f"Error: {error_mean}")
            print(f"RMSE per dimension: {rmse}")
            # print(f"Position RMSE: {position_rmse:.6f}")
            print(f"Velocity RMSE: {velocity_rmse:.6f}")
            print(f"Attitude RMSE: {attitude_rmse:.6f}")
            print(f"Angular Velocity RMSE: {angular_vel_rmse:.6f}")
            print(f"Overall RMSE: {overall_rmse:.6f}")
        
            if save_to_table:
                print("Saving to table!")
                # Convert tensor to list for JSON serialization
                results = {
                    "run_index": self.run_index,
                    "timestamp": datetime.now().isoformat(),
                    "truth_mean": truth_mean.cpu().tolist(),
                    "pred_mean": pred_mean.cpu().tolist(),
                    "error_mean": error_mean.cpu().tolist(),
                    "rmse_per_dim": rmse.cpu().tolist(),
                    # "position_rmse": position_rmse.item(),
                    "velocity_rmse": velocity_rmse.item(),
                    "attitude_rmse": attitude_rmse.item(),
                    "angular_vel_rmse": angular_vel_rmse.item(),
                    "overall_rmse": overall_rmse.item()
                }

                json_file = os.path.join(os.getcwd(), "experiment_results.json")
                print(f"Json file path: {json_file}")
                try:
                    # Load existing results if file exists
                    if os.path.exists(json_file):
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                    else:
                        print("file doesn't exist")
                        data = {"experiments": []}

                    # Add new results
                    data["experiments"].append(results)

                    # Save updated results
                    with open(json_file, 'w') as f:
                        json.dump(data, f, indent=4)
                    print(f"Results saved to {json_file}")

                except Exception as e:
                    print(f"Error saving results: {e}")
        self.world_model.train()

    def beh_train_step(self):
        _, states, _ = self.buffer.sample(self.config.behavior_learning.batch_size, 1)

        traj = [states[:,0,:]]
        log_probs = []
        acts = []
        dts = torch.ones((states.shape[0],1), dtype=torch.float32, device=self.device) * self.config.physics.refresh_rate
        c_masks = [torch.ones((states.shape[0], 1), dtype=torch.float32, device=self.device)]

        for t in range(self.config.behavior_learning.horizon):
            c_mask = flag(traj[t], self.thresholds) * c_masks[-1]
            if (c_mask < 0.5).any():
                print(f"Continue mask activated")

            if torch.isnan(traj[t]).any():
                print(f"Nan detected in state! t={t}")
            # print(f"Traj[t] = {traj[t]}")
            act, log_prob = self.actor(traj[t])
            act = c_mask * act
            log_prob = c_mask * log_prob

            if act is None or log_prob is None:
                print(f"None act or log_prob t={t}")

            if torch.isnan(act).any():
                print(f"Nan detected in action t={t}!")

            self.world_model.eval()
            next_state = self.world_model.predict(dts, traj[-1], act) * c_mask
            self.world_model.train()

            if torch.isnan(next_state).any():
            # if True:
                print(f"Nan detected in next state! t={t}")
                print(f"Act: {act}")
                print(f"Traj: {traj[-1]}")
                print(f"Next state: {next_state}")
            
            # If we have to normalize, modify this
            traj.append(next_state)
            log_probs.append(log_prob)
            acts.append(act)
            c_masks.append(c_mask)

        log_probs = torch.stack(log_probs, dim=1).unsqueeze(-1)
        
        v = [self.critic_copy(traj[-1])]
        for t in range(len(traj)-3, -1, -1):
            v_t = self.world_model.compute_reward(traj[t], self.target_state, acts[t]).unsqueeze(-1) + \
                    self.config.critic_model.discount_factor * (((1 - self.config.critic_model.lambda_val) * self.critic_copy(traj[t+1]) + \
                    self.config.critic_model.lambda_val * v[-1]))
            
            # print(f"State at t={t}: {traj[t]}")
            
            masked_v = (c_masks[t] * v_t) + (1-c_masks[t]) * -100
            # print(f"Masked V: {masked_v}")
            v.append(masked_v)
        v.reverse()
        lambda_val = torch.stack(v, dim=1)
        
        critic_val = torch.stack([self.critic(traj[x]) for x in range(len(traj)-1)], dim=1)
        critic_loss = F.smooth_l1_loss(critic_val, lambda_val.detach(), reduction='mean')
        
        critic_loss.backward(inputs=[param for param in self.critic.parameters()])
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 100)

        actor_loss = -torch.mean(torch.sum(log_probs * (lambda_val - critic_val).detach(), dim=1))

        entropy_pen = (self.config.behavior_learning.nu*log_probs).sum(1).mean()

        total_actor_loss = actor_loss + entropy_pen

        total_actor_loss.backward(inputs=[param for param in self.actor.parameters()])
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        if self.beh_updates % 25 == 0:
            self.writer.add_scalar("Behavior/critic_loss", critic_loss, self.beh_updates)
            self.writer.add_scalar("Behavior/actor_loss", actor_loss, self.beh_updates)
            self.writer.add_scalar("Behavior/entropy_pen", entropy_pen, self.beh_updates)
            self.writer.add_scalar("Behavior/total_actor_loss", total_actor_loss, self.beh_updates)

            critic_grad = torch.norm(torch.cat([param.grad.flatten() for param in self.critic.parameters()]))
            actor_grad = torch.norm(torch.cat([param.grad.flatten() for param in self.actor.parameters()]))

            self.writer.add_scalar("Behavior/critic_grad", critic_grad, self.beh_updates)
            self.writer.add_scalar("Behavior/actor_grad", actor_grad, self.beh_updates)

            total_reward = sum([self.world_model.compute_reward(traj[t], self.target_state, act).mean() for t, act in enumerate(acts)])
            self.writer.add_scalar('Reward/total', total_reward, self.beh_updates)

        if self.beh_updates % self.config.critic_model.hard_update == 0:
            print("Hard updating critic!")
            hard_update(self.critic_copy, self.critic)
            print("Done hard updating")

        self.beh_updates += 1

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.wm_optimizer.zero_grad()

    def wm_train_step(self):
        num_rollout = 32
        dts, states, actions = self.buffer.sample(self.config.training.batch_size, self.history+num_rollout)
        
        pred_traj = self.world_model.rollout(dts, states, actions, num_rollout)

        # loss = self.model.loss(
        #     torch.concat((pred_traj[:,1:,3:6], pred_traj[:,1:,9:12]), dim=-1), 
        #     torch.concat((states[:,1:,3:6], states[:,1:, 9:12]), dim=-1)
        # )

        loss = self.world_model.loss(
            pred_traj,
            states[:,self.history+2:,:]
        )

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=50.0)
        self.wm_optimizer.step()
        # self.seq_scheduler.step(loss.item())

        if self.batch_count % 25 == 0:
            self.validate()
            grad_norm = self.compute_gradient_norm()
            self.writer.add_scalar("Norms/gradient_norm", grad_norm, self.batch_count)
            weight_norm = self.compute_weight_norm()
            self.writer.add_scalar("Norms/weight_norm", weight_norm, self.batch_count)
            self.writer.add_scalar("Loss/train", loss, self.batch_count)
        
        self.batch_count += 1

        self.wm_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        

def wm_train_process_fn(buffer, actor, config_file, stop_event):
    try:
        num_updates = 0
        learner = Learner(buffer, actor, config_file)
        first = True
        while not stop_event.is_set():
            if learner.buffer.get_len() > learner.config.replay_buffer.start_learning:
                # if first:
                #     norm_stats = buffer.compute_normalization_stats()
                #     buffer.normalize_buffer_data()
                #     first = False
                learner.wm_train_step()
                # if num_updates > learner.config.behavior_learning.start_point:
                #     print("behavior step!")
                #     learner.beh_train_step()
                # num_updates += 1
            else:
                print(f"Not enough data yet: {learner.buffer.get_len()}")
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("Training process interrupted")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        if learner:
            learner.validate(save_to_table=True)
            print("Saving model...")
            # learner.save_wm(norm_stats)
            learner.save_wm()
            if 'learner' in locals():
                learner.close_writer()

def main(args=None) -> None:
    print('Waiting 5 seconds before starting...')
    time.sleep(7)
    print('Starting storage node...')
    
    rclpy.init(args=args)
    start_time = time.time()

    storage_node = Storage(None, None)
    config_file = storage_node.config_file

    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)

    config = AttrDict.from_dict(config_dict)
    
    timeout = config.timeout

    buffer = ReplayBuffer(config)
    actor = Actor(config).to(device=config.device)

    storage_node.buffer = buffer
    storage_node.actor = actor

    mp.set_start_method('spawn', force=True)

    ctx = mp.get_context('spawn')
    stop_event = ctx.Event()

    train_process = mp.Process(
        target=wm_train_process_fn, 
        args=[buffer, actor, config_file, stop_event]
    )
    train_process.start()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(storage_node)
            time.sleep(0.001)

            if time.time() - start_time > timeout:
                stop_event.set()
                print("\nTimeout reached! Shutting down.")
                break
    except KeyboardInterrupt:
        stop_event.set()
        print("keyboard interrupt")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        print("Waiting for training process to finish cleanup (max 30 seconds)...")
        train_process.join(timeout=60)  # Wait for graceful shutdown
        
        if train_process.is_alive():
            print("Training process didn't exit gracefully, forcing termination...")
            train_process.terminate()
            train_process.join(timeout=5)  # Wait for forced termination
            
        storage_node.destroy_node()
        rclpy.shutdown()
        
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
