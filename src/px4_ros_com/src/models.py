import torch
import torch.nn as nn
import torch.optim as optim
import utils
import math
from rk4_solver import RK4_Solver
import torch.nn.functional as F
import copy

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super(MLP, self).__init__()
        
        layers = []
        
        hidden_dims = [input_dim] + hidden_dims
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            elif activation == 'selu':
                layers.append(nn.SELU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'softplus':
                layers.append(nn.Softplus())
            else:
                print("Don't know that activation! Defaulting to RELU.")
                layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
class Actor(torch.nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()

        hidden_dims = [config.actor_model.hidden_size] * config.actor_model.hidden_layers

        layers = []
        dims = [config.actor_model.input_dim] + hidden_dims
        for i in range(1, len(dims)):
            layers.append(nn.Linear(dims[i-1], dims[i]))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.ModuleList(layers)
        self.mu = torch.nn.Linear(dims[-1], config.actor_model.output_dim)
        self.log_sigma = torch.nn.Linear(dims[-1], config.actor_model.output_dim)

    def forward(self, x_t, deterministic=False, with_logprob=False):
        out = x_t
        for layer in self.shared_layers:
            out = layer(out)

        # --- Start of code adapted from: Physics Informed Model Based RL
        # Author: Adithya Ramesh
        # Date: May 14 2023
        # Source: https://github.com/adi3e08/Physics_Informed_Model_Based_RL/blob/main/models/mbrl.py ---

        mu = self.mu(out)
        log_sigma = self.log_sigma(out)
        
        log_sigma = torch.clamp(log_sigma, min=-20.0, max=2.0)
        sigma = torch.exp(log_sigma)
        
        if deterministic:
            action = torch.tanh(mu)
            return action, None
        
        dist = torch.distributions.Normal(mu, sigma)
        x_t = dist.rsample()
        action = torch.tanh(x_t)

        if with_logprob:
            log_prob = dist.log_prob(x_t).sum(1)
            log_prob -= torch.log(torch.clamp(1-action.pow(2), min=1e-6)).sum(1)
        else:
            log_prob = None
    
        return action, log_prob
    
        # --- End of adapted code ---
    
class Critic(nn.Module):
    def __init__(self, config, device):
        super(Critic, self).__init__()

        hidden_dims = [config.critic_model.hidden_size] * config.critic_model.hidden_layers
        self.critic = MLP(config.critic_model.input_dim, hidden_dims, config.critic_model.output_dim, config.critic_model.activation).to(device=config.device)

    def loss():
        pass

class WorldModel(nn.Module):
    def __init__(self, config, device):
        super(WorldModel, self).__init__()
        # initialize the MLP
        hidden_dims = [config.force_model.hidden_size] * config.force_model.hidden_layers
        self.norm_config = config.normalization

        self._rate = config.physics.refresh_rate
        self._mass = config.physics.mass
        self._g = config.physics.g
        self._loss_scaler = config.training.loss_scaler
        self.epsilon = 1e-7
        self._beta = config.training.beta

        self.I = torch.tensor([[config.physics.I_xx, config.physics.I_xy, config.physics.I_xz],
                              [config.physics.I_yx, config.physics.I_yy, config.physics.I_yz],
                              [config.physics.I_zx, config.physics.I_zy, config.physics.I_zz]], device=device, dtype=torch.float32)
        
        self.I_inv = torch.inverse(self.I)

        self.model = MLP(config.force_model.input_dim, hidden_dims, config.force_model.output_dim, config.force_model.activation)
        # self.init_weights()
        self.device = device

        self.solver = RK4_Solver()

    def init_weights(self):
        def xavier_init(model):
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def uniform_init(model):
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        self.model.apply(uniform_init)

    def compute_normalization_stats(self, dataloader):
        states_list = []
        actions_list = []
        forces_list = []

        for states, actions, forces in dataloader:
            states_list.append(states)
            actions_list.append(actions)
            forces_list.append(forces)
        
        all_states = torch.cat(states_list, dim=0)
        all_actions = torch.cat(actions_list, dim=0)
        all_forces = torch.cat(forces_list, dim=0)
        
        self.states_mean = all_states.mean(dim=0)
        self.states_std = all_states.std(dim=0) + self.epsilon
        
        self.actions_mean = all_actions.mean(dim=0)
        self.actions_std = all_actions.std(dim=0) + self.epsilon

        self.forces_mean = all_forces.mean(dim=0)
        self.forces_std = all_forces.std(dim=0) + self.epsilon

        self.states_mean = self.states_mean.to(device=self.device)
        self.states_std = self.states_std.to(device=self.device)
        self.actions_mean = self.actions_mean.to(device=self.device)
        self.actions_std = self.actions_std.to(device=self.device)
        self.forces_mean = self.forces_mean.to(device=self.device)
        self.forces_std = self.forces_std.to(device=self.device)

        print(f"Data statistics: ")
        print(f"States mean: {self.states_mean}")
        print(f"States std: {self.states_std}")
        print(f"Actions mean: {self.actions_mean}")
        print(f"Actions std: {self.actions_std}")
        print(f"forces mean: {self.forces_mean}")
        print(f"forces std: {self.forces_std}")

    def compute_reward(state, target_state, control_input, tolerance):
        pos_error = torch.norm(state[:,0:3] - target_state[:,0:3])
        vel_error = torch.norm(state[:,3:6] - target_state[:,3:6])
        att_error = torch.norm(state[:,6:9] - target_state[:,6:9])
        
        w_p = 1.0   # weight for position error
        w_v = 0.5   # weight for velocity error
        w_a = 0.5   # weight for attitude error
        w_c = 0.1   # weight for control effort
        
        # Base reward: negative of weighted error terms
        reward = -(w_p * pos_error + w_v * vel_error + w_a * att_error)
        
        # Penalize high control inputs
        control_penalty = w_c * torch.norm(control_input)
        reward -= control_penalty
        
        # Bonus if within a tolerance threshold (stability bonus)
        if pos_error < tolerance and att_error < tolerance:
            reward += 1.0
        
        return reward

    def rollout(self, dts, x_t, act_inps):
        x_roll = [x_t]
        forces_roll = []
        prev_x = None
        seq_len = act_inps.shape[1]
        for i in range(1, seq_len):
            pred = self.predict(dts[:, i].unsqueeze(-1), x_roll[i-1], act_inps[:, i-1, :])

            # if self.norm_ranges.norm:
            #     pred_norm = torch.zeros((pred.shape[0], pred.shape[1], 6), dtype=torch.float32, device=self.device)
            #     pred_norm[:, 0:3] = utils.normalize(pred[:, :, 3:6], self.norm_ranges.velo_min, self.norm_ranges.velo_max)
            #     pred_norm[:, 3:6] = utils.normalize(pred[:, :, 6:9], self.norm_ranges.omega_min, self.norm_ranges.omega_max)
            #     pred_norm[:, 6:9] = utils.normalize(pred[:, :, 9:12], self.norm_ranges.omega_min, self.norm_ranges.omega_max)
                
            if torch.max(pred).item() > 1000 or torch.min(pred).item() < -1000:
                print(f"Warning: Large values detected at step {i}: {torch.max(pred).item()}")

            if prev_x is not None:
                delta = torch.abs(pred - prev_x).max().item()
                if delta > 1000:
                    print(f"Warning: Large state change detected at step {i}, delta: {delta}")
            prev_x = pred
            x_roll.append(pred)

        stacked = torch.stack(x_roll, dim=1)
        return stacked

    def _compute_derivatives(self, x, forces):
        """
        Compute state derivatives for RK4 integration
        State vector: Position: Xe, Ye, Ze (0:3)
                        Velocity: U, v, w (3:6)
                        Euler Rotation Angles: phi, theta, psi (6:9)
                        Body Rotation Rates: p, q, r (9:12)
        """
        V = x[:, 3:6]
        phi = x[:, 6]
        theta = x[:, 7]
        psi = x[:, 8]
        omega = x[:, 9:12]

        F = forces[:, 0:3]
        M = forces[:, 3:6]
        
        # Initialize derivative vector
        dx = torch.zeros_like(x, device=self.device)
        
        # Compute derivatives using equations of motion
        # Position derivatives (Earth frame)
        dx[:, 0:3] = torch.matmul(utils.get_DCM(phi, theta, psi), V.unsqueeze(-1)).squeeze(-1)

        # Velocity derivative (Body frame)
        dx[:, 3:6] = F/self._mass - torch.cross(omega, V)

        # Rotation derivative (0 is phi, 1 is theta)
        J = torch.zeros((x.shape[0], 3, 3), device=self.device, dtype=torch.float32)
        J[:, 0, 0] = 1
        J[:, 0, 1] = torch.sin(phi) * torch.tan(theta)
        J[:, 0, 2] = torch.cos(phi) * torch.tan(theta)
        J[:, 1, 1] = torch.cos(phi)
        J[:, 1, 2] = -torch.sin(phi)
        J[:, 2, 1] = torch.sin(phi) / torch.clamp(torch.cos(theta), min=self.epsilon)
        J[:, 2, 2] = torch.cos(phi) / torch.clamp(torch.cos(theta), min=self.epsilon)

        dx[:, 6:9] = torch.matmul(J, omega.unsqueeze(-1)).squeeze(-1)

        # Rotation rate derivative (Body-fixed frame)
        Iw = torch.matmul(self.I, omega.unsqueeze(-1))
        coriolis = torch.cross(omega, Iw.squeeze(-1), dim=1)
        dx[:, 9:12] = torch.matmul(self.I_inv, (M - coriolis).unsqueeze(-1)).squeeze(-1)
        
        return dx
    
    def six_dof(self, x_t, dt, forces):
        return self.solver.step(x_t, self._compute_derivatives, dt, forces)

    def predict(self, dt, x_t, actuator_input):
        '''
        input vector: Velocity: U, v, w (0:3)
                        Euler Rotation Angles: phi, theta, psi (3:6)
                        Body Rotation Rates: p, q, r (6:9)
        '''
        inp = torch.cat((actuator_input, x_t[:, 3:12]), dim=1)
        forces_norm = self.model(inp)

        if self.norm_config.norm:
            x_t_dn = torch.zeros_like(x_t)
            x_t_dn[:, 0:3] = x_t[:, 0:3]
            x_t_dn[:, 3:6] = utils.denormalize(x_t[:, 3:6], self.norm_config.velo_min, self.norm_config.velo_max)
            x_t_dn[:, 6:9] = utils.denormalize(x_t[:, 6:9], self.norm_config.euler_min, self.norm_config.euler_max)
            x_t_dn[:, 9:12] = utils.denormalize(x_t[:, 9:12], self.norm_config.omega_min, self.norm_config.omega_max)

        forces = torch.zeros_like(forces_norm)
        forces[:, 0:2] = utils.denormalize(forces_norm[:, 0:2], self.norm_config.fxy_min, self.norm_config.fxy_max)
        forces[:, 2] = utils.denormalize(forces_norm[:, 2], self.norm_config.fz_min, self.norm_config.fz_max)
        forces[:, 3:5] = utils.denormalize(forces_norm[:, 3:5], self.norm_config.mxy_min, self.norm_config.mxy_max)
        forces[:, 5] = utils.denormalize(forces_norm[:, 5], self.norm_config.mz_min, self.norm_config.mz_max)

        # print(torch.max(forces, dim=0))

        return self.six_dof(x_t_dn if self.norm_config.norm else x_t, dt, forces)

    def loss(self, pred, truth):
        huber_loss = F.smooth_l1_loss(pred, truth, reduction='none', beta=self._beta)

        return torch.mean(huber_loss)