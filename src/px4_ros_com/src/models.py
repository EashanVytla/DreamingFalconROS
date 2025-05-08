import torch
import torch.nn as nn
import torch.optim as optim
import utils
import math
from rk4_solver import RK4_Solver
import torch.nn.functional as F
import copy
from torch import multiprocessing as mp

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', dropout_rate=0.0):
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
            
                        # Add dropout after activation (except for the last hidden layer)
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

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
        self.config = config

        ctx = mp.get_context('spawn')
        self.lock = ctx.Lock()

    def forward(self, x_t, std=0.6):
        with self.lock:
            out = x_t
            for layer in self.shared_layers:
                out = layer(out)

            if torch.isnan(out).any(): print("nan actor out")

            mu = self.mu(out)

            if torch.isnan(mu).any(): print("nan actor mu")
            
            dist = torch.distributions.Normal(mu, std)
            x_t = dist.rsample()
            action = torch.tanh(x_t)

            if torch.isnan(action).any(): print("nan actor action")

            log_prob = dist.log_prob(x_t).sum(1)
            log_prob -= torch.log(torch.clamp(1-action.pow(2), min=1e-6)).sum(1)
        
            return action, log_prob

class Critic(nn.Module):
    def __init__(self, config, device):
        super(Critic, self).__init__()

        hidden_dims = [config.critic_model.hidden_size] * config.critic_model.hidden_layers
        self.critic = MLP(config.critic_model.input_dim, hidden_dims, config.critic_model.output_dim, config.critic_model.activation).to(device=config.device)

    def forward(self, x_t):
        return self.critic(x_t)
    
class InitializerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_layers, hidden_size, dropout_rate=0.0):
        """
        Args:
            input_dim (int): Dimension of the input features (e.g., concatenated state and action).
            hidden_dims (list of int): List of hidden layer sizes for the MLP.
            num_layers (int): Number of LSTM layers.
            hidden_size (int): Hidden state dimension for the LSTM.
        """
        super(InitializerMLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(current_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hdim

        # Final layer outputs a vector of size num_layers * 2 * hidden_size.
        self.mlp = nn.Sequential(*layers, nn.Linear(current_dim, num_layers * 2 * hidden_size))
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch, input_dim).
        
        Returns:
            tuple: (h0, c0) where each is of shape (num_layers, batch, hidden_size).
        """
        # x shape: (batch, input_dim)
        out = self.mlp(x)  # shape: (batch, num_layers * 2 * hidden_size)
        
        # Reshape to (batch, num_layers, 2 * hidden_size)
        out = out.view(x.size(0), self.num_layers, 2 * self.hidden_size)
        
        # Split the last dimension into two: one for h0 and one for c0.
        h0, c0 = out.split(self.hidden_size, dim=-1)  # each is (batch, num_layers, hidden_size)
        
        # Transpose to get shape (num_layers, batch, hidden_size)
        h0 = h0.transpose(0, 1).contiguous()
        c0 = c0.transpose(0, 1).contiguous()
        
        return h0, c0

class Cell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.0):
        """
        Initializes the SimpleGRU model.

        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state.
            output_size (int): Number of output features.
            num_layers (int): Number of stacked GRU layers.
        """
        super(Cell, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create a GRU layer; `batch_first=True` expects input of shape (batch, seq_len, input_size)
        self.cells = nn.RNN(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # A fully-connected layer to map the hidden state at the final time step to the output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass for the LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_size)
            hidden (tuple, optional): Tuple of initial hidden state and cell state,
                each with shape (num_layers, batch, hidden_size). If None, they are initialized to zeros.
        
        Returns:
            out (torch.Tensor): Output tensor of shape (batch, output_size)
            (hn, cn) (tuple): The hidden and cell states from the final time step,
                each with shape (num_layers, batch, hidden_size)
        """
        # Initialize hidden and cell states if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
            # hidden = (h0, c0)
        
        # LSTM forward pass.
        # out: tensor with shape (batch, seq_len, hidden_size)
        # (hn, cn): each with shape (num_layers, batch, hidden_size)
        out, hidden = self.cells(x, hidden)
        
        # Use the output from the final time step for prediction
        last_time_step = out[:, -1, :]  # shape: (batch, hidden_size)
        out = self.fc(last_time_step)   # shape: (batch, output_size)
        
        return out, hidden

class WorldModelRNN(nn.Module):
    def __init__(self, config, device):
        super(WorldModelRNN, self).__init__()
        self.num_layers = 8
        self.config = config
        self.hidden_size = 256

        self.rnn = Cell(
            input_size=14, 
            hidden_size=self.hidden_size, 
            output_size=9, 
            num_layers=self.num_layers,
            dropout_rate=0.1
        )

        # self.h0_encoder = InitializerMLP(
        #     input_dim=13 * self.config.rnn_model.history,
        #     hidden_dims=[15000],
        #     num_layers=num_layers,
        #     hidden_size=256,
        #     dropout_rate=0.1
        # )

        self.h0_encoder = nn.RNN(
            14,
            256,
            8,
            batch_first=True,
            dropout=0.1,
            device=self.config.device
        )

        self.dt = config.physics.refresh_rate

    def loss(self, pred, truth):
        """
        Compute weighted Huber loss between predictions and ground truth.
        Weights are defined internally based on the importance of different state dimensions.
        
        Args:
            pred: Predicted states [batch_size, state_dim]
            truth: Ground truth states [batch_size, state_dim]
        
        Returns:
            Weighted loss value (scalar)
        """
        B, T, D = pred.shape
        device = pred.device
        dtype  = pred.dtype
        weights = torch.linspace(T, 1, T, device=device, dtype=dtype)
        weights = weights.view(1, T, 1)
        per_elem = F.mse_loss(pred, truth, reduction='none')
        weighted = per_elem * weights
        return weighted.mean()
    
    def rollout(self, dts, states, acts, num_rollout):
        hist = states.shape[1] - num_rollout

        if hist != self.config.rnn_model.history:
            print(f"History length didn't match. hist={hist} config_hist={self.config.rnn_model.history}")
            return

        traj = []
        # hn = self.h0_encoder(torch.cat((states[:,0,:], acts[:,0,:]), dim=-1)).unsqueeze(0)
        
        # For MLP
        # inp = torch.cat((states[:,:hist,3:], acts[:,:hist,:]), dim=-1).flatten(start_dim=1)
        
        # For RNN
        inp = torch.cat((
                    dts[:, 1:hist+1].unsqueeze(-1),
                    states[:,:hist,3:],
                    acts[:, :hist, :]
                ), dim=-1)
        # h0, c0 = self.h0_encoder(inp)

        out, hn = self.h0_encoder(inp)

        pred_state = None

        for i in range(hist, hist + num_rollout - 1):
            inp = torch.cat((
                    dts[:, i+1:i+2].unsqueeze(-1),
                    pred_state.unsqueeze(1) if pred_state != None else states[:,hist:hist+1,3:],
                    acts[:, i:i+1, :]
                ), dim=-1)
            out, hn = self.rnn(inp, hn)
            pred_state = out
            traj.append(out)
        
        return torch.stack(traj, dim=1)


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

        self.model = MLP(config.force_model.input_dim, hidden_dims, config.force_model.output_dim, config.force_model.activation, config.force_model.dropout_rate)
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

    def compute_normalization_stats(self, states, actions):
        self.states_mean = states.mean(dim=0)
        self.states_std = states.std(dim=0) + self.epsilon
        
        self.actions_mean = actions.mean(dim=0)
        self.actions_std = actions.std(dim=0) + self.epsilon

        self.states_mean = self.states_mean.to(device=self.device)
        self.states_std = self.states_std.to(device=self.device)
        self.actions_mean = self.actions_mean.to(device=self.device)
        self.actions_std = self.actions_std.to(device=self.device)

        print(f"Data statistics: ")
        print(f"States mean: {self.states_mean}")
        print(f"States std: {self.states_std}")
        print(f"Actions mean: {self.actions_mean}")
        print(f"Actions std: {self.actions_std}")

    def compute_reward(self, state, target_state, control_input):
        vel_rew = torch.norm((state[:,3:6] - target_state[3:6]).unsqueeze(0))
        energy_rew = control_input.sum(dim=-1)
        stab_rew = torch.norm(state[:,9:12] - target_state[9:12].unsqueeze(0))
        
        w_v = 0.01
        w_e = 0.001
        w_s = 0.001
        
        # Base reward: negative of weighted error terms
        reward = -(w_v * vel_rew + w_e * energy_rew + w_s * stab_rew)
        
        return reward

    def rollout(self, dts, x_t, x_tm1, act_inps):
        x_roll = [x_t]
        forces_roll = []
        prev_x = None
        seq_len = act_inps.shape[1]

        prev_forces = self.invert_forces_and_moments(x_tm1, x_t, dts[:,0].unsqueeze(-1))

        for i in range(1, seq_len):
            pred, forces = self.predict(dts[:, i].unsqueeze(-1), x_roll[i-1], act_inps[:, i-1, :], prev_forces=prev_forces)

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
            prev_forces = forces
            x_roll.append(pred)

        stacked = torch.stack(x_roll, dim=1)
        return stacked
    
    def invert_forces_and_moments(self, x_tm1, x_t, dt):
        """
        Given state at time t and t+1, recover applied forces and moments.
        
        x_t, x_tp1: tensors of shape [batch, 12] in the same ordering as your forward model.
        dt:       scalar timestep (float)
        mass:     scalar mass m
        I:        inertia tensor of shape [3,3]
        
        Returns:
        forces:  tensor [batch, 3]
        moments: tensor [batch, 3]
        """
        # split out velocities and angular rates
        V_t      = x_tm1[:, 3:6]      # [U, V, W] at t
        V_tp1    = x_t[:, 3:6]
        omega_t  = x_tm1[:, 9:12]     # [p, q, r] at t
        omega_tp1= x_t[:, 9:12]
        
        # 1) linear acceleration estimate
        dv_dt = (V_tp1 - V_t) / dt  # [batch,3]
        # cross product omega x V
        coriolis_lin = torch.cross(omega_t, V_t, dim=1)
        # recover force
        F = self._mass * (dv_dt + coriolis_lin)  # [batch,3]
        
        # 2) angular acceleration estimate
        domega_dt = (omega_tp1 - omega_t) / dt  # [batch,3]
        # I * domega_dt
        I_omega_dot = torch.matmul(domega_dt.unsqueeze(-2), self.I).squeeze(-2)
        # coriolis_rot = omega x (I omega)
        I_omega = torch.matmul(omega_t.unsqueeze(-2), self.I).squeeze(-2)
        coriolis_rot = torch.cross(omega_t, I_omega, dim=1)
        # recover moment
        M = I_omega_dot + coriolis_rot       # [batch,3]
        
        return torch.cat((F, M), dim=1)

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

    def predict(self, dt, x_t, actuator_input, prev_forces: torch.Tensor):
        '''
        input vector: Velocity: U, v, w (0:3)
                        Euler Rotation Angles: phi, theta, psi (3:6)
                        Body Rotation Rates: p, q, r (6:9)
        '''
        # inp = torch.cat((actuator_input, x_t[:, 3:12]), dim=1)
        inp = torch.cat((actuator_input, prev_forces), dim=1)
        forces_norm = torch.tanh(self.model(inp))

        if self.norm_config.norm:
            x_t_dn = torch.zeros_like(x_t)
            x_t_dn[:, 0:3] = x_t[:, 0:3]
            x_t_dn[:, 3:6] = utils.denormalize(x_t[:, 3:6], self.norm_config.velo_min, self.norm_config.velo_max)
            x_t_dn[:, 6:9] = utils.denormalize(x_t[:, 6:9], self.norm_config.euler_min, self.norm_config.euler_max)
            x_t_dn[:, 9:12] = utils.denormalize(x_t[:, 9:12], self.norm_config.omega_min, self.norm_config.omega_max)

        # forces = torch.zeros_like(forces_norm)
        # forces[:, 0:2] = utils.denormalize(forces_norm[:, 0:2], self.norm_config.fxy_min, self.norm_config.fxy_max)
        # forces[:, 2] = utils.denormalize(forces_norm[:, 2], self.norm_config.fz_min, self.norm_config.fz_max)
        # forces[:, 3:5] = utils.denormalize(forces_norm[:, 3:5], self.norm_config.mxy_min, self.norm_config.mxy_max)
        # forces[:, 5] = utils.denormalize(forces_norm[:, 5], self.norm_config.mz_min, self.norm_config.mz_max)

        forces_norm = forces_norm + prev_forces

        # print(torch.max(forces, dim=0))

        return self.six_dof(x_t_dn if self.norm_config.norm else x_t, dt, forces_norm), forces_norm

    def loss(self, pred, truth):
        """
        Compute weighted Huber loss between predictions and ground truth.
        Weights are defined internally based on the importance of different state dimensions.
        
        Args:
            pred: Predicted states [batch_size, state_dim]
            truth: Ground truth states [batch_size, state_dim]
        
        Returns:
            Weighted loss value (scalar)
        """
        # Define weights for different state components
        # Format: [position(x,y,z), velocity(u,v,w), attitude(phi,theta,psi), angular_vel(p,q,r)]
        weights = torch.tensor([
            0.1, 0.1, 0.1,     # Position (xyz)
            0.1, 0.1, 0.1,        # Velocity (uvw)
            1.0, 1.0, 1.0,        # Attitude (euler angles)
            5.0, 5.0, 5.0         # Angular velocity
        ], device=pred.device)
        
        # Ensure weights has the right shape for broadcasting
        weights = weights.view(1, -1)
        
        # Compute element-wise Huber loss
        huber_loss = F.smooth_l1_loss(pred, truth, reduction='none', beta=self._beta)
        
        # Apply weights
        weighted_loss = huber_loss * weights
        
        # Return mean of weighted loss normalized by sum of weights
        return torch.sum(weighted_loss) / torch.sum(weights)