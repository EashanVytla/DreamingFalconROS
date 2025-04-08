import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F
import math

def quat_to_euler(q: np.array, device) -> torch.Tensor:
    # Create rotation object and get euler angles
    rot = R.from_quat(q, scalar_first=True)
    euler = rot.as_euler('xyz', degrees=False)  # Get angles in radians
    
    # Convert back to torch tensor
    return torch.tensor(euler, device=device, dtype=torch.float32)

def l2_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def diff_flag(x, thresholds, beta=10.0):
    """
    x: tensor of shape (batch_size, num_features)
    thresholds: tensor of shape (num_features,) with the threshold for each feature
    beta: sharpness parameter; higher beta -> closer to hard max
    1 is continue
    0 is terminate
    """
    # Compute the difference between features and their thresholds
    diff = x - thresholds  # shape: (batch_size, num_features)
    
    # Apply log-sum-exp along the features dimension as a smooth max
    smooth_max = (1.0 / beta) * torch.log(torch.sum(torch.exp(beta * diff), dim=1, keepdim=True))
    
    # Use a sigmoid to get an output between 0 and 1
    flag = F.softplus(smooth_max)
    return flag

def check_components_over_threshold(x, thresholds):
    """
    Checks which components of the state exceed their thresholds
    
    Args:
        x: tensor of shape (batch_size, num_features)
        thresholds: tensor of shape (num_features,) with thresholds for each feature
        
    Returns:
        Dictionary with boolean flags for each component and an overall termination flag
        1 means safe (continue), 0 means unsafe (terminate)
    """
    batch_size = x.shape[0]
    
    # Split into components (each shaped [batch_size, 3])
    pos = x[:, 0:3] 
    vel = x[:, 3:6]
    att = x[:, 6:9]
    ang_vel = x[:, 9:12]
    
    # Get corresponding thresholds
    pos_thresh = thresholds[0:3]
    vel_thresh = thresholds[3:6]
    att_thresh = thresholds[6:9]
    ang_vel_thresh = thresholds[9:12]
    
    # Calculate max value for each component
    pos_max = torch.max(torch.abs(pos), dim=1)[0]  # [batch_size, 3] -> [batch_size]
    vel_max = torch.max(torch.abs(vel), dim=1)[0]
    att_max = torch.max(torch.abs(att), dim=1)[0]
    ang_vel_max = torch.max(torch.abs(ang_vel), dim=1)[0]
    
    # Calculate threshold for each component (max of each threshold group)
    pos_thresh_max = torch.max(pos_thresh)
    vel_thresh_max = torch.max(vel_thresh)
    att_thresh_max = torch.max(att_thresh)
    ang_vel_thresh_max = torch.max(ang_vel_thresh)
    
    # Check if z-position is below 0 (unsafe)
    z_unsafe = (pos[:, 2] > 0).any()
    
    # Determine if each component exceeds threshold (1 = safe, 0 = terminate)
    # Position is safe if below max threshold AND z is not negative
    pos_max_safe = (torch.max(pos_max) <= pos_thresh_max).float()
    pos_safe = (pos_max_safe * (~z_unsafe).float()).float()
    
    vel_safe = (torch.max(vel_max) <= vel_thresh_max).float()
    att_safe = (torch.max(att_max) <= att_thresh_max).float()
    ang_vel_safe = (torch.max(ang_vel_max) <= ang_vel_thresh_max).float()
    
    # Overall termination flag (1 = continue, 0 = terminate)
    continue_flag = (pos_safe * vel_safe * att_safe * ang_vel_safe).float()
    
    # Return dictionary with component-wise and overall flags
    return {
        'position_safe': pos_safe.item(),
        'z_above_ground': (~z_unsafe).float().item(),  # New field showing if z ≥ 0
        'velocity_safe': vel_safe.item(),
        'attitude_safe': att_safe.item(),
        'angular_velocity_safe': ang_vel_safe.item(),
        'continue': continue_flag.item(),
        'component_max_values': {
            'position_max': torch.max(pos_max).item(),
            'z_min': torch.min(pos[:, 2]).item(),  # Added minimum z value
            'velocity_max': torch.max(vel_max).item(),
            'attitude_max': torch.max(att_max).item(),
            'angular_velocity_max': torch.max(ang_vel_max).item()
        },
        'component_thresholds': {
            'position_threshold': pos_thresh_max.item(),
            'velocity_threshold': vel_thresh_max.item(),
            'attitude_threshold': att_thresh_max.item(),
            'angular_velocity_threshold': ang_vel_thresh_max.item()
        }
    }

def flag(x, thresholds):
    """
    Determines if any feature exceeds its threshold or if z is below 0
    
    Args:
        x: tensor of shape (batch_size, num_features)
        thresholds: tensor of shape (num_features,) 
        
    Returns:
        Tensor of shape (batch_size, 1) where
        1 = continue (safe), 0 = terminate (unsafe)
    """
    # Check if any value exceeds its threshold
    over_threshold = torch.any(x > thresholds.unsqueeze(0), dim=1, keepdim=True)
    
    # Check if z-position is negative (each batch item separately)
    z_negative = (x[:, 2:3] > 0)
    
    # Combine both conditions (either over threshold or z negative)
    unsafe = over_threshold | z_negative
    
    # Return 1 if safe, 0 if unsafe
    return (~unsafe).float()

def denormalize(val, min, max):
    return ((val + 1) / 2) * (max - min) + min


def unwrap_angle(new_angle, prev_angle):
    delta = new_angle - prev_angle
    if delta > math.pi:
        new_angle -= 2 * math.pi
    elif delta < -math.pi:
        new_angle += 2 * math.pi
    return new_angle

# --- Start of code adapted from: Physics Informed Model Based RL
# Author: Adithya Ramesh
# Date: May 14 2023
# Source: https://github.com/adi3e08/Physics_Informed_Model_Based_RL/blob/main/models/mbrl.py ---
def hard_update(target, source):
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
# --- End of adapted code ---

def get_DCM(phi: torch.Tensor, theta: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    phi = torch.as_tensor(phi, dtype=torch.float32)
    theta = torch.as_tensor(theta, dtype=torch.float32)
    psi = torch.as_tensor(psi, dtype=torch.float32)
    
    c_phi = torch.cos(phi)
    s_phi = torch.sin(phi)
    c_theta = torch.cos(theta)
    s_theta = torch.sin(theta)
    c_psi = torch.cos(psi)
    s_psi = torch.sin(psi)
    
    m11 = c_theta * c_psi
    m12 = s_phi * s_theta * c_psi - c_phi * s_psi
    m13 = c_phi * s_theta * c_psi + s_phi * s_psi
    
    m21 = c_theta * s_psi
    m22 = s_phi * s_theta * s_psi + c_phi * c_psi
    m23 = c_phi * s_theta * s_psi - s_phi * c_psi
    
    m31 = -s_theta
    m32 = s_phi * c_theta
    m33 = c_phi * c_theta
    
    row1 = torch.stack([m11, m12, m13], dim=-1)
    row2 = torch.stack([m21, m22, m23], dim=-1)
    row3 = torch.stack([m31, m32, m33], dim=-1)
    
    L_EB = torch.stack([row1, row2, row3], dim=-2)
    
    return L_EB

def normalize(val, min, max):
    return 2 * (val - min) / (max - min) - 1

class AttrDict(dict):
    """Dictionary subclass that allows attribute access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_dict(data):
        """Recursively convert a dictionary into AttrDict."""
        if isinstance(data, dict):
            return AttrDict({key: AttrDict.from_dict(value) for key, value in data.items()})
        elif isinstance(data, list):
            return [AttrDict.from_dict(item) for item in data]
        else:
            return data