import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

def quat_to_euler(q: np.array, device) -> torch.Tensor:
    # Create rotation object and get euler angles
    rot = R.from_quat(q, scalar_first=True)
    euler = rot.as_euler('xyz', degrees=False)  # Get angles in radians
    
    # Convert back to torch tensor
    return torch.tensor(euler, device=device, dtype=torch.float32)

def l2_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

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